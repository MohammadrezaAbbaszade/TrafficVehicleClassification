"""Controlled ResNet18 Letterbox experiment on the existing private split.

Run the FC stage first, then the layer4+FC stage. This never replaces the
selected resnet18_final.pt checkpoint.
"""

from copy import deepcopy
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from scripts.resnet_preprocessing import make_resnet_transform
from scripts.run_experiments import VehicleDataset, device_for_run, manifest_data


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "checkpoints"
FEATURE_PATH = CHECKPOINT_DIR / "resnet18_letterbox_feature_best_loss.pt"
FINE_PATH = CHECKPOINT_DIR / "resnet18_letterbox_layer4_best_accuracy.pt"
SEED = 42
BATCH_SIZE = 32
PREPROCESSING = {
    "method": "letterbox",
    "resize": [224, 224],
    "fill": [0, 0, 0],
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(SEED)


def make_loaders():
    manifest, training_rows, validation_rows = manifest_data()
    transform = make_resnet_transform(PREPROCESSING)
    train_dataset = VehicleDataset(
        training_rows, transform, manifest["class_to_idx"]
    )
    val_dataset = VehicleDataset(
        validation_rows, transform, manifest["class_to_idx"]
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(SEED), num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )
    return manifest, train_loader, val_loader


def run_epoch(model, loader, loss_fn, device, optimizer=None, stage=None):
    training = optimizer is not None
    if training and stage == "feature":
        model.eval()  # The frozen backbone keeps its BatchNorm statistics.
        model.fc.train()
    elif training and stage == "fine_tune":
        model.train()
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            scores = model(images)
            loss = loss_fn(scores, labels)
            if training:
                loss.backward()
                optimizer.step()
            count = labels.size(0)
            total_loss += loss.item() * count
            correct += (scores.argmax(dim=1) == labels).sum().item()
            total += count
    return {"loss": total_loss / total, "accuracy": correct / total}


def train_stage(model, train_loader, val_loader, optimizer, device, stage, epochs,
                selection):
    loss_fn = nn.CrossEntropyLoss()
    history = []
    best_value = float("inf") if selection == "loss" else -1.0
    best_epoch = None
    best_state = None
    for epoch in range(1, epochs + 1):
        train = run_epoch(
            model, train_loader, loss_fn, device, optimizer=optimizer, stage=stage
        )
        validation = run_epoch(model, val_loader, loss_fn, device)
        row = {
            "epoch": epoch,
            "train_loss": train["loss"],
            "train_accuracy": train["accuracy"],
            "val_loss": validation["loss"],
            "val_accuracy": validation["accuracy"],
        }
        history.append(row)
        value = validation[selection]
        improved = value < best_value if selection == "loss" else value > best_value
        if improved:
            best_value = value
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
        print(
            f"{stage} | epoch {epoch:02d}/{epochs} | "
            f"train loss {train['loss']:.4f} | val loss {validation['loss']:.4f} | "
            f"train acc {train['accuracy']:.1%} | val acc {validation['accuracy']:.1%}",
            flush=True,
        )
    return history, best_epoch, best_state


def save_checkpoint(path, model_state, manifest, history, best_epoch,
                    training_mode, selection_metric, training_config):
    selected = history[best_epoch - 1]
    checkpoint = {
        "architecture": "resnet18",
        "training_mode": training_mode,
        "model_state_dict": {
            name: tensor.detach().cpu().clone()
            for name, tensor in model_state.items()
        },
        "class_to_idx": manifest["class_to_idx"],
        "split_fingerprint": manifest["fingerprint"],
        "preprocessing": deepcopy(PREPROCESSING),
        "training_config": training_config,
        "selected_epoch": best_epoch,
        "selection_metric": selection_metric,
        "validation_loss": selected["val_loss"],
        "validation_accuracy": selected["val_accuracy"],
        "history": history,
        # This threshold comes from the old model; recalibrate before promotion.
        "review_threshold": 0.90,
        "threshold_source": "baseline_reference_not_recalibrated",
    }
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    torch.save(checkpoint, path)
    print(
        f"Saved {path.name}: epoch {best_epoch}, "
        f"val loss {selected['val_loss']:.4f}, "
        f"val accuracy {selected['val_accuracy']:.1%}",
        flush=True,
    )
    return checkpoint


def run_feature_stage(epochs=30):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    set_seed()
    manifest, train_loader, val_loader = make_loaders()
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.requires_grad_(False)
    model.fc = nn.Linear(model.fc.in_features, len(manifest["class_to_idx"]))
    device = device_for_run()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
    print(
        f"Letterbox FC stage | device {device} | "
        f"train {len(train_loader.dataset)} | validation {len(val_loader.dataset)}",
        flush=True,
    )
    history, best_epoch, best_state = train_stage(
        model, train_loader, val_loader, optimizer, device,
        stage="feature", epochs=epochs, selection="loss",
    )
    return save_checkpoint(
        FEATURE_PATH, best_state, manifest, history, best_epoch, "fc_only",
        "validation_loss",
        {"seed": SEED, "optimizer": "Adam", "head_lr": 1e-3,
         "batch_size": BATCH_SIZE, "epochs": epochs},
    )


def run_fine_tune_stage(epochs=30):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if not FEATURE_PATH.is_file():
        raise FileNotFoundError(f"Run the Letterbox FC stage first: {FEATURE_PATH}")
    set_seed()
    manifest, train_loader, val_loader = make_loaders()
    feature = torch.load(FEATURE_PATH, map_location="cpu", weights_only=True)
    if feature["split_fingerprint"] != manifest["fingerprint"]:
        raise ValueError("The Letterbox feature checkpoint uses a different split")
    if feature["class_to_idx"] != manifest["class_to_idx"]:
        raise ValueError("The Letterbox feature checkpoint uses different classes")
    if feature["preprocessing"] != PREPROCESSING:
        raise ValueError("The feature checkpoint uses different preprocessing")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(manifest["class_to_idx"]))
    model.load_state_dict(feature["model_state_dict"])
    model.requires_grad_(False)
    model.layer4.requires_grad_(True)
    model.fc.requires_grad_(True)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.requires_grad_(False)
            module.eval()
    device = device_for_run()
    model = model.to(device)
    optimizer = torch.optim.Adam(
        [
            {"params": [p for p in model.layer4.parameters() if p.requires_grad],
             "lr": 1e-4},
            {"params": model.fc.parameters(), "lr": 1e-3},
        ],
        weight_decay=1e-4,
    )
    print(
        f"Letterbox layer4+FC stage | device {device} | "
        f"train {len(train_loader.dataset)} | validation {len(val_loader.dataset)}",
        flush=True,
    )
    history, best_epoch, best_state = train_stage(
        model, train_loader, val_loader, optimizer, device,
        stage="fine_tune", epochs=epochs, selection="accuracy",
    )
    return save_checkpoint(
        FINE_PATH, best_state, manifest, history, best_epoch,
        "layer4_plus_fc", "validation_accuracy",
        {"seed": SEED, "optimizer": "Adam", "layer4_lr": 1e-4,
         "head_lr": 1e-3, "weight_decay": 1e-4,
         "batch_size": BATCH_SIZE, "epochs": epochs,
         "feature_checkpoint": FEATURE_PATH.name},
    )
