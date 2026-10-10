"""Train a separate Letterbox ResNet18 with the four combined CNN settings.

The run has its own FC and layer4+FC checkpoints. It does not replace the
selected API checkpoint or the plain Letterbox experiment.
"""

from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from scripts.resnet_preprocessing import make_resnet_transform
from scripts.run_experiments import VehicleDataset, device_for_run, manifest_data
from scripts.run_resnet_letterbox import (
    BATCH_SIZE, CHECKPOINT_DIR, PREPROCESSING, SEED, run_epoch, set_seed,
)


COMBINED_FEATURE_PATH = CHECKPOINT_DIR / "resnet18_letterbox_combined_feature_best_loss.pt"
COMBINED_FINE_PATH = CHECKPOINT_DIR / "resnet18_letterbox_combined_layer4_best_accuracy.pt"
FLIP_P = 0.5
DROPOUT_P = 0.3
WEIGHT_DECAY = 1e-4
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 2


def make_loaders():
    manifest, training_rows, validation_rows = manifest_data()
    train_transform = make_resnet_transform(PREPROCESSING, train_flip_p=FLIP_P)
    val_transform = make_resnet_transform(PREPROCESSING)
    train_loader = DataLoader(
        VehicleDataset(training_rows, train_transform, manifest["class_to_idx"]),
        batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(SEED), num_workers=0,
    )
    val_loader = DataLoader(
        VehicleDataset(validation_rows, val_transform, manifest["class_to_idx"]),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )
    return manifest, train_loader, val_loader


def classifier(in_features, num_classes):
    return nn.Sequential(nn.Dropout(p=DROPOUT_P), nn.Linear(in_features, num_classes))


def train_stage(model, train_loader, val_loader, optimizer, device, stage, epochs,
                selection):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
    )
    loss_fn = nn.CrossEntropyLoss()
    history = []
    best_value = float("inf") if selection == "loss" else -1.0
    best_epoch = None
    best_state = None
    for epoch in range(1, epochs + 1):
        learning_rates = [group["lr"] for group in optimizer.param_groups]
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
            "learning_rates": learning_rates,
        }
        history.append(row)
        value = validation[selection]
        improved = value < best_value if selection == "loss" else value > best_value
        if improved:
            best_value = value
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
        scheduler.step(validation["loss"])
        print(
            f"combined {stage} | epoch {epoch:02d}/{epochs} | "
            f"train loss {train['loss']:.4f} | val loss {validation['loss']:.4f} | "
            f"train acc {train['accuracy']:.1%} | val acc {validation['accuracy']:.1%} | "
            f"lr {','.join(f'{rate:.0e}' for rate in learning_rates)}",
            flush=True,
        )
    return history, best_epoch, best_state


def save_checkpoint(path, model_state, manifest, history, best_epoch,
                    training_mode, selection_metric, epochs, feature_name=None):
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
        "classifier_dropout": DROPOUT_P,
        "training_augmentation": {"horizontal_flip_p": FLIP_P},
        "training_config": {
            "seed": SEED, "optimizer": "Adam", "batch_size": BATCH_SIZE,
            "epochs": epochs, "weight_decay": WEIGHT_DECAY,
            "layer4_lr": 1e-4 if training_mode == "layer4_plus_fc" else None,
            "head_lr": 1e-3,
            "scheduler": {
                "name": "ReduceLROnPlateau", "factor": SCHEDULER_FACTOR,
                "patience": SCHEDULER_PATIENCE, "monitor": "validation_loss",
            },
            "feature_checkpoint": feature_name,
        },
        "selected_epoch": best_epoch,
        "selection_metric": selection_metric,
        "validation_loss": selected["val_loss"],
        "validation_accuracy": selected["val_accuracy"],
        "history": history,
        # This is a baseline reference, not a calibrated combined-model value.
        "review_threshold": 0.90,
        "threshold_source": "baseline_reference_not_recalibrated",
    }
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)
    print(
        f"Saved {path.name}: epoch {best_epoch}, "
        f"val accuracy {selected['val_accuracy']:.1%}, "
        f"val loss {selected['val_loss']:.4f}",
        flush=True,
    )
    return checkpoint


def run_combined(feature_epochs=30, fine_epochs=30):
    """Run both stages from ImageNet weights and return the selected fine checkpoint."""
    if feature_epochs < 1 or fine_epochs < 1:
        raise ValueError("Both epoch counts must be positive")
    set_seed()
    manifest, train_loader, val_loader = make_loaders()
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.requires_grad_(False)
    model.fc = classifier(model.fc.in_features, len(manifest["class_to_idx"]))
    device = device_for_run()
    model = model.to(device)
    feature_optimizer = torch.optim.Adam(
        model.fc.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY
    )
    print(
        f"Combined Letterbox | device {device} | "
        f"train {len(train_loader.dataset)} | validation {len(val_loader.dataset)}",
        flush=True,
    )
    feature_history, feature_epoch, feature_state = train_stage(
        model, train_loader, val_loader, feature_optimizer, device,
        stage="feature", epochs=feature_epochs, selection="loss",
    )
    feature_checkpoint = save_checkpoint(
        COMBINED_FEATURE_PATH, feature_state, manifest, feature_history,
        feature_epoch, "fc_only", "validation_loss", feature_epochs,
    )

    # Restart the fine-tuning stage from the selected combined FC weights.
    set_seed()
    manifest, train_loader, val_loader = make_loaders()
    model = models.resnet18(weights=None)
    model.fc = classifier(model.fc.in_features, len(manifest["class_to_idx"]))
    model.load_state_dict(feature_checkpoint["model_state_dict"])
    model.requires_grad_(False)
    model.layer4.requires_grad_(True)
    model.fc.requires_grad_(True)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.requires_grad_(False)
            module.eval()
    model = model.to(device)
    fine_optimizer = torch.optim.Adam(
        [
            {"params": [p for p in model.layer4.parameters() if p.requires_grad],
             "lr": 1e-4},
            {"params": model.fc.parameters(), "lr": 1e-3},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    fine_history, fine_epoch, fine_state = train_stage(
        model, train_loader, val_loader, fine_optimizer, device,
        stage="fine_tune", epochs=fine_epochs, selection="accuracy",
    )
    return save_checkpoint(
        COMBINED_FINE_PATH, fine_state, manifest, fine_history, fine_epoch,
        "layer4_plus_fc", "validation_accuracy", fine_epochs,
        feature_name=COMBINED_FEATURE_PATH.name,
    )
