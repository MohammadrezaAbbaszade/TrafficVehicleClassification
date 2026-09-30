"""Reproducible vehicle-classification comparisons on the unified development split.

Run from the project root. Raw images and checkpoints stay local under .gitignore.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
BATCH_SIZE = 32
CLASS_LABELS = ('ambulance', 'autobus', 'kamyun', 'kamyunet', 'minibus', 'savari', 'taxi', 'vanet')


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def device_for_run():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def manifest_data():
    manifest = json.loads((ROOT / 'reports/base_split.json').read_text())
    if tuple(sorted(manifest['class_to_idx'], key=manifest['class_to_idx'].get)) != CLASS_LABELS:
        raise ValueError('Unexpected class mapping; recheck the dataset audit.')
    rows = {row['path']: row for row in manifest['samples']}
    train = [rows[path] for path in manifest['training_paths']]
    val = [rows[path] for path in manifest['validation_paths']]
    if manifest.get('schema_version') != 2:
        raise ValueError('Run scripts/data_split.py --write to build the unified split.')
    assert {r['content_hash'] for r in train}.isdisjoint({r['content_hash'] for r in val})
    assert {r['near_duplicate_group'] for r in train}.isdisjoint(
        {r['near_duplicate_group'] for r in val}
    )
    return manifest, train, val


def cnn_transform(augment=False):
    steps = [transforms.Lambda(lambda image: ImageOps.pad(
        image.convert('RGB'), (128, 128), method=Image.Resampling.BILINEAR,
        color=(0, 0, 0),
    ))]
    if augment:
        steps.append(transforms.RandomHorizontalFlip(p=0.5))
    steps += [transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)]
    return transforms.Compose(steps)


class VehicleDataset(Dataset):
    def __init__(self, rows, transform, class_to_idx):
        self.rows = rows
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(ROOT / row['path']) as image:
            prepared = self.transform(image.convert('RGB'))
        return prepared, self.class_to_idx[row['label']]


class VehicleCNN(nn.Module):
    def __init__(self, num_classes=8, pool='max', dropout=0.0):
        super().__init__()
        pool_layer = nn.MaxPool2d if pool == 'max' else nn.AvgPool2d
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), pool_layer(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), pool_layer(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1), nn.Dropout(dropout),
            nn.Linear(32 * 32 * 32, num_classes),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


class BalancedBatchSampler(Sampler[list[int]]):
    """Four examples of each of eight classes in every 32-image batch."""
    def __init__(self, labels, batch_size=BATCH_SIZE, seed=SEED):
        classes = sorted(set(labels))
        if len(classes) != 8 or batch_size % len(classes):
            raise ValueError('Expected eight classes and batch size divisible by eight.')
        self.buckets = {c: [i for i, label in enumerate(labels) if label == c] for c in classes}
        if any(not bucket for bucket in self.buckets.values()):
            raise ValueError('A class has no images.')
        self.per_class = batch_size // len(classes)
        self.num_batches = math.ceil(len(labels) / batch_size)
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(self.num_batches):
            batch = [rng.choice(self.buckets[c]) for c in sorted(self.buckets)
                     for _ in range(self.per_class)]
            rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


def simulated_imbalance(rows, class_to_idx):
    # Identical retained examples for shuffle and balanced-batch experiments.
    rng = random.Random(SEED)
    retained = []
    kept_positions = []
    for label in CLASS_LABELS:
        positions = [i for i, row in enumerate(rows) if row['label'] == label]
        if label in {'kamyun', 'kamyunet'}:
            positions = sorted(rng.sample(positions, max(1, len(positions)//4)))
        kept_positions.extend(positions)
    for i in sorted(kept_positions):
        retained.append(rows[i])
    private_file = ROOT / 'reports/imbalance_indices.json'
    private_file.write_text(json.dumps({
        'seed': SEED, 'original_training_positions': sorted(kept_positions),
        'retained_class_counts': dict(Counter(row['label'] for row in retained)),
    }, indent=2) + '\n')
    return retained


def make_loaders(manifest, train_rows, val_rows, experiment):
    train_transform = cnn_transform(augment=experiment in {'augment', 'regularized'})
    val_transform = cnn_transform()
    if experiment in {'imbalance_shuffle', 'balanced_batches'}:
        train_rows = simulated_imbalance(train_rows, manifest['class_to_idx'])
    train_dataset = VehicleDataset(train_rows, train_transform, manifest['class_to_idx'])
    val_dataset = VehicleDataset(val_rows, val_transform, manifest['class_to_idx'])
    if experiment == 'balanced_batches':
        labels = [manifest['class_to_idx'][r['label']] for r in train_rows]
        train_loader = DataLoader(train_dataset, batch_sampler=BalancedBatchSampler(labels), num_workers=0)
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, val_loader, train_rows


def make_model(experiment, device):
    model = VehicleCNN(pool='avg' if experiment == 'avg_pool' else 'max',
                       dropout=0.3 if experiment in {'dropout', 'regularized'} else 0.0)
    groups = [{'params': model.parameters(), 'lr': 1e-3}]
    return model.to(device), groups


def batch_loss(scores, labels, experiment):
    if experiment == 'bce':
        targets = F.one_hot(labels, num_classes=8).float()
        return F.binary_cross_entropy_with_logits(scores, targets)
    return F.cross_entropy(scores, labels)


def run_epoch(model, loader, optimizer, device, experiment):
    model.train()
    total_loss = correct = count = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        scores = model(images)
        loss = batch_loss(scores, labels, experiment)
        loss.backward()
        optimizer.step()
        count += len(labels)
        total_loss += loss.item() * len(labels)
        correct += (scores.argmax(dim=1) == labels).sum().item()
    return {'loss': total_loss/count, 'accuracy': correct/count}


@torch.no_grad()
def evaluate(model, loader, device, experiment, with_predictions=False):
    model.eval()
    total_loss = correct = count = 0
    truth, predictions, confidences = [], [], []
    for images, labels in loader:
        scores = model(images.to(device))
        labels = labels.to(device)
        loss = batch_loss(scores, labels, experiment)
        predicted = scores.argmax(dim=1)
        count += len(labels)
        total_loss += loss.item() * len(labels)
        correct += (predicted == labels).sum().item()
        if with_predictions:
            truth.extend(labels.cpu().tolist())
            predictions.extend(predicted.cpu().tolist())
            if experiment != 'bce':
                confidences.extend(torch.softmax(scores, dim=1).max(dim=1).values.cpu().tolist())
    metrics = {'loss': total_loss/count, 'accuracy': correct/count}
    if with_predictions:
        metrics['truth'] = truth
        metrics['predictions'] = predictions
        metrics['confidences'] = confidences
    return metrics


def summarise(truth, predictions):
    report = classification_report(truth, predictions, labels=list(range(8)),
                                   target_names=list(CLASS_LABELS), output_dict=True, zero_division=0)
    cm = confusion_matrix(truth, predictions, labels=list(range(8)))
    per_class = {name: {key: float(report[name][key]) for key in ('precision','recall','f1-score','support')}
                 for name in CLASS_LABELS}
    return {
        'accuracy': float(report['accuracy']),
        'macro_precision': float(report['macro avg']['precision']),
        'macro_recall': float(report['macro avg']['recall']),
        'macro_f1': float(report['macro avg']['f1-score']),
        'lowest_recall_class': min(CLASS_LABELS, key=lambda name: per_class[name]['recall']),
        'lowest_precision_class': min(CLASS_LABELS, key=lambda name: per_class[name]['precision']),
        'per_class': per_class,
        'confusion_counts': cm.tolist(),
    }


def run(experiment, epochs):
    set_seed()
    manifest, train_rows, val_rows = manifest_data()
    device = device_for_run()
    train_loader, val_loader, used_train_rows = make_loaders(manifest, train_rows, val_rows, experiment)
    model, groups = make_model(experiment, device)
    if experiment in {'adamw', 'regularized'}:
        optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(groups)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
                 if experiment in {'scheduler', 'regularized'} else None)
    print('experiment', experiment, 'device', device, 'train', len(used_train_rows),
          'validation', len(val_rows), 'epochs', epochs, flush=True)
    print('parameters total', sum(p.numel() for p in model.parameters()),
          'trainable', sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)
    history = []
    best_accuracy = -1.0
    best_state = None
    best_epoch = None
    for epoch in range(1, epochs+1):
        train = run_epoch(model, train_loader, optimizer, device, experiment)
        val = evaluate(model, val_loader, device, experiment)
        if val['accuracy'] > best_accuracy:
            best_accuracy = val['accuracy']
            best_epoch = epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        history.append({'epoch': epoch, 'train': train, 'validation': val,
                        'learning_rates': [group['lr'] for group in optimizer.param_groups]})
        if scheduler is not None:
            scheduler.step(val['loss'])
        print(f"epoch {epoch:02d}/{epochs} train={train['accuracy']:.1%} "
              f"val={val['accuracy']:.1%} val_loss={val['loss']:.4f} "
              f"lr={[g['lr'] for g in optimizer.param_groups]}", flush=True)
    model.load_state_dict(best_state)
    final = evaluate(model, val_loader, device, experiment, with_predictions=True)
    summary = summarise(final['truth'], final['predictions'])
    summary.update({
        'experiment': experiment, 'split_fingerprint': manifest['fingerprint'],
        'training_images': len(used_train_rows), 'validation_images': len(val_rows),
        'best_epoch': best_epoch, 'epochs': epochs,
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'history': history,
        'configuration': {
            'seed': SEED, 'batch_size': BATCH_SIZE,
            'loss': 'BCEWithLogitsLoss with one-hot float targets' if experiment=='bce' else 'CrossEntropyLoss',
            'preprocessing': 'RGB pad 128; ToTensor; Normalize(0.5,0.5)',
            'simulated_imbalance': experiment in {'imbalance_shuffle','balanced_batches'},
            'unclean_policy': manifest['unclean_label_policy'],
            'unclean_training_images': sum(row['source']=='unclean' for row in used_train_rows),
            'neysan_training_images': sum(row['source']=='neysan' for row in used_train_rows),
            'optimizer': type(optimizer).__name__,
        },
    })
    report_path = ROOT / 'reports/experiment_results.json'
    reports = json.loads(report_path.read_text()) if report_path.exists() else {}
    summary['result_status'] = 'current_split'
    reports['_split'] = {'current_fingerprint': manifest['fingerprint'],
                         'comparison_rule': 'Compare results only when split_fingerprint matches.'}
    reports[experiment] = summary
    report_path.write_text(json.dumps(reports, indent=2) + '\n')
    checkpoint_dir = ROOT / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    torch.save({
        'architecture': 'VehicleCNN',
        'strategy': experiment, 'model_state_dict': best_state,
        'class_to_idx': manifest['class_to_idx'],
        'split_fingerprint': manifest['fingerprint'],
        'best_epoch': best_epoch,
        'preprocessing': summary['configuration']['preprocessing'],
        'validation_accuracy': summary['accuracy'],
    }, checkpoint_dir / f'{experiment}_best.pt')
    print(f"RESULT {experiment}: accuracy={summary['accuracy']:.1%} "
          f"macro_f1={summary['macro_f1']:.3f} best_epoch={best_epoch}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', choices=[
        'baseline', 'augment', 'dropout', 'avg_pool', 'adamw', 'scheduler',
        'regularized', 'imbalance_shuffle', 'balanced_batches', 'bce',
    ])
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    run(args.experiment, args.epochs)
