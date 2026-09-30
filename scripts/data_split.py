"""Build one private, group-safe development split from both vehicle datasets.

The former test folder is part of the development pool; mentor-held test images
are never available here. Pending unclean labels are used provisionally and
remain identified as such in the manifest.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
REVIEW_FILE = ROOT / "reports/data_review_decisions.csv"
SPLIT_FILE = ROOT / "reports/base_split.json"
SUMMARY_FILE = ROOT / "reports/data_summary.json"
LABELS = ("ambulance", "autobus", "kamyun", "kamyunet", "minibus", "savari", "taxi", "vanet")
SEED = 42
N_FOLDS = 5
NEAR_DUPLICATE_DISTANCE = 4


def perceptual_hash(path: Path) -> int:
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L").resize((32, 32)), dtype=np.float64)
    frequencies = dctn(pixels, norm="ortho")[:8, :8]
    median = np.median(frequencies[1:, :])
    return sum(int(bit) << index for index, bit in enumerate((frequencies > median).ravel()))


def eligible_samples():
    with REVIEW_FILE.open(newline="", encoding="utf-8") as file:
        inventory = list(csv.DictReader(file))
    if not inventory:
        raise ValueError("The private data-review inventory is empty.")
    by_hash = defaultdict(list)
    for item in inventory:
        decision = item["decision"] or "pending"
        if decision not in {"pending", "accept", "exclude", "relabel"}:
            raise ValueError(f"Unknown review decision: {decision}")
        if decision == "exclude" or not item["content_hash"]:
            continue
        original = item["original_label"]
        label = item["reviewed_label"] if decision == "relabel" else original
        if original == "neysan" and decision != "relabel":
            label = "vanet"
        if label not in LABELS:
            raise ValueError(f"Unknown target label for {item['path']}: {label}")
        path = Path(item["path"])
        if len(path.parts) != 4 or path.parts[0] not in {"dataset", "dataset_extra"}:
            raise ValueError(f"Unexpected dataset path: {path}")
        if not (ROOT / path).is_file():
            raise FileNotFoundError(ROOT / path)
        source = "neysan" if original == "neysan" else (
            "unclean" if path.parts[1] == "unclean" else "labelled"
        )
        status = "provisional" if decision == "pending" and path.parts[1] == "unclean" else (
            "reviewed" if decision in {"accept", "relabel"} else "source_label"
        )
        by_hash[item["content_hash"]].append({
            "path": path.as_posix(), "label": label,
            "original_label": original, "content_hash": item["content_hash"],
            "source": source, "label_status": status,
        })
    samples = []
    conflicts = 0
    for content_hash, copies in by_hash.items():
        if len({row["label"] for row in copies}) != 1:
            conflicts += 1
            continue
        # Prefer a source-labelled copy when the same pixels also occur in unclean.
        samples.append(min(copies, key=lambda row: (
            row["label_status"] == "provisional", row["path"]
        )))
    samples.sort(key=lambda row: row["path"])
    return samples, {"inventory_images": len(inventory), "conflicting_exact_groups": conflicts,
                     "excluded_exact_copies": sum(len(group) - 1 for group in by_hash.values()
                                                  if len({r['label'] for r in group}) == 1)}


def near_duplicate_groups(samples):
    signatures = [perceptual_hash(ROOT / row["path"]) for row in samples]
    parent = list(range(len(samples)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left in range(len(samples)):
        for right in range(left + 1, len(samples)):
            if (signatures[left] ^ signatures[right]).bit_count() <= NEAR_DUPLICATE_DISTANCE:
                parent[find(left)] = find(right)
    roots = [find(index) for index in range(len(samples))]
    return roots


def choose_split(samples, groups):
    strata = [f"{row['label']}:{row['source']}" for row in samples]
    total = Counter(strata)
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    candidates = []
    for training, validation in splitter.split(np.zeros(len(samples)), strata, groups):
        counts = Counter(strata[index] for index in validation)
        if any(counts[key] == 0 or counts[key] == total[key] for key in total):
            continue
        target = {key: total[key] / N_FOLDS for key in total}
        imbalance = sum(abs(counts[key] - target[key]) / max(1, target[key]) for key in total)
        candidates.append((imbalance, abs(len(validation) - len(samples) / N_FOLDS), training, validation))
    if not candidates:
        raise ValueError("No group-safe fold has every class and source in both partitions.")
    _, _, train_indices, val_indices = min(candidates, key=lambda item: item[:2])
    train = [samples[int(index)]["path"] for index in train_indices]
    validation = [samples[int(index)]["path"] for index in val_indices]
    assert set(groups[index] for index in train_indices).isdisjoint(
        groups[index] for index in val_indices
    )
    return sorted(train), sorted(validation)


def build_manifest():
    samples, audit = eligible_samples()
    groups = near_duplicate_groups(samples)
    train, validation = choose_split(samples, groups)
    for sample, group in zip(samples, groups):
        sample["near_duplicate_group"] = group
    manifest = {
        "schema_version": 2,
        "seed": SEED,
        "validation_ratio_target": 1 / N_FOLDS,
        "near_duplicate_phash_distance": NEAR_DUPLICATE_DISTANCE,
        "class_to_idx": {label: index for index, label in enumerate(LABELS)},
        "sources": ["dataset", "dataset_extra"],
        "unclean_label_policy": "pending labels are provisional; reviewed decisions override them",
        "neysan_target_label": "vanet",
        "samples": samples,
        "training_paths": train,
        "validation_paths": validation,
    }
    manifest["fingerprint"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    by_path = {row["path"]: row for row in samples}
    summary = {
        **audit, "split_fingerprint": manifest["fingerprint"],
        "eligible_unique_images": len(samples), "training_images": len(train),
        "validation_images": len(validation),
        "training_by_class": dict(Counter(by_path[path]["label"] for path in train)),
        "validation_by_class": dict(Counter(by_path[path]["label"] for path in validation)),
        "training_by_source": dict(Counter(by_path[path]["source"] for path in train)),
        "validation_by_source": dict(Counter(by_path[path]["source"] for path in validation)),
        "near_duplicate_groups": len(set(groups)),
        "provisional_validation_images": sum(by_path[path]["label_status"] == "provisional"
                                             for path in validation),
    }
    return manifest, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Replace the ignored private split and summary")
    args = parser.parse_args()
    manifest, summary = build_manifest()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.write:
        SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        SPLIT_FILE.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        SUMMARY_FILE.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("Saved private split:", SPLIT_FILE)


if __name__ == "__main__":
    main()
