"""Run the selected eight-class model on a folder of vehicle crops."""

import argparse
import json
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from predict import DEFAULT_CHECKPOINT, VehiclePredictor


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def image_paths(directory):
    if not directory.is_dir():
        raise ValueError(f"Input is not a directory: {directory}")
    paths = sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise ValueError(f"No supported images found in: {directory}")
    return paths


def source_label(image_path, directory, class_names):
    parts = image_path.relative_to(directory).parts
    if len(parts) < 2:
        raise ValueError(
            f"Labeled mode needs class subfolders; image is at the root: {parts[0]}"
        )
    label = parts[0].casefold()
    mapped_label = "vanet" if label == "neysan" else label
    if mapped_label not in class_names:
        expected = ", ".join(class_names + ["neysan"])
        raise ValueError(f"Unknown class folder {parts[0]!r}; expected: {expected}")
    return label, mapped_label


def accuracy_summary(rows):
    if not rows:
        return None
    correct = sum(row["correct"] for row in rows)
    return {"images": len(rows), "correct": correct, "accuracy": correct / len(rows)}


def predict_directory(directory, checkpoint_path=DEFAULT_CHECKPOINT, labeled=False):
    directory = Path(directory).expanduser().resolve()
    paths = image_paths(directory)
    predictor = VehiclePredictor(checkpoint_path)
    records = []

    for path in paths:
        source = target = None
        if labeled:
            source, target = source_label(path, directory, predictor.class_names)
        try:
            with Image.open(path) as image:
                result = predictor.predict(image)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise RuntimeError(f"Could not read image: {path.relative_to(directory)}") from exc

        record = {"image": path.relative_to(directory).as_posix(), **result}
        if labeled:
            record["source_label"] = source
            record["true_class"] = target
            record["correct"] = result["predicted_class"] == target
        records.append(record)

    output = {"mode": "labeled" if labeled else "unlabeled", "images": records}
    if labeled:
        ordinary = [row for row in records if row["source_label"] != "neysan"]
        neysan = [row for row in records if row["source_label"] == "neysan"]
        output["label_mapping"] = {"neysan": "vanet"}
        output["metrics"] = {
            "without_neysan": accuracy_summary(ordinary),
            "with_neysan": accuracy_summary(records),
            "neysan_only": accuracy_summary(neysan),
        }
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Predict every image in a folder with the selected ResNet18 checkpoint."
    )
    parser.add_argument("directory", type=Path, help="Folder of vehicle images")
    parser.add_argument(
        "--labeled", action="store_true",
        help="Read ground-truth labels from class subfolders and report accuracy",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--output", type=Path,
        help="Save JSON to a local path instead of printing it to standard output",
    )
    args = parser.parse_args()

    result = predict_directory(args.directory, args.checkpoint, args.labeled)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Saved {len(result['images'])} predictions to {args.output}")


if __name__ == "__main__":
    main()
