"""Export private, per-folder predictions for the mentor-provided test sets."""

import json
from pathlib import Path

from predict import format_prediction_percentages
from predict_folder import predict_directory


ROOT = Path(__file__).resolve().parents[1]
TEST_FOLDERS = ("test0", "test1", "test1_2", "test2", "test3", "test4")
NINE_CLASS_FOLDERS = {"test1", "test1_2"}


def main():
    data_root = ROOT / "TestingData"
    output_dir = ROOT / ".local"
    if not data_root.is_dir():
        raise FileNotFoundError(f"Test data folder not found: {data_root}")

    reports = {}
    for name in TEST_FOLDERS:
        print(f"Processing {name}...", flush=True)
        report = predict_directory(data_root / name, labeled=True)
        if name in NINE_CLASS_FOLDERS:
            rows = report["images"]
            correct = sum(
                row["predicted_class"] == row["source_label"] for row in rows
            )
            report["metrics"]["strict_nine_class"] = {
                "images": len(rows),
                "correct": correct,
                "accuracy": correct / len(rows),
            }
        report["images"] = [
            format_prediction_percentages(row) for row in report["images"]
        ]
        reports[name] = report

    output_dir.mkdir(exist_ok=True)
    for name, report in reports.items():
        output_path = output_dir / f"{name}.json"
        temporary_path = output_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(output_path)
        metric = report["metrics"]["with_neysan"]
        print(
            f"{name}: {metric['correct']}/{metric['images']} correct "
            f"({metric['accuracy']:.1%}) -> {output_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
