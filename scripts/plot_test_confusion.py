"""Plot private confusion matrices from exported test predictions."""

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / ".local" / "confusion_matrices"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".local" / "matplotlib"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix  # noqa: E402

from scripts.export_test_predictions import NINE_CLASS_FOLDERS, TEST_FOLDERS  # noqa: E402


def save_matrix(name, report, strict_nine_class=False):
    rows = report["images"]
    labels = list(report["model_classes"])
    if strict_nine_class:
        labels.append("neysan")
        true_labels = [row["source_label"] for row in rows]
        policy = "strict_9_class"
    else:
        true_labels = [row["true_class"] for row in rows]
        policy = "neysan_as_vanet"
    predicted_labels = [row["predicted_class"] for row in rows]

    counts = confusion_matrix(true_labels, predicted_labels, labels=labels)
    normalized = confusion_matrix(
        true_labels, predicted_labels, labels=labels, normalize="true"
    )
    figure, axes = plt.subplots(1, 2, figsize=(19, 8))
    for axis, matrix, title, value_format in (
        (axes[0], counts, "Number of images", "d"),
        (axes[1], normalized, "Fraction of each true class", ".2f"),
    ):
        ConfusionMatrixDisplay(matrix, display_labels=labels).plot(
            ax=axis, cmap="Blues", values_format=value_format,
            colorbar=False, xticks_rotation=45,
        )
        axis.set_title(title)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
    figure.suptitle(f"{name} | {policy} | {len(rows)} images")
    figure.tight_layout()
    output_path = OUTPUT_DIR / f"{name}_{policy}.png"
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    print(output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "folders", nargs="*", choices=TEST_FOLDERS,
        help="Test folders to plot; omit for all six",
    )
    args = parser.parse_args()
    for name in args.folders or TEST_FOLDERS:
        report_path = ROOT / ".local" / f"{name}.json"
        if not report_path.is_file():
            raise FileNotFoundError(
                f"Missing {report_path}; run python -m scripts.export_test_predictions first"
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        save_matrix(name, report)
        if name in NINE_CLASS_FOLDERS:
            save_matrix(name, report, strict_nine_class=True)


if __name__ == "__main__":
    main()
