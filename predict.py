"""Predict a traffic-vehicle class with the selected ResNet18 checkpoint."""

import argparse
import json
from decimal import Decimal
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "resnet18_final.pt"


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class VehiclePredictor:
    """Load the final model once and predict any number of PIL images."""

    def __init__(self, checkpoint_path=DEFAULT_CHECKPOINT, device=None):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["architecture"] != "resnet18":
            raise ValueError("This predictor requires a ResNet18 checkpoint.")
        class_to_idx = checkpoint["class_to_idx"]
        if sorted(class_to_idx.values()) != list(range(len(class_to_idx))):
            raise ValueError("Checkpoint class indices must start at zero and be consecutive.")
        self.class_names = [
            name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])
        ]
        self.threshold = float(checkpoint["review_threshold"])
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Checkpoint review threshold must be between 0 and 1.")

        preprocessing = checkpoint["preprocessing"]
        self.transform = transforms.Compose([
            transforms.Resize(tuple(preprocessing["resize"])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=preprocessing["mean"], std=preprocessing["std"]
            ),
        ])
        self.device = device if device is not None else choose_device()
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.class_names))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict(self, image: Image.Image):
        image_batch = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            scores = self.model(image_batch)
            probabilities_tensor = torch.softmax(scores, dim=1)[0].cpu()

        confidence, predicted_index = probabilities_tensor.max(dim=0)
        probabilities = {
            class_name: float(probabilities_tensor[index])
            for index, class_name in enumerate(self.class_names)
        }
        return {
            "predicted_class": self.class_names[int(predicted_index)],
            "confidence": float(confidence),
            "probabilities": probabilities,
            "needs_review": bool(float(confidence) < self.threshold),
        }


def predict_image(image_path, checkpoint_path=DEFAULT_CHECKPOINT):
    """Keep the original path-based command-line interface."""
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    predictor = VehiclePredictor(checkpoint_path)
    with Image.open(image_path) as image:
        return predictor.predict(image)


def format_prediction_json(result):
    """Render numeric JSON values in plain decimal notation."""
    def fixed_number(value):
        return format(Decimal(str(value)), "f")

    probability_items = list(result["probabilities"].items())
    lines = [
        "{",
        f'  "predicted_class": {json.dumps(result["predicted_class"], ensure_ascii=False)},',
        f'  "confidence": {fixed_number(result["confidence"])},',
        '  "probabilities": {',
    ]
    for index, (class_name, value) in enumerate(probability_items):
        comma = "," if index < len(probability_items) - 1 else ""
        lines.append(
            f"    {json.dumps(class_name, ensure_ascii=False)}: {fixed_number(value)}{comma}"
        )
    lines.extend([
        "  },",
        f'  "needs_review": {json.dumps(result["needs_review"])}',
        "}",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Classify one traffic-vehicle image.")
    parser.add_argument("image_path", type=Path, help="Path to an input image")
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
        help="Path to a ResNet18 checkpoint with review-threshold metadata",
    )
    args = parser.parse_args()
    result = predict_image(args.image_path, args.checkpoint)
    print(format_prediction_json(result))


if __name__ == "__main__":
    main()
