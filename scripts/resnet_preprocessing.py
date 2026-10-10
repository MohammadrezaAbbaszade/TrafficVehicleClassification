"""Shared ResNet18 preprocessing for training and checkpoint-based inference."""

from PIL import Image, ImageOps
from torchvision import transforms


class Letterbox:
    """Fit the whole image inside a fixed canvas without changing its aspect ratio."""

    def __init__(self, size=(224, 224), fill=(0, 0, 0)):
        self.size = tuple(size)
        self.fill = tuple(fill)

    def __call__(self, image):
        contained = ImageOps.contain(
            image.convert("RGB"), self.size, method=Image.Resampling.BILINEAR
        )
        canvas = Image.new("RGB", self.size, self.fill)
        left = (self.size[0] - contained.width) // 2
        top = (self.size[1] - contained.height) // 2
        canvas.paste(contained, (left, top))
        return canvas


def make_resnet_transform(preprocessing, train_flip_p=0.0):
    """Honor the transform recorded in a checkpoint; old checkpoints use resize."""
    size = tuple(preprocessing["resize"])
    method = preprocessing.get("method", "resize")
    if method == "letterbox":
        geometry = Letterbox(size=size, fill=preprocessing["fill"])
    elif method == "resize":
        geometry = transforms.Resize(size)
    else:
        raise ValueError(f"Unknown ResNet preprocessing method: {method}")
    if not 0.0 <= train_flip_p <= 1.0:
        raise ValueError("train_flip_p must be between 0 and 1")
    steps = [geometry]
    if train_flip_p:
        steps.append(transforms.RandomHorizontalFlip(p=train_flip_p))
    steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=preprocessing["mean"], std=preprocessing["std"]
        ),
    ])
    return transforms.Compose(steps)
