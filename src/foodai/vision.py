"""Shared input transformations; import only when training/evaluation begins."""
from __future__ import annotations


def get_transforms(image_size: int, *, train: bool):
    from torchvision import transforms as T
    normalize = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    if train:
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.80, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.10, hue=0.02),
            T.ToTensor(), normalize,
        ])
    return T.Compose([
        T.Resize(round(image_size * 1.14)),
        T.CenterCrop(image_size),
        T.ToTensor(), normalize,
    ])


def get_model(num_classes: int, *, pretrained: bool):
    """EfficientNetV2-S baseline; no download if pretrained=False."""
    from torch import nn
    from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s
    weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
    model = efficientnet_v2_s(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def get_image_dataset(records, root, image_size: int, *, train: bool):
    """Torch Dataset preserving immutable manifest labels (never relabel implicitly)."""
    return ManifestDataset(records, root, image_size, train=train)


class ManifestDataset:
    """Module-level, map-style dataset so Windows spawn workers can pickle it."""
    def __init__(self, records, root, image_size: int, *, train: bool):
        self.records = list(records)
        self.root = root
        self.transform = get_transforms(image_size, train=train)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from PIL import Image, UnidentifiedImageError
        rec = self.records[index]
        try:
            with Image.open(self.root / rec.relative_path) as img:
                image = img.convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise RuntimeError(f"Unreadable image: {rec.relative_path}") from exc
        return self.transform(image), rec.label_id
