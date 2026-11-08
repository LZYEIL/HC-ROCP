"""Manifest-backed image datasets shared by the three classifier experiments."""

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from shared.paths import image_path


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(train: bool, image_size: int = 224):
    """Build training augmentation or deterministic evaluation preprocessing."""
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(15, interpolation=InterpolationMode.BILINEAR),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ManifestDataset(Dataset):
    def __init__(self, manifest: str, split: str, train: bool, image_size: int = 224):
        """Select one manifest split and its image transform."""
        frame = pd.read_csv(manifest)
        self.frame = frame.loc[frame["split"] == split].reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"No rows for split={split!r} in {manifest}")
        self.transform = build_transform(train=train, image_size=image_size)

    def __len__(self):
        """Return the dataset length."""
        return len(self.frame)

    def __getitem__(self, index):
        """Return one dataset example."""
        row = self.frame.iloc[index]
        with Image.open(image_path(row["path"])) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(row["label"]), str(row["sample_id"])
