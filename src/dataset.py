import csv
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


LABEL_NAMES = {0: "good", 1: "defect"}


def build_transforms(image_size: int = 224, train: bool = False) -> transforms.Compose:
    steps: list[Any] = [transforms.Resize((image_size, image_size))]
    if train:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=8),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            ]
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(steps)


class MVTecBottleDataset(Dataset):
    def __init__(
        self,
        metadata_path: str | Path = "results/metadata.csv",
        split: str = "train",
        image_size: int = 224,
        train_transforms: bool | None = None,
    ) -> None:
        self.metadata_path = Path(metadata_path)
        self.split = split
        self.transform = build_transforms(image_size=image_size, train=(split == "train" if train_transforms is None else train_transforms))
        self.samples = self._load_samples()

        if not self.samples:
            raise ValueError(f"No samples found for split={split!r} in {self.metadata_path}")

    def _load_samples(self) -> list[dict[str, str]]:
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {self.metadata_path}. Run src/check_data.py first.")

        with self.metadata_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            return [row for row in reader if row["experiment_split"] == self.split]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, int, dict[str, str]]:
        sample = self.samples[index]
        image_path = Path(sample["image_path"])
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)

        label = int(sample["binary_label"])
        return tensor, label, sample


def class_counts(dataset: MVTecBottleDataset) -> dict[int, int]:
    counts = {0: 0, 1: 0}
    for sample in dataset.samples:
        counts[int(sample["binary_label"])] += 1
    return counts
