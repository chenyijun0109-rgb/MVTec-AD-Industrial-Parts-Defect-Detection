import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


DEFECT_TYPES = ("broken_large", "broken_small", "contamination")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass(frozen=True)
class Sample:
    image_path: Path
    mask_path: Path | None
    original_split: str
    defect_type: str
    binary_label: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check MVTec AD bottle data and create metadata.")
    parser.add_argument("--data_dir", type=Path, default=Path("data/bottle"), help="Path to MVTec bottle data.")
    parser.add_argument("--output_dir", type=Path, default=Path("results"), help="Directory for CSV outputs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits.")
    parser.add_argument("--normal_train_ratio", type=float, default=0.8, help="Ratio of train/good used for training.")
    parser.add_argument("--defect_train_ratio", type=float, default=0.6, help="Ratio of each defect type used for training.")
    parser.add_argument("--defect_val_ratio", type=float, default=0.2, help="Ratio of each defect type used for validation.")
    return parser.parse_args()


def list_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def inspect_image(path: Path) -> dict[str, object]:
    try:
        with Image.open(path) as image:
            return {
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "readable": True,
                "error": "",
            }
    except Exception as exc:  # noqa: BLE001 - keep the check report useful for corrupted files.
        return {
            "width": "",
            "height": "",
            "mode": "",
            "readable": False,
            "error": str(exc),
        }


def build_samples(data_dir: Path) -> list[Sample]:
    samples: list[Sample] = []

    for image_path in list_images(data_dir / "train" / "good"):
        samples.append(
            Sample(
                image_path=image_path,
                mask_path=None,
                original_split="train",
                defect_type="good",
                binary_label=0,
            )
        )

    for image_path in list_images(data_dir / "test" / "good"):
        samples.append(
            Sample(
                image_path=image_path,
                mask_path=None,
                original_split="test",
                defect_type="good",
                binary_label=0,
            )
        )

    for defect_type in DEFECT_TYPES:
        for image_path in list_images(data_dir / "test" / defect_type):
            mask_name = f"{image_path.stem}_mask{image_path.suffix}"
            samples.append(
                Sample(
                    image_path=image_path,
                    mask_path=data_dir / "ground_truth" / defect_type / mask_name,
                    original_split="test",
                    defect_type=defect_type,
                    binary_label=1,
                )
            )

    return samples


def assign_splits(
    samples: list[Sample],
    seed: int,
    normal_train_ratio: float,
    defect_train_ratio: float,
    defect_val_ratio: float,
) -> dict[Path, str]:
    rng = random.Random(seed)
    split_by_path: dict[Path, str] = {}

    train_good = [sample for sample in samples if sample.original_split == "train" and sample.defect_type == "good"]
    rng.shuffle(train_good)
    normal_train_count = round(len(train_good) * normal_train_ratio)
    for index, sample in enumerate(train_good):
        split_by_path[sample.image_path] = "train" if index < normal_train_count else "val"

    for sample in samples:
        if sample.original_split == "test" and sample.defect_type == "good":
            split_by_path[sample.image_path] = "test"

    for defect_type in DEFECT_TYPES:
        defect_samples = [sample for sample in samples if sample.defect_type == defect_type]
        rng.shuffle(defect_samples)
        train_count = round(len(defect_samples) * defect_train_ratio)
        val_count = round(len(defect_samples) * defect_val_ratio)
        for index, sample in enumerate(defect_samples):
            if index < train_count:
                split = "train"
            elif index < train_count + val_count:
                split = "val"
            else:
                split = "test"
            split_by_path[sample.image_path] = split

    return split_by_path


def relative(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def write_reports(samples: list[Sample], split_by_path: dict[Path, str], data_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_check_path = output_dir / "data_check.csv"
    metadata_path = output_dir / "metadata.csv"

    with data_check_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "image_path",
                "mask_path",
                "original_split",
                "defect_type",
                "binary_label",
                "image_width",
                "image_height",
                "image_mode",
                "image_readable",
                "image_error",
                "mask_exists",
                "mask_readable",
                "mask_width",
                "mask_height",
                "mask_size_matches_image",
            ],
        )
        writer.writeheader()
        for sample in samples:
            image_info = inspect_image(sample.image_path)
            mask_info = None
            mask_exists = sample.mask_path.exists() if sample.mask_path else ""
            if sample.mask_path and sample.mask_path.exists():
                mask_info = inspect_image(sample.mask_path)

            writer.writerow(
                {
                    "image_path": relative(sample.image_path, data_dir.parent),
                    "mask_path": relative(sample.mask_path, data_dir.parent) if sample.mask_path else "",
                    "original_split": sample.original_split,
                    "defect_type": sample.defect_type,
                    "binary_label": sample.binary_label,
                    "image_width": image_info["width"],
                    "image_height": image_info["height"],
                    "image_mode": image_info["mode"],
                    "image_readable": image_info["readable"],
                    "image_error": image_info["error"],
                    "mask_exists": mask_exists,
                    "mask_readable": mask_info["readable"] if mask_info else "",
                    "mask_width": mask_info["width"] if mask_info else "",
                    "mask_height": mask_info["height"] if mask_info else "",
                    "mask_size_matches_image": (
                        image_info["width"] == mask_info["width"] and image_info["height"] == mask_info["height"]
                        if mask_info
                        else ""
                    ),
                }
            )

    with metadata_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "image_path",
                "mask_path",
                "original_split",
                "defect_type",
                "binary_label",
                "experiment_split",
            ],
        )
        writer.writeheader()
        for sample in sorted(samples, key=lambda item: item.image_path.as_posix()):
            writer.writerow(
                {
                    "image_path": relative(sample.image_path, Path.cwd()),
                    "mask_path": relative(sample.mask_path, Path.cwd()) if sample.mask_path else "",
                    "original_split": sample.original_split,
                    "defect_type": sample.defect_type,
                    "binary_label": sample.binary_label,
                    "experiment_split": split_by_path[sample.image_path],
                }
            )


def print_summary(samples: list[Sample], split_by_path: dict[Path, str]) -> None:
    print("Data summary")
    for original_split in ("train", "test"):
        subset = [sample for sample in samples if sample.original_split == original_split]
        print(f"- original {original_split}: {len(subset)} images")

    print("Defect type counts")
    for defect_type in ("good", *DEFECT_TYPES):
        count = sum(1 for sample in samples if sample.defect_type == defect_type)
        print(f"- {defect_type}: {count}")

    print("Experiment split counts")
    for split in ("train", "val", "test"):
        subset = [sample for sample in samples if split_by_path[sample.image_path] == split]
        good_count = sum(1 for sample in subset if sample.binary_label == 0)
        defect_count = sum(1 for sample in subset if sample.binary_label == 1)
        print(f"- {split}: {len(subset)} images (good={good_count}, defect={defect_count})")


def main() -> None:
    args = parse_args()
    samples = build_samples(args.data_dir)
    split_by_path = assign_splits(
        samples=samples,
        seed=args.seed,
        normal_train_ratio=args.normal_train_ratio,
        defect_train_ratio=args.defect_train_ratio,
        defect_val_ratio=args.defect_val_ratio,
    )
    write_reports(samples, split_by_path, args.data_dir, args.output_dir)
    print_summary(samples, split_by_path)
    print(f"Saved data check to {args.output_dir / 'data_check.csv'}")
    print(f"Saved metadata to {args.output_dir / 'metadata.csv'}")


if __name__ == "__main__":
    main()
