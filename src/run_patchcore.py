import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from dataset import MVTecBottleDataset
from patchcore import (
    PatchCoreResult,
    ResNet18FeatureExtractor,
    batch_metadata_to_rows,
    binary_metrics,
    coreset_subsample,
    load_mask,
    localization_metrics,
    nearest_neighbor_distances,
    normalize_map,
    patch_embeddings,
    per_defect_recall,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PatchCore unsupervised anomaly detection on MVTec bottle.")
    parser.add_argument("--metadata_path", type=Path, default=Path("results/metadata.csv"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/patchcore"))
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--coreset_ratio", type=float, default=0.1)
    parser.add_argument("--threshold_percentile", type=float, default=99.0)
    parser.add_argument("--map_threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_pretrained", action="store_true", help="Use randomly initialized ResNet18 features.")
    parser.add_argument("--max_visualizations", type=int, default=8)
    return parser.parse_args()


class FilteredDataset(Subset):
    @property
    def samples(self) -> list[dict[str, str]]:
        dataset = self.dataset
        if not isinstance(dataset, MVTecBottleDataset):
            return []
        return [dataset.samples[index] for index in self.indices]


def filter_dataset(dataset: MVTecBottleDataset, *, label: int | None = None) -> FilteredDataset:
    indices = [
        index
        for index, sample in enumerate(dataset.samples)
        if label is None or int(sample["binary_label"]) == label
    ]
    if not indices:
        raise ValueError(f"No samples left after filtering split={dataset.split!r}, label={label!r}.")
    return FilteredDataset(dataset, indices)


def assert_normal_only_memory_dataset(dataset: FilteredDataset) -> None:
    labels = sorted({int(sample["binary_label"]) for sample in dataset.samples})
    if labels != [0]:
        raise ValueError(f"PatchCore memory bank must be built from normal samples only, but got labels={labels}.")


def make_loader(dataset: Dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def build_memory_bank(
    extractor: ResNet18FeatureExtractor,
    loader: DataLoader,
    device: torch.device,
    coreset_ratio: float,
    seed: int,
) -> torch.Tensor:
    embeddings: list[torch.Tensor] = []
    with torch.no_grad():
        for images, _labels, _metadata in tqdm(loader, desc="Building PatchCore memory bank", leave=False):
            images = images.to(device)
            batch_embeddings = patch_embeddings(extractor(images))
            embeddings.append(batch_embeddings.reshape(-1, batch_embeddings.shape[-1]).cpu())

    memory_bank = torch.cat(embeddings, dim=0).to(device)
    return coreset_subsample(memory_bank, ratio=coreset_ratio, seed=seed)


def score_dataset(
    extractor: ResNet18FeatureExtractor,
    loader: DataLoader,
    memory_bank: torch.Tensor,
    image_size: int,
    device: torch.device,
) -> PatchCoreResult:
    image_scores: list[float] = []
    anomaly_maps: list[np.ndarray] = []
    rows: list[dict[str, str]] = []

    with torch.no_grad():
        for images, _labels, batch_metadata in tqdm(loader, desc="Scoring images", leave=False):
            images = images.to(device)
            embeddings = patch_embeddings(extractor(images))
            batch_size, patch_h, patch_w, channels = embeddings.shape
            distances = nearest_neighbor_distances(embeddings.reshape(-1, channels), memory_bank)
            distance_maps = distances.reshape(batch_size, 1, patch_h, patch_w)
            distance_maps = F.interpolate(distance_maps, size=(image_size, image_size), mode="bilinear", align_corners=False)

            for distance_map in distance_maps.squeeze(1).cpu().numpy():
                image_scores.append(float(distance_map.max()))
                anomaly_maps.append(normalize_map(distance_map))

            rows.extend(batch_metadata_to_rows(batch_metadata))

    return PatchCoreResult(image_scores=image_scores, anomaly_maps=anomaly_maps, metadata=rows)


def choose_threshold(scores: list[float], percentile: float) -> float:
    if not scores:
        raise ValueError("Cannot choose PatchCore threshold from an empty score list.")
    return float(np.percentile(np.asarray(scores, dtype=np.float32), percentile))


def save_predictions(
    result: PatchCoreResult,
    predictions: list[int],
    output_path: Path,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["image_path", "defect_type", "true_label", "pred_label", "anomaly_score", "is_correct"],
        )
        writer.writeheader()
        for score, pred_label, row in zip(result.image_scores, predictions, result.metadata, strict=True):
            true_label = int(row["binary_label"])
            writer.writerow(
                {
                    "image_path": row["image_path"],
                    "defect_type": row["defect_type"],
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "anomaly_score": f"{score:.6f}",
                    "is_correct": true_label == pred_label,
                }
            )


def save_localization(
    result: PatchCoreResult,
    output_dir: Path,
    image_size: int,
    map_threshold: float,
) -> dict[str, Any]:
    maps_dir = output_dir / "anomaly_maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for index, (anomaly_map, row) in enumerate(zip(result.anomaly_maps, result.metadata, strict=True)):
        image_path = Path(row["image_path"])
        map_path = maps_dir / f"{index:04d}_{image_path.stem}_{row['defect_type']}.npy"
        np.save(map_path, anomaly_map)

        mask = load_mask(row["mask_path"], image_size)
        output_row: dict[str, Any] = {
            "image_path": row["image_path"],
            "mask_path": row["mask_path"],
            "defect_type": row["defect_type"],
            "true_label": int(row["binary_label"]),
            "has_mask": mask is not None,
            "anomaly_map_path": map_path.as_posix(),
        }
        if mask is not None:
            output_row.update(localization_metrics(anomaly_map, mask, threshold=map_threshold))
        else:
            output_row.update(
                {
                    "anomaly_iou": "",
                    "anomaly_dice": "",
                    "pointing_hit": "",
                    "mean_anomaly_inside_mask": "",
                    "mean_anomaly_outside_mask": "",
                }
            )
        rows.append(output_row)

    csv_path = output_dir / "patchcore_localization.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    localized = [row for row in rows if row["has_mask"]]
    summary: dict[str, Any] = {
        "num_samples": len(rows),
        "num_localization_samples": len(localized),
    }
    if localized:
        summary.update(
            {
                "mean_anomaly_iou": float(np.mean([row["anomaly_iou"] for row in localized])),
                "mean_anomaly_dice": float(np.mean([row["anomaly_dice"] for row in localized])),
                "pointing_hit_rate": float(np.mean([1.0 if row["pointing_hit"] else 0.0 for row in localized])),
                "mean_anomaly_inside_mask": float(np.mean([row["mean_anomaly_inside_mask"] for row in localized])),
                "mean_anomaly_outside_mask": float(np.mean([row["mean_anomaly_outside_mask"] for row in localized])),
            }
        )
    return summary


def load_display_image(image_path: Path, image_size: int) -> np.ndarray:
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize((image_size, image_size))
        return np.asarray(image, dtype=np.float32) / 255.0


def overlay_heatmap(image: np.ndarray, anomaly_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heatmap = plt.get_cmap("jet")(anomaly_map)[..., :3]
    return np.clip((1 - alpha) * image + alpha * heatmap, 0, 1)


def save_visualization(result: PatchCoreResult, output_path: Path, image_size: int, max_visualizations: int) -> None:
    selected_indices = [index for index, row in enumerate(result.metadata) if int(row["binary_label"]) == 1][:max_visualizations]
    cols = 4
    rows_count = max(1, len(selected_indices))
    fig, axes = plt.subplots(rows_count, cols, figsize=(cols * 3.2, rows_count * 2.9))
    axes_array = np.atleast_2d(axes)

    if not selected_indices:
        for ax in axes_array.flatten():
            ax.axis("off")
        axes_array[0, 0].text(0.5, 0.5, "No defect samples", ha="center", va="center")
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return

    for row_index, sample_index in enumerate(selected_indices):
        row = result.metadata[sample_index]
        image = load_display_image(Path(row["image_path"]), image_size)
        anomaly_map = result.anomaly_maps[sample_index]
        mask = load_mask(row["mask_path"], image_size)
        overlay = overlay_heatmap(image, anomaly_map)
        panels = [
            ("Image", image),
            ("Ground-truth mask", mask if mask is not None else np.zeros_like(anomaly_map)),
            ("PatchCore map", anomaly_map),
            (f"Overlay score={result.image_scores[sample_index]:.2f}", overlay),
        ]

        for col_index, (title, panel) in enumerate(panels):
            ax = axes_array[row_index, col_index]
            ax.imshow(panel, cmap="gray" if col_index in (1, 2) else None, vmin=0, vmax=1)
            ax.set_title(title, fontsize=9)
            ax.axis("off")

    fig.suptitle("PatchCore Anomaly Maps vs Ground-truth Masks", fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985), h_pad=1.4, w_pad=1.0)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_score_distribution(result: PatchCoreResult, threshold: float, output_path: Path) -> None:
    good_scores = [score for score, row in zip(result.image_scores, result.metadata, strict=True) if int(row["binary_label"]) == 0]
    defect_scores = [score for score, row in zip(result.image_scores, result.metadata, strict=True) if int(row["binary_label"]) == 1]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(good_scores, bins=12, alpha=0.75, label="good")
    ax.hist(defect_scores, bins=12, alpha=0.75, label="defect")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label="threshold")
    ax.set_xlabel("PatchCore anomaly score")
    ax.set_ylabel("Image count")
    ax.set_title("PatchCore Score Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.no_pretrained:
        print(
            "Warning: --no_pretrained uses random frozen ResNet18 features. "
            "This is a fallback/ablation mode and usually performs worse than default PatchCore."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = ResNet18FeatureExtractor(pretrained=not args.no_pretrained).to(device)
    extractor.eval()

    train_dataset = MVTecBottleDataset(args.metadata_path, split="train", image_size=args.image_size, train_transforms=False)
    val_dataset = MVTecBottleDataset(args.metadata_path, split="val", image_size=args.image_size, train_transforms=False)
    test_dataset = MVTecBottleDataset(args.metadata_path, split="test", image_size=args.image_size, train_transforms=False)

    train_good = filter_dataset(train_dataset, label=0)
    val_good = filter_dataset(val_dataset, label=0)
    assert_normal_only_memory_dataset(train_good)
    assert_normal_only_memory_dataset(val_good)

    memory_bank = build_memory_bank(
        extractor=extractor,
        loader=make_loader(train_good, args.batch_size, args.num_workers),
        device=device,
        coreset_ratio=args.coreset_ratio,
        seed=args.seed,
    )

    val_result = score_dataset(
        extractor=extractor,
        loader=make_loader(val_good, args.batch_size, args.num_workers),
        memory_bank=memory_bank,
        image_size=args.image_size,
        device=device,
    )
    threshold = choose_threshold(val_result.image_scores, args.threshold_percentile)

    test_result = score_dataset(
        extractor=extractor,
        loader=make_loader(test_dataset, args.batch_size, args.num_workers),
        memory_bank=memory_bank,
        image_size=args.image_size,
        device=device,
    )

    y_true = [int(row["binary_label"]) for row in test_result.metadata]
    y_pred = [1 if score > threshold else 0 for score in test_result.image_scores]
    localization_summary = save_localization(test_result, args.output_dir, args.image_size, args.map_threshold)
    metrics = {
        **binary_metrics(y_true, y_pred),
        "per_defect_type_recall": per_defect_recall(y_true, y_pred, test_result.metadata),
        "num_test_samples": len(y_true),
        "method": "PatchCore",
        "supervised_classifier_checkpoint_used": False,
        "supervised_classifier_checkpoint_path": None,
        "mvtec_memory_bank_split": "train/good",
        "mvtec_threshold_split": "val/good",
        "mvtec_training_labels_used": [0],
        "backbone_source": "imagenet_pretrained_resnet18_frozen" if not args.no_pretrained else "random_resnet18_frozen",
        "threshold": threshold,
        "threshold_source": f"val/good p{args.threshold_percentile:g}",
        "coreset_ratio": args.coreset_ratio,
        "memory_bank_size": int(memory_bank.shape[0]),
        "localization": localization_summary,
    }

    with (args.output_dir / "patchcore_metrics.json").open("w", encoding="utf-8") as json_file:
        json.dump(metrics, json_file, indent=2)

    save_predictions(test_result, y_pred, args.output_dir / "patchcore_predictions.csv")
    save_visualization(test_result, args.output_dir / "patchcore_heatmaps.png", args.image_size, args.max_visualizations)
    save_score_distribution(test_result, threshold, args.output_dir / "patchcore_score_distribution.png")

    print(json.dumps(metrics, indent=2))
    print("PatchCore did not load results/best_model.pth or any supervised MVTec classifier checkpoint.")
    print("Memory bank was built from train/good only; validation threshold was selected from val/good only.")
    print(f"Saved PatchCore metrics to {args.output_dir / 'patchcore_metrics.json'}")
    print(f"Saved PatchCore predictions to {args.output_dir / 'patchcore_predictions.csv'}")
    print(f"Saved PatchCore heatmaps to {args.output_dir / 'patchcore_heatmaps.png'}")


if __name__ == "__main__":
    main()
