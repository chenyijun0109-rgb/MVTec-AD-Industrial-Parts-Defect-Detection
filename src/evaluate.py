import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LABEL_NAMES, MVTecBottleDataset
from model import build_resnet18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained MVTec bottle classifier.")
    parser.add_argument("--metadata_path", type=Path, default=Path("results/metadata.csv"))
    parser.add_argument("--model_path", type=Path, default=Path("results/best_model.pth"))
    parser.add_argument("--output_dir", type=Path, default=Path("results"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def load_model(model_path: Path, image_size: int, device: torch.device) -> tuple[torch.nn.Module, int]:
    checkpoint = torch.load(model_path, map_location=device)
    checkpoint_image_size = int(checkpoint.get("image_size", image_size))
    model = build_resnet18(num_classes=2, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint_image_size


def collect_predictions(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[int], list[int], list[float], list[dict[str, str]]]:
    y_true: list[int] = []
    y_pred: list[int] = []
    confidences: list[float] = []
    metadata: list[dict[str, str]] = []

    with torch.no_grad():
        for images, labels, batch_metadata in tqdm(loader, leave=False):
            images = images.to(device)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            batch_confidence, predictions = probabilities.max(dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())
            confidences.extend(batch_confidence.cpu().tolist())

            batch_size = labels.size(0)
            for index in range(batch_size):
                metadata.append({key: value[index] for key, value in batch_metadata.items()})

    return y_true, y_pred, confidences, metadata


def confusion_matrix_2x2(y_true: list[int], y_pred: list[int]) -> list[list[int]]:
    matrix = [[0, 0], [0, 0]]
    for true_label, pred_label in zip(y_true, y_pred, strict=True):
        matrix[true_label][pred_label] += 1
    return matrix


def binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float | list[list[int]]]:
    matrix = confusion_matrix_2x2(y_true, y_pred)
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": matrix,
    }


def save_confusion_matrix(y_true: list[int], y_pred: list[int], output_path: Path) -> None:
    matrix = np.array(confusion_matrix_2x2(y_true, y_pred))
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(2), [LABEL_NAMES[0], LABEL_NAMES[1]])
    ax.set_yticks(np.arange(2), [LABEL_NAMES[0], LABEL_NAMES[1]])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    threshold = matrix.max() / 2 if matrix.max() > 0 else 0
    for row in range(2):
        for col in range(2):
            color = "white" if matrix[row, col] > threshold else "black"
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def per_defect_recall(y_true: list[int], y_pred: list[int], metadata: list[dict[str, str]]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for defect_type in sorted({row["defect_type"] for row in metadata}):
        indices = [index for index, row in enumerate(metadata) if row["defect_type"] == defect_type]
        if not indices:
            result[defect_type] = None
            continue
        type_true = [y_true[index] for index in indices]
        type_pred = [y_pred[index] for index in indices]
        positive_count = sum(type_true)
        true_positive = sum(1 for true_label, pred_label in zip(type_true, type_pred, strict=True) if true_label == 1 and pred_label == 1)
        result[defect_type] = true_positive / positive_count if positive_count else None
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, image_size = load_model(args.model_path, args.image_size, device)
    dataset = MVTecBottleDataset(args.metadata_path, split="test", image_size=image_size, train_transforms=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    y_true, y_pred, confidences, metadata = collect_predictions(model, loader, device)

    metrics = {
        **binary_metrics(y_true, y_pred),
        "per_defect_type_recall": per_defect_recall(y_true, y_pred, metadata),
        "num_test_samples": len(y_true),
    }

    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as json_file:
        json.dump(metrics, json_file, indent=2)

    predictions_path = args.output_dir / "predictions.csv"
    with predictions_path.open("w", encoding="utf-8") as csv_file:
        csv_file.write("image_path,defect_type,true_label,pred_label,confidence,is_correct\n")
        for true_label, pred_label, confidence, row in zip(y_true, y_pred, confidences, metadata, strict=True):
            csv_file.write(
                f"{row['image_path']},{row['defect_type']},{true_label},{pred_label},{confidence:.6f},{true_label == pred_label}\n"
            )

    save_confusion_matrix(y_true, y_pred, args.output_dir / "confusion_matrix.png")

    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {args.output_dir / 'metrics.json'}")
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved confusion matrix to {args.output_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
