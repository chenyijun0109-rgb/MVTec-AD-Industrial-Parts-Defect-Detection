import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a unified comparison report for supervised and anomaly detection methods.")
    parser.add_argument("--classifier_metrics", type=Path, default=Path("results/metrics.json"))
    parser.add_argument("--classifier_localization", type=Path, default=Path("results/localization_metrics.json"))
    parser.add_argument("--patchcore_metrics", type=Path, default=Path("results/patchcore/patchcore_metrics.json"))
    parser.add_argument("--output_csv", type=Path, default=Path("results/method_comparison.csv"))
    parser.add_argument("--output_md", type=Path, default=Path("results/method_comparison.md"))
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    with path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "No" if not value else "Yes"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def method_rows(classifier_metrics: dict[str, Any], classifier_localization: dict[str, Any], patchcore_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    patchcore_localization = patchcore_metrics.get("localization", {})
    return [
        {
            "Method": "ResNet18 classifier + Grad-CAM",
            "Learning setting": "Supervised binary classification",
            "Training data used": "train/good + train/defect",
            "Defect labels used for training": "Yes",
            "Supervised checkpoint used": "Yes",
            "Accuracy": classifier_metrics["accuracy"],
            "Precision": classifier_metrics["precision"],
            "Recall": classifier_metrics["recall"],
            "F1-score": classifier_metrics["f1_score"],
            "Localization IoU": classifier_localization.get("mean_cam_iou"),
            "Localization Dice": classifier_localization.get("mean_cam_dice"),
            "Pointing-hit rate": classifier_localization.get("pointing_hit_rate"),
            "Main heatmap": "Grad-CAM",
        },
        {
            "Method": "PatchCore anomaly detector",
            "Learning setting": "Unsupervised normal-only anomaly detection",
            "Training data used": patchcore_metrics.get("mvtec_memory_bank_split", "train/good"),
            "Defect labels used for training": "No",
            "Supervised checkpoint used": patchcore_metrics.get("supervised_classifier_checkpoint_used", False),
            "Accuracy": patchcore_metrics["accuracy"],
            "Precision": patchcore_metrics["precision"],
            "Recall": patchcore_metrics["recall"],
            "F1-score": patchcore_metrics["f1_score"],
            "Localization IoU": patchcore_localization.get("mean_anomaly_iou"),
            "Localization Dice": patchcore_localization.get("mean_anomaly_dice"),
            "Pointing-hit rate": patchcore_localization.get("pointing_hit_rate"),
            "Main heatmap": "PatchCore anomaly map",
        },
    ]


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(value) for key, value in row.items()})


def write_markdown(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row[header]) for header in headers) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    classifier_metrics = load_json(args.classifier_metrics)
    classifier_localization = load_json(args.classifier_localization)
    patchcore_metrics = load_json(args.patchcore_metrics)

    rows = method_rows(classifier_metrics, classifier_localization, patchcore_metrics)
    write_csv(rows, args.output_csv)
    write_markdown(rows, args.output_md)

    print(f"Saved method comparison CSV to {args.output_csv}")
    print(f"Saved method comparison Markdown to {args.output_md}")


if __name__ == "__main__":
    main()
