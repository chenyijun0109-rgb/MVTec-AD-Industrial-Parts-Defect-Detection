import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


LABEL_NAMES = {0: "good", 1: "defect"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create prediction visualization grids.")
    parser.add_argument("--predictions_path", type=Path, default=Path("results/predictions.csv"))
    parser.add_argument("--output_dir", type=Path, default=Path("results"))
    parser.add_argument("--max_images", type=int, default=12)
    return parser.parse_args()


def read_predictions(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions file: {path}. Run src/evaluate.py first.")
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def make_grid(rows: list[dict[str, str]], output_path: Path, title: str, max_images: int) -> None:
    selected = rows[:max_images]
    cols = 4
    rows_count = max(1, (len(selected) + cols - 1) // cols)
    fig, axes = plt.subplots(rows_count, cols, figsize=(cols * 3.2, rows_count * 3.6))
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax in axes_list:
        ax.axis("off")

    if not selected:
        axes_list[0].text(0.5, 0.5, "No samples", ha="center", va="center", fontsize=14)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return

    for ax, row in zip(axes_list, selected, strict=False):
        image = Image.open(row["image_path"]).convert("RGB")
        ax.imshow(image)
        ax.axis("off")
        true_label = LABEL_NAMES[int(row["true_label"])]
        pred_label = LABEL_NAMES[int(row["pred_label"])]
        confidence = float(row["confidence"])
        color = "green" if row["is_correct"] == "True" else "red"
        ax.set_title(
            f"true: {true_label}\npred: {pred_label} ({confidence:.2f})\n{row['defect_type']}",
            fontsize=9,
            color=color,
        )

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = read_predictions(args.predictions_path)

    correct = [row for row in predictions if row["is_correct"] == "True"]
    incorrect = [row for row in predictions if row["is_correct"] == "False"]

    make_grid(correct, args.output_dir / "sample_predictions.png", "Correct Predictions", args.max_images)
    make_grid(incorrect, args.output_dir / "misclassified_samples.png", "Misclassified Samples", args.max_images)

    print(f"Saved sample predictions to {args.output_dir / 'sample_predictions.png'}")
    print(f"Saved misclassified samples to {args.output_dir / 'misclassified_samples.png'}")


if __name__ == "__main__":
    main()
