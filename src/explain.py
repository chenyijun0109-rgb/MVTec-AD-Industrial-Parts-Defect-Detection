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
from torchvision import transforms

from dataset import LABEL_NAMES, MVTecBottleDataset
from model import build_resnet18


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM explanations and mask-level localization metrics.")
    parser.add_argument("--metadata_path", type=Path, default=Path("results/metadata.csv"))
    parser.add_argument("--model_path", type=Path, default=Path("results/best_model.pth"))
    parser.add_argument("--output_dir", type=Path, default=Path("results"))
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--target_class", type=int, default=1, choices=[0, 1], help="Class used for Grad-CAM.")
    parser.add_argument("--cam_threshold", type=float, default=0.5, help="Threshold on normalized CAM for IoU/Dice.")
    parser.add_argument("--max_visualizations", type=int, default=8)
    return parser.parse_args()


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activations)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _save_gradients(
        self,
        _module: torch.nn.Module,
        _grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        self.gradients = grad_output[0].detach()

    def __call__(self, image_tensor: torch.Tensor, target_class: int) -> tuple[np.ndarray, np.ndarray]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)
        score = logits[:, target_class].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)

        cam_np = cam.squeeze().cpu().numpy()
        cam_np = normalize_array(cam_np)
        probs = torch.softmax(logits.detach(), dim=1).squeeze().cpu().numpy()
        return cam_np, probs

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()


def normalize_array(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32)
    min_value = float(array.min())
    max_value = float(array.max())
    if max_value <= min_value:
        return np.zeros_like(array, dtype=np.float32)
    return (array - min_value) / (max_value - min_value)


def load_model(model_path: Path, image_size: int, device: torch.device) -> tuple[torch.nn.Module, int]:
    checkpoint = torch.load(model_path, map_location=device)
    checkpoint_image_size = int(checkpoint.get("image_size", image_size))
    model = build_resnet18(num_classes=2, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint_image_size


def load_image_tensor(image_path: Path, image_size: int, device: torch.device) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
        ]
    )
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        return transform(image).unsqueeze(0).to(device)


def load_display_image(image_path: Path, image_size: int) -> np.ndarray:
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize((image_size, image_size))
        return np.asarray(image, dtype=np.float32) / 255.0


def load_mask(mask_path: str, image_size: int) -> np.ndarray | None:
    if not mask_path:
        return None
    path = Path(mask_path)
    if not path.exists():
        return None
    with Image.open(path) as mask:
        mask = mask.convert("L").resize((image_size, image_size), resample=Image.Resampling.NEAREST)
        return (np.asarray(mask) > 0).astype(np.uint8)


def localization_metrics(cam: np.ndarray, mask: np.ndarray, threshold: float) -> dict[str, float | bool]:
    cam_binary = cam >= threshold
    mask_binary = mask.astype(bool)

    intersection = np.logical_and(cam_binary, mask_binary).sum()
    union = np.logical_or(cam_binary, mask_binary).sum()
    cam_area = cam_binary.sum()
    mask_area = mask_binary.sum()

    max_y, max_x = np.unravel_index(int(np.argmax(cam)), cam.shape)
    pointing_hit = bool(mask_binary[max_y, max_x])
    inside_values = cam[mask_binary]
    outside_values = cam[~mask_binary]

    return {
        "cam_iou": float(intersection / union) if union else 0.0,
        "cam_dice": float((2 * intersection) / (cam_area + mask_area)) if cam_area + mask_area else 0.0,
        "pointing_hit": pointing_hit,
        "mean_cam_inside_mask": float(inside_values.mean()) if inside_values.size else 0.0,
        "mean_cam_outside_mask": float(outside_values.mean()) if outside_values.size else 0.0,
    }


def overlay_heatmap(image: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    cmap = plt.get_cmap("jet")
    heatmap = cmap(cam)[..., :3]
    return np.clip((1 - alpha) * image + alpha * heatmap, 0, 1)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    localized = [row for row in rows if row["has_mask"]]
    summary: dict[str, Any] = {
        "num_samples": len(rows),
        "num_localization_samples": len(localized),
    }
    if not localized:
        return summary

    summary["mean_cam_iou"] = float(np.mean([row["cam_iou"] for row in localized]))
    summary["mean_cam_dice"] = float(np.mean([row["cam_dice"] for row in localized]))
    summary["mean_cam_activation_inside_mask"] = float(np.mean([row["mean_cam_inside_mask"] for row in localized]))
    summary["mean_cam_activation_outside_mask"] = float(np.mean([row["mean_cam_outside_mask"] for row in localized]))
    summary["pointing_hit_rate"] = float(np.mean([1.0 if row["pointing_hit"] else 0.0 for row in localized]))

    by_type: dict[str, dict[str, float]] = {}
    for defect_type in sorted({row["defect_type"] for row in localized}):
        subset = [row for row in localized if row["defect_type"] == defect_type]
        by_type[defect_type] = {
            "count": len(subset),
            "mean_cam_iou": float(np.mean([row["cam_iou"] for row in subset])),
            "mean_cam_dice": float(np.mean([row["cam_dice"] for row in subset])),
            "pointing_hit_rate": float(np.mean([1.0 if row["pointing_hit"] else 0.0 for row in subset])),
        }
    summary["per_defect_type"] = by_type
    return summary


def save_visualization(rows: list[dict[str, Any]], output_path: Path, image_size: int, max_visualizations: int) -> None:
    selected = [row for row in rows if row["has_mask"]][:max_visualizations]
    cols = 4
    rows_count = max(1, len(selected))
    fig, axes = plt.subplots(rows_count, cols, figsize=(cols * 3.2, rows_count * 2.9))
    axes_array = np.atleast_2d(axes)

    if not selected:
        for ax in axes_array.flatten():
            ax.axis("off")
        axes_array[0, 0].text(0.5, 0.5, "No masked defect samples", ha="center", va="center")
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return

    for row_index, row in enumerate(selected):
        image = load_display_image(Path(row["image_path"]), image_size)
        mask = load_mask(row["mask_path"], image_size)
        cam = np.load(row["cam_path"])
        overlay = overlay_heatmap(image, cam)

        panels = [
            ("Image", image),
            ("Ground-truth mask", mask if mask is not None else np.zeros_like(cam)),
            ("Grad-CAM", cam),
            (f"Overlay IoU={row['cam_iou']:.2f}", overlay),
        ]
        for col_index, (title, panel) in enumerate(panels):
            ax = axes_array[row_index, col_index]
            ax.imshow(panel, cmap="gray" if col_index in (1, 2) else None, vmin=0, vmax=1)
            ax.set_title(title, fontsize=9)
            ax.axis("off")

    fig.suptitle("Grad-CAM vs Ground-truth Defect Masks", fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985), h_pad=1.4, w_pad=1.0)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cam_dir = args.output_dir / "gradcam"
    cam_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, image_size = load_model(args.model_path, args.image_size, device)
    dataset = MVTecBottleDataset(args.metadata_path, split=args.split, image_size=image_size, train_transforms=False)
    grad_cam = GradCAM(model, model.layer4[-1])

    rows: list[dict[str, Any]] = []
    try:
        for sample in dataset.samples:
            image_path = Path(sample["image_path"])
            image_tensor = load_image_tensor(image_path, image_size, device)
            cam, probs = grad_cam(image_tensor, target_class=args.target_class)
            pred_label = int(np.argmax(probs))
            mask = load_mask(sample["mask_path"], image_size)
            cam_path = cam_dir / f"{image_path.stem}_{sample['defect_type']}_cam.npy"
            np.save(cam_path, cam)

            row: dict[str, Any] = {
                "image_path": sample["image_path"],
                "mask_path": sample["mask_path"],
                "defect_type": sample["defect_type"],
                "true_label": int(sample["binary_label"]),
                "pred_label": pred_label,
                "pred_label_name": LABEL_NAMES[pred_label],
                "defect_probability": float(probs[1]),
                "has_mask": mask is not None,
                "cam_path": cam_path.as_posix(),
            }
            if mask is not None:
                row.update(localization_metrics(cam, mask, args.cam_threshold))
            else:
                row.update(
                    {
                        "cam_iou": "",
                        "cam_dice": "",
                        "pointing_hit": "",
                        "mean_cam_inside_mask": "",
                        "mean_cam_outside_mask": "",
                    }
                )
            rows.append(row)
    finally:
        grad_cam.close()

    csv_path = args.output_dir / "gradcam_localization.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    summary_path = args.output_dir / "localization_metrics.json"
    with summary_path.open("w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, indent=2)

    visualization_path = args.output_dir / "gradcam_mask_overlay.png"
    save_visualization(rows, visualization_path, image_size, args.max_visualizations)

    print(json.dumps(summary, indent=2))
    print(f"Saved Grad-CAM arrays to {cam_dir}")
    print(f"Saved localization rows to {csv_path}")
    print(f"Saved localization summary to {summary_path}")
    print(f"Saved Grad-CAM visualization to {visualization_path}")


if __name__ == "__main__":
    main()
