from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models


@dataclass(frozen=True)
class PatchCoreResult:
    image_scores: list[float]
    anomaly_maps: list[np.ndarray]
    metadata: list[dict[str, str]]


class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
        except AttributeError:
            model = models.resnet18(pretrained=pretrained)

        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3

        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(images)
        x = self.layer1(x)
        feature_2 = self.layer2(x)
        feature_3 = self.layer3(feature_2)
        return [feature_2, feature_3]


def patch_embeddings(features: list[torch.Tensor]) -> torch.Tensor:
    pooled = [F.avg_pool2d(feature, kernel_size=3, stride=1, padding=1) for feature in features]
    target_size = pooled[0].shape[-2:]
    aligned = [
        feature if feature.shape[-2:] == target_size else F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)
        for feature in pooled
    ]
    embeddings = torch.cat(aligned, dim=1)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.permute(0, 2, 3, 1).contiguous()


def batch_metadata_to_rows(batch_metadata: dict[str, list[str]]) -> list[dict[str, str]]:
    keys = list(batch_metadata.keys())
    batch_size = len(batch_metadata[keys[0]]) if keys else 0
    return [{key: str(batch_metadata[key][index]) for key in keys} for index in range(batch_size)]


def coreset_subsample(embeddings: torch.Tensor, ratio: float, seed: int) -> torch.Tensor:
    if not 0 < ratio <= 1:
        raise ValueError("--coreset_ratio must be in the range (0, 1].")

    sample_count = max(1, int(round(len(embeddings) * ratio)))
    if sample_count >= len(embeddings):
        return embeddings

    generator = torch.Generator(device=embeddings.device)
    generator.manual_seed(seed)
    indices = torch.randperm(len(embeddings), generator=generator, device=embeddings.device)[:sample_count]
    return embeddings[indices]


def nearest_neighbor_distances(query: torch.Tensor, memory_bank: torch.Tensor, query_chunk_size: int = 1024) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for start in range(0, len(query), query_chunk_size):
        query_chunk = query[start : start + query_chunk_size]
        distances = torch.cdist(query_chunk, memory_bank)
        chunks.append(distances.min(dim=1).values)
    return torch.cat(chunks, dim=0)


def normalize_map(anomaly_map: np.ndarray) -> np.ndarray:
    min_value = float(anomaly_map.min())
    max_value = float(anomaly_map.max())
    if max_value <= min_value:
        return np.zeros_like(anomaly_map, dtype=np.float32)
    return ((anomaly_map - min_value) / (max_value - min_value)).astype(np.float32)


def load_mask(mask_path: str, image_size: int) -> np.ndarray | None:
    if not mask_path:
        return None
    path = Path(mask_path)
    if not path.exists():
        return None
    with Image.open(path) as mask:
        mask = mask.convert("L").resize((image_size, image_size), resample=Image.Resampling.NEAREST)
        return (np.asarray(mask) > 0).astype(np.uint8)


def localization_metrics(anomaly_map: np.ndarray, mask: np.ndarray, threshold: float) -> dict[str, float | bool]:
    predicted = anomaly_map >= threshold
    target = mask.astype(bool)

    intersection = np.logical_and(predicted, target).sum()
    union = np.logical_or(predicted, target).sum()
    predicted_area = predicted.sum()
    target_area = target.sum()
    max_y, max_x = np.unravel_index(int(np.argmax(anomaly_map)), anomaly_map.shape)

    inside = anomaly_map[target]
    outside = anomaly_map[~target]
    return {
        "anomaly_iou": float(intersection / union) if union else 0.0,
        "anomaly_dice": float((2 * intersection) / (predicted_area + target_area)) if predicted_area + target_area else 0.0,
        "pointing_hit": bool(target[max_y, max_x]),
        "mean_anomaly_inside_mask": float(inside.mean()) if inside.size else 0.0,
        "mean_anomaly_outside_mask": float(outside.mean()) if outside.size else 0.0,
    }


def binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    matrix = [[0, 0], [0, 0]]
    for true_label, pred_label in zip(y_true, y_pred, strict=True):
        matrix[true_label][pred_label] += 1

    tn, fp = matrix[0]
    fn, tp = matrix[1]
    total = tn + fp + fn + tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": matrix,
    }


def per_defect_recall(y_true: list[int], y_pred: list[int], metadata: list[dict[str, str]]) -> dict[str, float | None]:
    recalls: dict[str, float | None] = {}
    for defect_type in sorted({row["defect_type"] for row in metadata}):
        indices = [index for index, row in enumerate(metadata) if row["defect_type"] == defect_type]
        positives = [index for index in indices if y_true[index] == 1]
        if not positives:
            recalls[defect_type] = None
            continue
        true_positives = sum(1 for index in positives if y_pred[index] == 1)
        recalls[defect_type] = true_positives / len(positives)
    return recalls


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
