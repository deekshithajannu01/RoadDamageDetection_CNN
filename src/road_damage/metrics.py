"""Stable extraction of useful metrics from an Ultralytics validator result."""

from __future__ import annotations

from typing import Any

import numpy as np

from road_damage.constants import CLASS_NAMES


def _array(value: Any, size: int, default: float = 0.0) -> np.ndarray:
    if value is None:
        return np.full(size, default, dtype=float)
    result = np.asarray(value, dtype=float).reshape(-1)
    if len(result) < size:
        result = np.pad(result, (0, size - len(result)), constant_values=default)
    return result[:size]


def per_class_scores_from_confusion(
    matrix: np.ndarray, class_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate per-class precision, recall, and F1 at one operating point."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape[0] < class_count + 1 or matrix.shape[1] < class_count + 1:
        raise ValueError(f"Unexpected confusion matrix shape: {matrix.shape}")
    true_positive = np.diag(matrix[:class_count, :class_count])
    predicted = matrix[:class_count, :].sum(axis=1)
    actual = matrix[:, :class_count].sum(axis=0)
    precision = np.divide(true_positive, predicted, out=np.zeros(class_count), where=predicted > 0)
    recall = np.divide(true_positive, actual, out=np.zeros(class_count), where=actual > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(class_count),
        where=(precision + recall) > 0,
    )
    return precision, recall, f1


def scores_from_confusion(matrix: np.ndarray, class_count: int) -> dict[str, float]:
    """Calculate macro scores from a raw detection confusion matrix."""
    precision, recall, f1 = per_class_scores_from_confusion(matrix, class_count)
    return {
        "macro_f1": float(f1.mean()),
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
    }


def extract_detection_metrics(metrics: Any) -> dict[str, object]:
    box = metrics.box
    size = len(CLASS_NAMES)
    class_indices = np.asarray(getattr(box, "ap_class_index", np.arange(size)), dtype=int).reshape(
        -1
    )
    precision = np.zeros(size, dtype=float)
    recall = np.zeros(size, dtype=float)
    ap50 = np.zeros(size, dtype=float)
    maps = np.zeros(size, dtype=float)
    for target, source in (
        (precision, _array(getattr(box, "p", None), len(class_indices))),
        (recall, _array(getattr(box, "r", None), len(class_indices))),
        (ap50, _array(getattr(box, "ap50", None), len(class_indices))),
        (maps, _array(getattr(box, "ap", None), len(class_indices))),
    ):
        valid = class_indices[(class_indices >= 0) & (class_indices < size)]
        target[valid] = source[: len(valid)]
    support = _array(getattr(metrics, "nt_per_class", None), size).astype(int)
    per_class = {}
    for index, name in enumerate(CLASS_NAMES):
        denominator = precision[index] + recall[index]
        per_class[name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(2 * precision[index] * recall[index] / denominator) if denominator else 0.0,
            "map50": float(ap50[index]),
            "map50_95": float(maps[index]),
            "support": int(support[index]),
        }
    return {
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "mean_precision": float(box.mp),
        "mean_recall": float(box.mr),
        "per_class": per_class,
    }
