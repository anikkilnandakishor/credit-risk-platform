"""Class imbalance utilities for credit risk modeling."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve


def class_imbalance_stats(y: np.ndarray) -> dict[str, float | int]:
    """
    Compute class counts, imbalance ratio, and XGBoost scale_pos_weight.

    scale_pos_weight = n_negative / n_positive (standard XGBoost convention).
    """
    y = np.asarray(y).astype(int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        raise ValueError("No positive (default) samples in training labels.")

    ratio = n_neg / n_pos
    scale_pos_weight = ratio
    return {
        "n_negative": n_neg,
        "n_positive": n_pos,
        "negative_rate": n_neg / len(y),
        "positive_rate": n_pos / len(y),
        "imbalance_ratio": ratio,
        "scale_pos_weight": scale_pos_weight,
    }


def print_class_imbalance(y: np.ndarray, label: str = "Training set") -> dict[str, float | int]:
    """Print and return imbalance statistics."""
    stats = class_imbalance_stats(y)
    print("\n" + "-" * 60)
    print(f"CLASS IMBALANCE — {label}")
    print("-" * 60)
    print(f"  Negative (0): {stats['n_negative']:,}  ({stats['negative_rate']:.2%})")
    print(f"  Positive (1): {stats['n_positive']:,}  ({stats['positive_rate']:.2%})")
    print(f"  Imbalance ratio (neg/pos): {stats['imbalance_ratio']:.2f} : 1")
    print(f"  scale_pos_weight:          {stats['scale_pos_weight']:.4f}")
    print("-" * 60)
    return stats


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f2",
    min_precision: float = 0.12,
) -> tuple[float, float]:
    """
    Select decision threshold to improve minority-class recall.

    Default: maximize F2 (recall-weighted) among thresholds with precision >= min_precision.
    Falls back to max recall if no threshold meets the precision floor.

    Returns (best_threshold, best_score).
    """
    y_true = np.asarray(y_true)
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    best_threshold = 0.5
    best_score = -1.0
    fallback_threshold = 0.5
    fallback_recall = -1.0

    for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
        if r > fallback_recall:
            fallback_recall = r
            fallback_threshold = float(t)

        if metric == "recall":
            score = r if p >= min_precision else -1.0
        elif metric == "f1":
            score = 2 * p * r / (p + r) if (p + r) > 0 and p >= min_precision else -1.0
        elif metric == "f2":
            # F2 weights recall 2x over precision
            score = (5 * p * r / (4 * p + r)) if (p + r) > 0 and p >= min_precision else -1.0
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = float(t)

    if best_score < 0:
        return fallback_threshold, fallback_recall

    return best_threshold, best_score
