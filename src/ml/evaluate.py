"""Model evaluation metrics for credit risk."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.helpers import gini_from_auc


def evaluate_model(
    model,
    X,
    y,
    threshold: float = 0.5,
    y_proba: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Compute classification metrics on hold-out data.

    Parameters
    ----------
    model : fitted classifier with predict_proba
    X : array-like feature matrix
    y : array-like true labels
    threshold : decision threshold for positive class
    y_proba : optional precomputed probabilities
    """
    y = np.asarray(y)
    proba = y_proba if y_proba is not None else model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)

    auc = roc_auc_score(y, proba)
    cm = confusion_matrix(y, preds)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "roc_auc": float(auc),
        "gini": float(gini_from_auc(auc)),
        "confusion_matrix": cm,
        "predictions": preds,
        "probabilities": proba,
        "classification_report": classification_report(y, preds),
    }


def print_evaluation_report(metrics: dict[str, Any]) -> None:
    """Print metrics and confusion matrix to stdout."""
    cm = metrics["confusion_matrix"]

    print("\n" + "=" * 60)
    print("MODEL EVALUATION (test set)")
    print("=" * 60)
    print(f"  Threshold: {metrics.get('threshold', 0.5):.4f}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-score:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"  Gini:      {metrics['gini']:.4f}")
    print("\nConfusion matrix:")
    print(f"                 Predicted 0    Predicted 1")
    print(f"  Actual 0      {cm[0, 0]:>10,}    {cm[0, 1]:>10,}")
    print(f"  Actual 1      {cm[1, 0]:>10,}    {cm[1, 1]:>10,}")
    print("\nClassification report:")
    print(metrics["classification_report"])
    print("=" * 60)


def format_metrics(metrics: dict) -> str:
    """Human-readable metrics summary."""
    return (
        f"Threshold: {metrics.get('threshold', 0.5):.4f}\n"
        f"Accuracy:  {metrics['accuracy']:.4f}\n"
        f"Precision: {metrics['precision']:.4f}\n"
        f"Recall:    {metrics['recall']:.4f}\n"
        f"F1:        {metrics['f1']:.4f}\n"
        f"ROC-AUC:   {metrics['roc_auc']:.4f}"
    )
