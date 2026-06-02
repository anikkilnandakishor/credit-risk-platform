"""XGBoost training pipeline for Home Credit Default Risk."""

from __future__ import annotations

import argparse
import joblib
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from src.ml.evaluate import evaluate_model, print_evaluation_report
from src.ml.imbalance import find_optimal_threshold, print_class_imbalance
from src.ml.training_data import load_processed_splits
from src.ml.xgboost_trainer import train_xgboost_classifier
from src.utils.config import MODEL_PATH, RANDOM_STATE
from src.utils.helpers import ensure_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_pipeline(
    model_path: Path | None = None,
    test_size: float = 0.2,
    apply_smote: bool = False,
    max_rows: int | None = 50_000,
    **xgb_kwargs: Any,
) -> dict[str, Any]:
    """
    End-to-end training with imbalance-aware XGBoost:
      - scale_pos_weight from pre-SMOTE training labels
      - ROC-AUC early stopping
      - regularized hyperparameters
      - F1-optimized threshold for improved minority recall
    """
    model_path = model_path or MODEL_PATH
    ensure_dir(model_path.parent)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        cat_cols,
        num_cols,
        y_train_before_smote,
    ) = load_processed_splits(
        test_size=test_size,
        apply_smote=apply_smote,
        max_rows=max_rows,
    )

    imbalance = print_class_imbalance(y_train_before_smote, label="training (pre-SMOTE)")
    scale_pos_weight = imbalance["scale_pos_weight"]

    # Validation fold for early stopping + threshold tuning (stratified)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    model = train_xgboost_classifier(
        X_tr,
        y_tr,
        X_val,
        y_val,
        scale_pos_weight=scale_pos_weight,
        **xgb_kwargs,
    )

    val_proba = model.predict_proba(X_val)[:, 1]
    optimal_threshold, threshold_score = find_optimal_threshold(
        y_val, val_proba, metric="f2", min_precision=0.12
    )
    logger.info(
        "Optimal threshold (F2, min precision 0.12 on validation): %.4f (score=%.4f)",
        optimal_threshold,
        threshold_score,
    )

    test_proba = model.predict_proba(X_test)[:, 1]
    metrics_default = evaluate_model(
        model, X_test, y_test, threshold=0.5, y_proba=test_proba
    )
    metrics_tuned = evaluate_model(
        model, X_test, y_test, threshold=optimal_threshold, y_proba=test_proba
    )

    print("\n--- Metrics at default threshold (0.5) ---")
    print_evaluation_report(metrics_default)

    print("\n--- Metrics at tuned threshold (recall-focused) ---")
    print_evaluation_report(metrics_tuned)

    artifact = {
        "model": model,
        "preprocessor": preprocessor,
        "feature_names": list(preprocessor.get_feature_names_out()),
        "categorical_columns": cat_cols,
        "numeric_columns": num_cols,
        "scale_pos_weight": scale_pos_weight,
        "imbalance_ratio": imbalance["imbalance_ratio"],
        "decision_threshold": optimal_threshold,
        "metrics": {
            k: v
            for k, v in metrics_tuned.items()
            if k not in ("predictions", "probabilities")
        },
        "metrics_default_threshold": {
            k: v
            for k, v in metrics_default.items()
            if k not in ("predictions", "probabilities")
        },
        "model_type": "xgboost",
    }
    joblib.dump(artifact, model_path)
    logger.info("Model saved to %s", model_path)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train XGBoost credit default risk model"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=MODEL_PATH,
        help="Output path (default: models/model.pkl)",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--smote",
        action="store_true",
        help="Apply SMOTE (off by default; scale_pos_weight handles imbalance)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=50_000,
        help="Subsample rows for faster training (0 = full dataset)",
    )
    args = parser.parse_args()

    max_rows = args.max_rows if args.max_rows > 0 else None
    train_pipeline(
        model_path=args.output,
        test_size=args.test_size,
        apply_smote=args.smote,
        max_rows=max_rows,
    )


if __name__ == "__main__":
    main()
