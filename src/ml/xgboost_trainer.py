"""XGBoost model construction and training."""

from __future__ import annotations

from typing import Any

import numpy as np
from xgboost import XGBClassifier

from src.utils.config import RANDOM_STATE
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Tuned for ROC-AUC with stronger regularization to limit overfitting
DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_estimators": 800,
    "max_depth": 4,
    "learning_rate": 0.03,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "colsample_bylevel": 0.7,
    "min_child_weight": 12,
    "gamma": 0.2,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "max_delta_step": 1,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

EARLY_STOPPING_ROUNDS = 40


def build_xgboost_classifier(
    scale_pos_weight: float,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
    **overrides: Any,
) -> XGBClassifier:
    """Create an XGBoost classifier with imbalance weighting and regularization."""
    params = {**DEFAULT_PARAMS, **overrides}
    return XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=early_stopping_rounds,
        **params,
    )


def train_xgboost_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scale_pos_weight: float,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
    **model_kwargs: Any,
) -> XGBClassifier:
    """Fit XGBoost with scale_pos_weight; early stopping optimizes validation AUC."""
    model = build_xgboost_classifier(
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=early_stopping_rounds,
        **model_kwargs,
    )
    logger.info(
        "Training XGBoost (scale_pos_weight=%.4f, eval_metric=auc) ...",
        scale_pos_weight,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    logger.info("Best iteration: %s", getattr(model, "best_iteration", "n/a"))
    return model
