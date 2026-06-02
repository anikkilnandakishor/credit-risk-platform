"""Load and preprocess data for model training."""

from __future__ import annotations

import pandas as pd

from src.data.loader import load_application_train
from src.data.preprocessor import (
    identify_column_types,
    prepare_train_test_splits_with_preprocessor,
)
from src.utils.config import RANDOM_STATE
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_raw_data() -> pd.DataFrame:
    """Load Home Credit application_train.csv."""
    logger.info("Loading raw training data")
    return load_application_train()


def load_processed_splits(
    df: pd.DataFrame | None = None,
    test_size: float = 0.2,
    apply_smote: bool = False,
    random_state: int | None = None,
    max_rows: int | None = None,
):
    """
    Load data, separate features/target, encode categoricals, scale numerics,
    split train/test, and optionally apply SMOTE.

    Returns
    -------
    X_train, X_test, y_train, y_test, preprocessor,
    categorical_cols, numeric_cols, y_train_before_smote
    """
    if df is None:
        df = load_raw_data()

    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=random_state or RANDOM_STATE)
        logger.info("Subsampled to %d rows for training", max_rows)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        y_train_before_smote,
    ) = prepare_train_test_splits_with_preprocessor(
        df,
        test_size=test_size,
        random_state=random_state,
        apply_smote=apply_smote,
    )

    _, categorical_cols, numeric_cols = _get_feature_column_groups(df)
    logger.info(
        "Processed splits — train: %s, test: %s",
        X_train.shape,
        X_test.shape,
    )
    return (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        categorical_cols,
        numeric_cols,
        y_train_before_smote,
    )


def _get_feature_column_groups(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Return all feature, categorical, and numeric column names."""
    from src.data.preprocessor import TARGET_COLUMN, _split_features_target

    X, _ = _split_features_target(df)
    numeric_cols, categorical_cols = identify_column_types(
        pd.concat([X, df[TARGET_COLUMN]], axis=1)
    )
    return list(X.columns), categorical_cols, numeric_cols
