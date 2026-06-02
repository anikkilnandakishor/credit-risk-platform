"""Load Home Credit Default Risk dataset files."""

from pathlib import Path

import pandas as pd

from src.utils.config import APPLICATION_TEST, APPLICATION_TRAIN, DATA_DIR, TARGET_COLUMN
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_csv(path: Path, nrows: int | None = None, **kwargs) -> pd.DataFrame:
    """Load a CSV file with logging and optional row limit."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. "
            f"Place Home Credit files in {DATA_DIR}."
        )
    if nrows is not None:
        logger.info("Loading %s (first %s rows)", path.name, f"{nrows:,}")
    else:
        logger.info("Loading %s (all rows)", path.name)
    return pd.read_csv(path, nrows=nrows, **kwargs)


def load_application_train(nrows: int | None = None) -> pd.DataFrame:
    """Load training applications (includes TARGET)."""
    return load_csv(APPLICATION_TRAIN, nrows=nrows)


def load_application_test(nrows: int | None = None) -> pd.DataFrame:
    """Load test applications (no TARGET)."""
    return load_csv(APPLICATION_TEST, nrows=nrows)


def load_train_test(
    train_path: Path | None = None,
    test_path: Path | None = None,
    train_nrows: int | None = None,
    test_nrows: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test application tables."""
    train = load_csv(train_path or APPLICATION_TRAIN, nrows=train_nrows)
    test = load_csv(test_path or APPLICATION_TEST, nrows=test_nrows)
    return train, test


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features and target from training data."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected column '{TARGET_COLUMN}' in training data.")
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])
    return X, y
