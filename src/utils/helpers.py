"""Shared helper utilities."""

from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: Path) -> Path:
    """Create directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing-value counts and percentages per column."""
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    return pd.DataFrame({"missing": missing, "pct": pct}).query("missing > 0")


def gini_from_auc(auc: float) -> float:
    """Convert ROC-AUC to Gini coefficient."""
    return 2 * auc - 1


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide with fallback when denominator is zero."""
    if denominator == 0:
        return default
    return numerator / denominator
