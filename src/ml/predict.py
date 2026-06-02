"""Inference for credit default risk."""

from pathlib import Path

import joblib
import pandas as pd

from src.data.preprocessor import transform_inference
from src.utils.config import MODEL_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_artifact(path: Path | None = None) -> dict:
    """Load saved model artifact."""
    path = path or MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run: python -m src.ml.train"
        )
    return joblib.load(path)


def predict_proba(df: pd.DataFrame, artifact: dict | None = None) -> pd.DataFrame:
    """
    Return default probability per SK_ID_CURR.

    Expects raw application features (same schema as training).
    """
    artifact = artifact or load_artifact()
    model = artifact["model"]
    preprocessor = artifact["preprocessor"]

    ids = df["SK_ID_CURR"] if "SK_ID_CURR" in df.columns else df.index
    X = transform_inference(df, preprocessor)
    proba = model.predict_proba(X)[:, 1]

    return pd.DataFrame(
        {
            "SK_ID_CURR": ids.values if hasattr(ids, "values") else ids,
            "DEFAULT_PROBABILITY": proba,
        }
    )


def predict_batch(
    df: pd.DataFrame,
    threshold: float | None = None,
    artifact: dict | None = None,
) -> pd.DataFrame:
    """Binary predictions at tuned or custom threshold."""
    artifact = artifact or load_artifact()
    if threshold is None:
        threshold = artifact.get("decision_threshold", 0.5)
    result = predict_proba(df, artifact=artifact)
    result["PREDICTION"] = (result["DEFAULT_PROBABILITY"] >= threshold).astype(int)
    return result
