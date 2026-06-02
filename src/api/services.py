"""Business logic for the FastAPI backend."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.api.schemas import (
    ChatResponse,
    MetricsResponse,
    PredictBatchResponse,
    PredictRequest,
    PredictResponse,
)
from src.data.database import CreditRiskDatabase
from src.data.loader import load_application_train
from src.ml.predict import load_artifact, predict_batch
from src.talk_to_data import ask
from src.talk_to_data.security import layer1_validate_input
from src.utils.config import DB_PATH, MODEL_PATH, RANDOM_STATE


class ModelNotLoadedError(RuntimeError):
    pass


class DatabaseNotReadyError(RuntimeError):
    pass


_artifact: dict[str, Any] | None = None


def load_model() -> None:
    """Load trained model artifact into memory (call on startup)."""
    global _artifact
    if not MODEL_PATH.exists():
        raise ModelNotLoadedError(
            f"Model file not found at {MODEL_PATH}. Run: python -m src.ml.train"
        )
    _artifact = load_artifact(MODEL_PATH)


def unload_model() -> None:
    global _artifact
    _artifact = None


def is_model_loaded() -> bool:
    return _artifact is not None


def _require_artifact() -> dict[str, Any]:
    if _artifact is None:
        raise ModelNotLoadedError("Model is not loaded.")
    return _artifact


def is_database_ready() -> bool:
    if not DB_PATH.exists():
        return False
    try:
        return CreditRiskDatabase().table_row_count() > 0
    except Exception:
        return False


def _inference_template_row() -> pd.Series:
    """Portfolio median/mode row for filling missing features at inference."""
    df = load_application_train()
    if len(df) > 10_000:
        df = df.sample(n=10_000, random_state=RANDOM_STATE)
    row = df.drop(columns=["TARGET"], errors="ignore").iloc[0].copy()
    numeric = df.drop(columns=["TARGET"], errors="ignore").select_dtypes(include="number")
    for col in numeric.columns:
        row[col] = numeric[col].median()
    for col in df.select_dtypes(include="object").columns:
        if col != "TARGET":
            row[col] = df[col].mode().iloc[0] if not df[col].mode().empty else row[col]
    return row


def _application_to_row(req: PredictRequest) -> pd.Series:
    row = _inference_template_row()
    row["AMT_INCOME_TOTAL"] = req.amt_income_total
    row["AMT_CREDIT"] = req.amt_credit
    row["AMT_ANNUITY"] = req.amt_annuity
    row["DAYS_BIRTH"] = -int(req.age_years * 365.25)
    row["CODE_GENDER"] = req.code_gender
    row["NAME_CONTRACT_TYPE"] = req.name_contract_type
    row["NAME_INCOME_TYPE"] = req.name_income_type
    row["CNT_CHILDREN"] = req.cnt_children
    if req.ext_source_1 is not None:
        row["EXT_SOURCE_1"] = req.ext_source_1
    if req.ext_source_2 is not None:
        row["EXT_SOURCE_2"] = req.ext_source_2
    if req.ext_source_3 is not None:
        row["EXT_SOURCE_3"] = req.ext_source_3
    row["SK_ID_CURR"] = 999999
    return row


def predict_application(req: PredictRequest) -> PredictResponse:
    artifact = _require_artifact()
    threshold = float(artifact.get("decision_threshold", 0.5))
    df = pd.DataFrame([_application_to_row(req)])
    scored = predict_batch(df, threshold=threshold, artifact=artifact)
    prob = float(scored["DEFAULT_PROBABILITY"].iloc[0])
    pred = int(scored["PREDICTION"].iloc[0])
    return PredictResponse(
        default_probability=round(prob, 6),
        prediction=pred,
        risk_label="high_risk" if pred == 1 else "low_risk",
        threshold=threshold,
    )


def predict_applications(requests: list[PredictRequest]) -> PredictBatchResponse:
    results = [predict_application(r) for r in requests]
    high_risk = sum(1 for r in results if r.prediction == 1)
    return PredictBatchResponse(
        count=len(results),
        high_risk_count=high_risk,
        results=results,
    )


def get_model_metrics() -> MetricsResponse:
    if not is_model_loaded():
        return MetricsResponse(model_loaded=False, metrics={})
    artifact = _require_artifact()
    return MetricsResponse(
        model_loaded=True,
        model_type=artifact.get("model_type"),
        decision_threshold=artifact.get("decision_threshold"),
        scale_pos_weight=artifact.get("scale_pos_weight"),
        metrics=artifact.get("metrics", {}),
    )


def chat_with_data(question: str) -> ChatResponse:
    if not is_database_ready():
        raise DatabaseNotReadyError(
            "Database not ready. Run: python -m src.data.database --load"
        )
    layer1_validate_input(question)
    result = ask(question)
    data = result.data
    return ChatResponse(
        question=result.question,
        sql=result.sql,
        answer=result.answer,
        data=data.head(100).to_dict(orient="records"),
        row_count=int(len(data)),
    )
