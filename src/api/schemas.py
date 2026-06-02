"""Pydantic request/response models for the REST API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Single loan application for risk scoring."""

    amt_income_total: float = Field(..., gt=0, examples=[200000.0])
    amt_credit: float = Field(..., gt=0, examples=[450000.0])
    amt_annuity: float = Field(..., ge=0, examples=[22000.0])
    age_years: int = Field(35, ge=18, le=100)
    code_gender: str = Field("M", pattern="^[MFX]$")
    name_contract_type: str = Field("Cash loans")
    name_income_type: str = Field("Working")
    cnt_children: int = Field(0, ge=0)
    ext_source_1: float | None = Field(None, ge=0, le=1)
    ext_source_2: float | None = Field(None, ge=0, le=1)
    ext_source_3: float | None = Field(None, ge=0, le=1)


class PredictBatchRequest(BaseModel):
    """Multiple applications (same schema as single predict)."""

    applications: list[PredictRequest] = Field(..., min_length=1, max_length=100)


class PredictResponse(BaseModel):
    default_probability: float
    prediction: int
    risk_label: str
    threshold: float


class PredictBatchResponse(BaseModel):
    count: int
    high_risk_count: int
    results: list[PredictResponse]


class MetricsResponse(BaseModel):
    model_loaded: bool
    model_type: str | None = None
    decision_threshold: float | None = None
    scale_pos_weight: float | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        examples=["What is the average income of defaulters?"],
    )


class ChatResponse(BaseModel):
    question: str
    sql: str
    answer: str
    data: list[dict[str, Any]]
    row_count: int
