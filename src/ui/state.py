"""Cached resources for the Streamlit dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.data.loader import load_application_train
from src.ml.predict import load_artifact
from src.utils.config import APPLICATION_TRAIN, MODEL_PATH, RANDOM_STATE


@st.cache_resource(show_spinner="Loading ML model…")
def get_artifact() -> dict | None:
    if not MODEL_PATH.exists():
        return None
    return load_artifact()


@st.cache_data(show_spinner="Loading portfolio sample…")
def get_portfolio_sample(n_rows: int = 25_000) -> pd.DataFrame:
    if not APPLICATION_TRAIN.exists():
        return pd.DataFrame()
    df = load_application_train()
    if len(df) > n_rows:
        df = df.sample(n=n_rows, random_state=RANDOM_STATE)
    return df


@st.cache_data
def get_profile_template() -> pd.Series:
    """Median/mode profile for single-application prediction form."""
    df = get_portfolio_sample(10_000)
    if df.empty:
        return pd.Series(dtype=float)
    row = df.drop(columns=["TARGET"], errors="ignore").iloc[0].copy()
    numeric = df.drop(columns=["TARGET"], errors="ignore").select_dtypes(include="number")
    for col in numeric.columns:
        row[col] = numeric[col].median()
    for col in df.select_dtypes(include="object").columns:
        if col != "TARGET":
            row[col] = df[col].mode().iloc[0] if not df[col].mode().empty else row[col]
    return row


def model_available() -> bool:
    return MODEL_PATH.exists()
