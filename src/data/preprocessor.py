"""Feature preprocessing pipeline for Home Credit Default Risk."""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.utils.config import RANDOM_STATE, TARGET_COLUMN
from src.utils.logger import get_logger

logger = get_logger(__name__)

ID_COLUMN = "SK_ID_CURR"


def identify_column_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split numeric and categorical feature columns."""
    exclude = {ID_COLUMN, TARGET_COLUMN}
    cols = [c for c in df.columns if c not in exclude]
    numeric = df[cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical = df[cols].select_dtypes(include=["object", "str"]).columns.tolist()
    return numeric, categorical


def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> ColumnTransformer:
    """
    Build sklearn ColumnTransformer:
      - numeric: median imputation + StandardScaler
      - categorical: most-frequent imputation + ordinal (label) encoding
    """
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "label_encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    dtype=np.int32,
                ),
            ),
        ]
    )
    transformers = []
    if numeric_cols:
        transformers.append(("numeric", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", categorical_pipe, categorical_cols))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Training data must include '{TARGET_COLUMN}' column.")
    y = df[TARGET_COLUMN].astype(int)
    X = df.drop(columns=[TARGET_COLUMN])
    if ID_COLUMN in X.columns:
        X = X.drop(columns=[ID_COLUMN])
    return X, y


def _apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
    smote_k_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    minority_count = int(np.bincount(y_train).min())
    k = min(smote_k_neighbors, minority_count - 1)
    if k < 1:
        logger.warning(
            "Skipping SMOTE: not enough minority samples (count=%d).",
            minority_count,
        )
        return X_train, y_train

    logger.info("Applying SMOTE (k_neighbors=%d) on training set ...", k)
    smote = SMOTE(random_state=random_state, k_neighbors=k)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    logger.info(
        "After SMOTE — train size: %d, class balance: %s",
        len(y_res),
        dict(zip(*np.unique(y_res, return_counts=True))),
    )
    return X_res, y_res


def prepare_train_test_splits(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int | None = None,
    apply_smote: bool = True,
    smote_k_neighbors: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Full preprocessing workflow for model training.

    Steps:
      1. Stratified train-test split
      2. Fit imputers, label encoders (OrdinalEncoder), and scalers on train only
      3. Transform train and test
      4. Apply SMOTE on training set only (imbalanced-learn)

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
    """
    X_train, X_test, y_train, y_test, _, _ = prepare_train_test_splits_with_preprocessor(
        df,
        test_size=test_size,
        random_state=random_state,
        apply_smote=apply_smote,
        smote_k_neighbors=smote_k_neighbors,
    )
    return X_train, X_test, y_train, y_test


def prepare_train_test_splits_with_preprocessor(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int | None = None,
    apply_smote: bool = True,
    smote_k_neighbors: int = 5,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, ColumnTransformer, np.ndarray
]:
    """
    Same as prepare_train_test_splits but also returns the fitted preprocessor
    (for inference and model persistence).
    """
    random_state = RANDOM_STATE if random_state is None else random_state
    X, y = _split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    numeric_cols, categorical_cols = identify_column_types(X_train)
    logger.info(
        "Fitting preprocessor on train set (%d rows): %d numeric, %d categorical columns",
        len(X_train),
        len(numeric_cols),
        len(categorical_cols),
    )

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    X_train_arr = preprocessor.fit_transform(X_train)
    X_test_arr = preprocessor.transform(X_test)

    y_train_arr = np.asarray(y_train)
    y_test_arr = np.asarray(y_test)
    y_train_before_smote = y_train_arr.copy()

    if apply_smote:
        X_train_arr, y_train_arr = _apply_smote(
            X_train_arr, y_train_arr, random_state, smote_k_neighbors
        )

    return X_train_arr, X_test_arr, y_train_arr, y_test_arr, preprocessor, y_train_before_smote


def transform_inference(
    df: pd.DataFrame,
    preprocessor: ColumnTransformer,
) -> np.ndarray:
    """Transform raw application rows using a fitted preprocessor."""
    X = df.copy()
    if ID_COLUMN in X.columns:
        X = X.drop(columns=[ID_COLUMN])
    if TARGET_COLUMN in X.columns:
        X = X.drop(columns=[TARGET_COLUMN])
    return preprocessor.transform(X)
