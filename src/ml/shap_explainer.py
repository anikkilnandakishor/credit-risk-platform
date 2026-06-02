"""
SHAP explainability for the trained XGBoost credit risk model.

Usage:
    python -m src.ml.shap_explainer
    python -m src.ml.shap_explainer --samples 2000 --waterfall-index 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.data.loader import load_application_train
from src.data.preprocessor import transform_inference
from src.ml.predict import load_artifact
from src.utils.config import MODEL_PATH, PROJECT_ROOT, RANDOM_STATE
from src.utils.helpers import ensure_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "documents" / "shap"


def load_model_and_features(
    model_path: Path | None = None,
    n_samples: int = 2000,
) -> tuple[object, pd.DataFrame, np.ndarray, list[str], dict]:
    """Load artifact, sample data, and build processed feature matrix."""
    artifact = load_artifact(model_path)
    model = artifact["model"]
    preprocessor = artifact["preprocessor"]
    feature_names = list(
        artifact.get("feature_names") or preprocessor.get_feature_names_out()
    )

    logger.info("Loading %d sample rows for SHAP analysis", n_samples)
    df = load_application_train()
    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=RANDOM_STATE)

    X = transform_inference(df, preprocessor)
    X_df = pd.DataFrame(X, columns=feature_names)
    return model, X_df, df, feature_names, artifact


def compute_shap_values(model, X_df: pd.DataFrame) -> tuple[shap.TreeExplainer, np.ndarray, float]:
    """Generate SHAP values for the positive (default) class."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_df)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    expected = explainer.expected_value
    if isinstance(expected, (list, np.ndarray)):
        expected = float(expected[1]) if len(expected) > 1 else float(expected[0])
    else:
        expected = float(expected)

    logger.info("Computed SHAP values: shape %s", shap_values.shape)
    return explainer, np.asarray(shap_values), expected


def print_top_features(
    shap_values: np.ndarray,
    feature_names: list[str],
    top_n: int = 10,
) -> pd.DataFrame:
    """Print and return top features by mean absolute SHAP value."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    print("\n" + "=" * 60)
    print(f"TOP {top_n} FEATURES BY MEAN |SHAP|")
    print("=" * 60)
    for rank, row in importance.head(top_n).iterrows():
        print(f"  {rank + 1:2d}. {row['feature']:<45} {row['mean_abs_shap']:.6f}")
    print("=" * 60)

    return importance


def plot_feature_importance(
    shap_values: np.ndarray,
    X_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Bar plot of global feature importance (mean |SHAP|)."""
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_df,
        plot_type="bar",
        show=False,
        max_display=20,
    )
    plt.title("SHAP feature importance (mean |SHAP value|)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("Saved feature importance plot: %s", output_path)


def plot_summary(
    shap_values: np.ndarray,
    X_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Beeswarm summary plot of SHAP values."""
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_df, show=False, max_display=20)
    plt.title("SHAP summary plot", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("Saved summary plot: %s", output_path)


def plot_waterfall(
    shap_values: np.ndarray,
    X_df: pd.DataFrame,
    expected_value: float,
    feature_names: list[str],
    row_index: int,
    output_path: Path,
) -> None:
    """Waterfall plot explaining a single prediction."""
    explanation = shap.Explanation(
        values=shap_values[row_index],
        base_values=expected_value,
        data=X_df.iloc[row_index].values,
        feature_names=feature_names,
    )
    shap.plots.waterfall(explanation, max_display=15, show=False)
    plt.title(
        f"SHAP waterfall — sample index {row_index}",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("Saved waterfall plot: %s", output_path)


def pick_waterfall_index(model, X_df: pd.DataFrame, y_proba_row: int | None = None) -> int:
    """Choose a high-risk sample for the waterfall plot unless index is given."""
    if y_proba_row is not None:
        return y_proba_row
    proba = model.predict_proba(X_df)[:, 1]
    return int(np.argmax(proba))


def run_shap_analysis(
    model_path: Path | None = None,
    n_samples: int = 2000,
    waterfall_index: int | None = None,
    top_n: int = 10,
    output_dir: Path | None = None,
) -> dict:
    """
    Full SHAP workflow: compute values, plots, and top-feature report.

    Saves:
      - documents/shap/feature_importance.png
      - documents/shap/summary_plot.png
      - documents/shap/waterfall_plot.png
    """
    output_dir = output_dir or OUTPUT_DIR
    ensure_dir(output_dir)

    model, X_df, _raw_df, feature_names, artifact = load_model_and_features(
        model_path=model_path,
        n_samples=n_samples,
    )
    explainer, shap_values, expected_value = compute_shap_values(model, X_df)

    importance = print_top_features(shap_values, feature_names, top_n=top_n)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)

    plot_feature_importance(
        shap_values,
        X_df,
        output_dir / "feature_importance.png",
    )
    plot_summary(shap_values, X_df, output_dir / "summary_plot.png")

    wf_index = pick_waterfall_index(model, X_df, waterfall_index)
    plot_waterfall(
        shap_values,
        X_df,
        expected_value,
        feature_names,
        wf_index,
        output_dir / "waterfall_plot.png",
    )

    return {
        "explainer": explainer,
        "shap_values": shap_values,
        "expected_value": expected_value,
        "feature_importance": importance,
        "waterfall_index": wf_index,
        "artifact": artifact,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SHAP explainability for credit risk XGBoost model"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_PATH,
        help="Path to models/model.pkl",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="Number of rows to sample for SHAP",
    )
    parser.add_argument(
        "--waterfall-index",
        type=int,
        default=None,
        help="Row index for waterfall plot (default: highest risk score)",
    )
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for SHAP plots",
    )
    args = parser.parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {args.model_path}. Run: python -m src.ml.train"
        )

    run_shap_analysis(
        model_path=args.model_path,
        n_samples=args.samples,
        waterfall_index=args.waterfall_index,
        top_n=args.top_n,
        output_dir=args.output_dir,
    )
    print(f"\nPlots saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
