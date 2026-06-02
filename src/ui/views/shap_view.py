"""SHAP explainability section."""

import matplotlib.pyplot as plt
import streamlit as st

import pandas as pd

from src.ml.shap_explainer import compute_shap_values, load_model_and_features
from src.ui.state import model_available
from src.utils.config import PROJECT_ROOT

SHAP_IMG_DIR = PROJECT_ROOT / "documents" / "shap"


@st.cache_resource(show_spinner="Computing SHAP values (may take a minute)…")
def _cached_shap(n_samples: int):
    model, X_df, raw_df, feature_names, artifact = load_model_and_features(n_samples=n_samples)
    _, shap_values, expected = compute_shap_values(model, X_df)
    mean_abs = abs(shap_values).mean(axis=0)
    importance = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return {
        "model": model,
        "X_df": X_df,
        "shap_values": shap_values,
        "expected": expected,
        "feature_names": feature_names,
        "importance": importance,
    }


def _matplotlib_bar_top_features(importance_df):
    fig, ax = plt.subplots(figsize=(8, 5))
    top = importance_df.head(12).sort_values("mean_abs_shap")
    ax.barh(top["feature"], top["mean_abs_shap"], color="#2563eb")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top feature drivers of default risk")
    plt.tight_layout()
    return fig


def render() -> None:
    st.subheader("SHAP explainability")

    if not model_available():
        st.error("Train a model before viewing SHAP explanations.")
        return

    n_samples = st.sidebar.slider(
        "SHAP sample size",
        min_value=500,
        max_value=5000,
        value=1500,
        step=500,
        key="shap_samples",
    )

    with st.spinner("Running SHAP analysis…"):
        try:
            bundle = _cached_shap(n_samples)
        except Exception as e:
            st.error(f"SHAP failed: {e}")
            return

    importance = bundle["importance"]

    tab_imp, tab_plots, tab_waterfall = st.tabs(
        ["Feature importance", "Saved plots", "Waterfall (single loan)"]
    )

    with tab_imp:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.dataframe(importance.head(15), use_container_width=True, hide_index=True)
        with c2:
            st.pyplot(_matplotlib_bar_top_features(importance))

    with tab_plots:
        for name, title in [
            ("feature_importance.png", "Global importance"),
            ("summary_plot.png", "SHAP summary (beeswarm)"),
            ("waterfall_plot.png", "Example waterfall"),
        ]:
            path = SHAP_IMG_DIR / name
            if path.exists():
                st.markdown(f"**{title}**")
                st.image(str(path), use_container_width=True)
            else:
                st.caption(
                    f"`{name}` not found. Generate with: "
                    f"`python -m src.ml.shap_explainer --samples {n_samples}`"
                )

    with tab_waterfall:
        st.caption("Explain one scored application from the SHAP sample.")
        idx = st.number_input(
            "Row index in SHAP sample",
            min_value=0,
            max_value=len(bundle["X_df"]) - 1,
            value=0,
        )
        if st.button("Generate waterfall"):
            import shap

            explanation = shap.Explanation(
                values=bundle["shap_values"][idx],
                base_values=bundle["expected"],
                data=bundle["X_df"].iloc[idx].values,
                feature_names=bundle["feature_names"],
            )
            shap.plots.waterfall(explanation, max_display=12, show=False)
            st.pyplot(plt.gcf())
            plt.close()
