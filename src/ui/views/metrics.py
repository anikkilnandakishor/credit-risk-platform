"""Model evaluation metrics page."""

import streamlit as st

from src.ui.state import get_artifact, model_available


def render() -> None:
    st.subheader("Model performance")

    if not model_available():
        st.error("Train a model first: `python -m src.ml.train`")
        return

    artifact = get_artifact()
    metrics = artifact.get("metrics", {})
    metrics_default = artifact.get("metrics_default_threshold", {})

    st.markdown("#### Hold-out test metrics (tuned threshold)")
    if metrics:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")
        c2.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
        c3.metric("Precision", f"{metrics.get('precision', 0):.4f}")
        c4.metric("Recall", f"{metrics.get('recall', 0):.4f}")
        c5.metric("F1", f"{metrics.get('f1', 0):.4f}")

        if "gini" in metrics:
            st.metric("Gini coefficient", f"{metrics['gini']:.4f}")

        st.markdown("**Decision threshold:** " + str(metrics.get("threshold", "N/A")))

        cm = metrics.get("confusion_matrix")
        if cm is not None:
            st.markdown("#### Confusion matrix")
            import pandas as pd

            st.dataframe(
                pd.DataFrame(
                    cm,
                    index=["Actual 0", "Actual 1"],
                    columns=["Predicted 0", "Predicted 1"],
                ),
                use_container_width=True,
            )

        if metrics.get("classification_report"):
            with st.expander("Classification report"):
                st.text(metrics["classification_report"])
    else:
        st.warning("No metrics stored in model artifact.")

    if metrics_default:
        st.markdown("---")
        st.markdown("#### Metrics at default threshold (0.5)")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("ROC-AUC", f"{metrics_default.get('roc_auc', 0):.4f}")
        d2.metric("Precision", f"{metrics_default.get('precision', 0):.4f}")
        d3.metric("Recall", f"{metrics_default.get('recall', 0):.4f}")
        d4.metric("F1", f"{metrics_default.get('f1', 0):.4f}")

    st.markdown("---")
    st.markdown("#### Model metadata")
    meta = {
        "Model type": artifact.get("model_type", "unknown"),
        "Features": len(artifact.get("feature_names", [])),
        "Scale pos weight": artifact.get("scale_pos_weight"),
        "Imbalance ratio": artifact.get("imbalance_ratio"),
        "Decision threshold": artifact.get("decision_threshold"),
    }
    for k, v in meta.items():
        if v is not None:
            st.write(f"**{k}:** {v}")
