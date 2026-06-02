"""
Credit Risk Intelligence Platform — Streamlit dashboard.

Run: streamlit run app.py
"""

import streamlit as st

from src.ui import styles
from src.ui.sidebar_nav import render_navigation
from src.ui.state import model_available
from src.ui.views import chatbot, dashboard, metrics, prediction, shap_view
from src.utils.config import MODEL_PATH

PAGES = {
    "Dashboard": dashboard.render,
    "Risk prediction": prediction.render,
    "Model metrics": metrics.render,
    "SHAP explainability": shap_view.render,
    "Data chatbot": chatbot.render,
}

st.set_page_config(
    page_title="Credit Risk Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    styles.inject_styles()

    with st.sidebar:
        st.markdown("## 🏦 Credit Risk AI")
        st.markdown("*Home Credit intelligence*")
        st.markdown("---")

        page = render_navigation(list(PAGES.keys()))

        st.markdown("---")
        st.markdown("**System status**")
        if model_available():
            st.success(f"Model loaded")
            st.caption(str(MODEL_PATH.name))
        else:
            st.error("Model missing")
            st.caption("Run: python -m src.ml.train")

        st.markdown("---")
        st.markdown(
            """
            **Capabilities**
            - Portfolio analytics
            - XGBoost scoring
            - SHAP explanations
            - NL → SQL chatbot
            """
        )

    titles = {
        "Dashboard": (
            "Executive dashboard",
            "Real-time portfolio insights and risk distributions",
        ),
        "Risk prediction": (
            "Loan risk prediction",
            "Score individual applications or batch files with the saved XGBoost model",
        ),
        "Model metrics": (
            "Model evaluation",
            "Classification metrics and confusion matrix from training",
        ),
        "SHAP explainability": (
            "Model explainability",
            "Understand which features drive default predictions",
        ),
        "Data chatbot": (
            "AI data assistant",
            "Ask business questions in plain English — secured NL to SQL",
        ),
    }
    title, subtitle = titles.get(page, ("Credit Risk Intelligence", ""))
    styles.page_header(title, subtitle)

    PAGES[page]()


if __name__ == "__main__":
    main()
