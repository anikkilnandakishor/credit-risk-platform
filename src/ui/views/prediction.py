"""Loan risk prediction form and batch scoring."""

import pandas as pd
import streamlit as st

from src.ml.predict import predict_batch
from src.ui.state import get_artifact, get_profile_template, model_available


def _build_application_from_form(template: pd.Series) -> pd.DataFrame:
    row = template.copy()
    row["AMT_INCOME_TOTAL"] = st.session_state.get("pred_income", row.get("AMT_INCOME_TOTAL"))
    row["AMT_CREDIT"] = st.session_state.get("pred_credit", row.get("AMT_CREDIT"))
    row["AMT_ANNUITY"] = st.session_state.get("pred_annuity", row.get("AMT_ANNUITY"))
    age = st.session_state.get("pred_age", 35)
    row["DAYS_BIRTH"] = -int(age * 365.25)
    row["CODE_GENDER"] = st.session_state.get("pred_gender", row.get("CODE_GENDER", "M"))
    row["NAME_CONTRACT_TYPE"] = st.session_state.get(
        "pred_contract", row.get("NAME_CONTRACT_TYPE", "Cash loans")
    )
    row["NAME_INCOME_TYPE"] = st.session_state.get(
        "pred_income_type", row.get("NAME_INCOME_TYPE", "Working")
    )
    row["CNT_CHILDREN"] = st.session_state.get("pred_children", row.get("CNT_CHILDREN", 0))
    row["EXT_SOURCE_1"] = st.session_state.get("pred_ext1", row.get("EXT_SOURCE_1", 0.5))
    row["EXT_SOURCE_2"] = st.session_state.get("pred_ext2", row.get("EXT_SOURCE_2", 0.5))
    row["EXT_SOURCE_3"] = st.session_state.get("pred_ext3", row.get("EXT_SOURCE_3", 0.5))
    row["SK_ID_CURR"] = 999999
    return pd.DataFrame([row])


def render() -> None:
    st.subheader("Loan risk prediction")

    if not model_available():
        st.error("No saved model found. Train with: `python -m src.ml.train`")
        return

    artifact = get_artifact()
    threshold = artifact.get("decision_threshold", 0.5)

    tab_form, tab_batch = st.tabs(["Single application", "Batch upload"])

    with tab_form:
        template = get_profile_template()
        if template.empty:
            st.warning("Cannot build form without training data.")
            return

        st.caption(
            "Enter key application fields. Remaining features use portfolio medians from training data."
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.number_input(
                "Annual income ($)",
                min_value=0.0,
                value=float(template.get("AMT_INCOME_TOTAL", 150000)),
                step=5000.0,
                key="pred_income",
            )
            st.number_input(
                "Credit amount ($)",
                min_value=0.0,
                value=float(template.get("AMT_CREDIT", 500000)),
                step=10000.0,
                key="pred_credit",
            )
            st.number_input(
                "Annuity / installment ($)",
                min_value=0.0,
                value=float(template.get("AMT_ANNUITY", 25000)),
                step=500.0,
                key="pred_annuity",
            )
        with c2:
            st.number_input("Age (years)", min_value=18, max_value=100, value=35, key="pred_age")
            st.selectbox("Gender", ["M", "F", "X"], key="pred_gender")
            st.selectbox(
                "Contract type",
                ["Cash loans", "Revolving loans"],
                key="pred_contract",
            )
            st.selectbox(
                "Income type",
                ["Working", "Commercial associate", "Pensioner", "State servant", "Student"],
                key="pred_income_type",
            )
        with c3:
            st.number_input("Children", min_value=0, max_value=20, value=0, key="pred_children")
            st.slider("EXT_SOURCE_1", 0.0, 1.0, float(template.get("EXT_SOURCE_1", 0.5) or 0.5), key="pred_ext1")
            st.slider("EXT_SOURCE_2", 0.0, 1.0, float(template.get("EXT_SOURCE_2", 0.5) or 0.5), key="pred_ext2")
            st.slider("EXT_SOURCE_3", 0.0, 1.0, float(template.get("EXT_SOURCE_3", 0.5) or 0.5), key="pred_ext3")

        if st.button("Score application", type="primary", use_container_width=True):
            app_df = _build_application_from_form(template)
            result = predict_batch(app_df, threshold=threshold, artifact=artifact)
            prob = float(result["DEFAULT_PROBABILITY"].iloc[0])
            pred = int(result["PREDICTION"].iloc[0])

            st.markdown("---")
            m1, m2, m3 = st.columns(3)
            m1.metric("Default probability", f"{prob:.1%}")
            m2.metric("Decision threshold", f"{threshold:.2f}")
            risk_label = "HIGH RISK" if pred == 1 else "LOW RISK"
            m3.metric("Classification", risk_label)

            css = "risk-high" if pred == 1 else "risk-low"
            st.markdown(
                f'<p class="{css}">Recommendation: '
                f"{'Review / decline or enhanced monitoring' if pred == 1 else 'Standard approval path'}"
                f"</p>",
                unsafe_allow_html=True,
            )

    with tab_batch:
        st.caption("Upload CSV with the same schema as `application_train.csv` (without TARGET).")
        uploaded = st.file_uploader("Application file", type=["csv"])
        if uploaded is not None:
            batch = pd.read_csv(uploaded)
            with st.spinner("Scoring applications…"):
                results = predict_batch(batch, threshold=threshold, artifact=artifact)
            st.dataframe(results.head(200), use_container_width=True)
            st.download_button(
                "Download scores",
                results.to_csv(index=False),
                "risk_scores.csv",
                "text/csv",
            )
            high_risk = int((results["PREDICTION"] == 1).sum())
            st.info(f"{high_risk:,} of {len(results):,} flagged as high risk.")
