"""Portfolio dashboard with KPIs and charts."""

import plotly.express as px
import streamlit as st

from src.ui.state import get_portfolio_sample


def render() -> None:
    st.subheader("Portfolio overview")
    df = get_portfolio_sample()

    if df.empty:
        st.warning("Training data not found. Add `data/application_train.csv` to view charts.")
        return

    default_rate = df["TARGET"].mean()
    n = len(df)
    n_default = int(df["TARGET"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Applications", f"{n:,}")
    c2.metric("Default rate", f"{default_rate:.2%}")
    c3.metric("Defaulters", f"{n_default:,}")
    c4.metric("Median income", f"${df['AMT_INCOME_TOTAL'].median():,.0f}")

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        target_counts = df["TARGET"].value_counts().rename(
            {0: "Performing", 1: "Default"}
        )
        fig = px.pie(
            values=target_counts.values,
            names=target_counts.index,
            title="Loan outcome mix",
            color_discrete_sequence=["#22c55e", "#ef4444"],
            hole=0.45,
        )
        fig.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=360)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.histogram(
            df,
            x="AMT_INCOME_TOTAL",
            nbins=50,
            title="Income distribution",
            color_discrete_sequence=["#2563eb"],
        )
        fig.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=360)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        by_contract = (
            df.groupby("NAME_CONTRACT_TYPE")["TARGET"]
            .mean()
            .reset_index(name="default_rate")
        )
        fig = px.bar(
            by_contract,
            x="NAME_CONTRACT_TYPE",
            y="default_rate",
            title="Default rate by product",
            color="default_rate",
            color_continuous_scale="Reds",
        )
        fig.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=320)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        df_age = df.copy()
        df_age["age"] = (-df_age["DAYS_BIRTH"] / 365.25).clip(18, 80)
        fig = px.scatter(
            df_age.sample(min(3000, len(df_age)), random_state=42),
            x="age",
            y="AMT_CREDIT",
            color="TARGET",
            color_discrete_map={0: "#22c55e", 1: "#ef4444"},
            opacity=0.5,
            title="Credit exposure vs age",
            labels={"TARGET": "Default"},
        )
        fig.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=320)
        st.plotly_chart(fig, use_container_width=True)

    st.caption("Charts use a stratified sample of application_train for responsiveness.")
