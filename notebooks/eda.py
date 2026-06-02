"""
Home Credit Default Risk — Exploratory Data Analysis

Run from project root:
    python notebooks/eda.py

Outputs figures to documents/eda/
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_application_train
from src.utils.helpers import ensure_dir, missing_summary

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
OUTPUT_DIR = PROJECT_ROOT / "documents" / "eda"
PALETTE = {0: "#2E86AB", 1: "#E94F37", "0": "#2E86AB", "1": "#E94F37"}
TARGET_LABELS = {0: "No difficulty", 1: "Payment difficulties", "0": "No difficulty", "1": "Payment difficulties"}
PALETTE_LIST = ["#2E86AB", "#E94F37"]
FIG_DPI = 150

plt.rcParams.update(
    {
        "figure.dpi": FIG_DPI,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "font.family": "sans-serif",
    }
)
sns.set_theme(style="whitegrid", font_scale=1.05)


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except ImportError:
        return False


def save_fig(name: str) -> None:
    path = OUTPUT_DIR / f"{name}.png"
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    if _in_notebook():
        plt.show()
    plt.close()
    print(f"  Saved {path.relative_to(PROJECT_ROOT)}")


def add_age_years(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["AGE_YEARS"] = (-out["DAYS_BIRTH"] / 365.25).round(1)
    return out


def credit_income_ratio(df: pd.DataFrame) -> pd.Series:
    return df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)


# ---------------------------------------------------------------------------
# 1. Dataset overview
# ---------------------------------------------------------------------------
def dataset_overview(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("1. DATASET OVERVIEW")
    print("=" * 72)
    n_rows, n_cols = df.shape
    mem_mb = df.memory_usage(deep=True).sum() / 1024**2
    n_numeric = df.select_dtypes(include=np.number).shape[1]
    n_object = df.select_dtypes(include=["object", "str"]).shape[1]

    print(f"  Applications (rows):     {n_rows:>12,}")
    print(f"  Features (columns):      {n_cols:>12,}")
    print(f"  Numeric features:        {n_numeric:>12,}")
    print(f"  Categorical features:    {n_object:>12,}")
    print(f"  Memory usage:            {mem_mb:>12.1f} MB")
    print(f"  Duplicate SK_ID_CURR:    {df['SK_ID_CURR'].duplicated().sum():>12,}")

    overview = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "mem_mb": mem_mb,
        "default_rate": df["TARGET"].mean(),
    }

    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 3, width_ratios=[1.2, 1, 1])
    ax0 = fig.add_subplot(gs[0])
    ax0.axis("off")
    summary_text = textwrap.dedent(
        f"""
        Home Credit — Application Train

        Records:      {n_rows:,}
        Variables:    {n_cols:,}
        Default rate: {overview['default_rate']:.2%}
        Memory:       {mem_mb:.1f} MB

        TARGET = 1 → client had repayment difficulties
        TARGET = 0 → loan repaid normally
        """
    ).strip()
    ax0.text(0.05, 0.5, summary_text, va="center", fontsize=11, family="monospace")

    ax1 = fig.add_subplot(gs[1])
    dtype_counts = pd.Series(
        {"Numeric": n_numeric, "Categorical": n_object}
    )
    colors = ["#2E86AB", "#A23B72"]
    ax1.pie(dtype_counts, labels=dtype_counts.index, autopct="%1.0f%%", colors=colors, startangle=90)
    ax1.set_title("Feature types")

    ax2 = fig.add_subplot(gs[2])
    key_stats = df[["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY"]].median()
    ax2.barh(key_stats.index.str.replace("AMT_", ""), key_stats.values / 1e3, color="#F18F01")
    ax2.set_xlabel("Median (thousands)")
    ax2.set_title("Key amount medians (k)")

    plt.suptitle("Dataset overview", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig("01_dataset_overview")
    return overview


# ---------------------------------------------------------------------------
# 2. Missing values
# ---------------------------------------------------------------------------
def missing_value_analysis(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 72)
    print("2. MISSING VALUE ANALYSIS")
    print("=" * 72)
    missing = missing_summary(df).sort_values("pct", ascending=False)
    cols_with_missing = len(missing)
    print(f"  Columns with any missing: {cols_with_missing} / {df.shape[1]}")
    print(f"  Max missing %:            {missing['pct'].max():.1f}%")
    print("\n  Top 10 columns by missing %:")
    print(missing.head(10).to_string())

    top_n = 20
    plot_df = missing.head(top_n).sort_values("pct")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].barh(plot_df.index, plot_df["pct"], color="#6C757D", edgecolor="white")
    axes[0].set_xlabel("Missing (%)")
    axes[0].set_title(f"Top {top_n} columns by missing rate")
    axes[0].axvline(50, color="#E94F37", ls="--", lw=1, label="50% threshold")
    axes[0].legend()

    miss_bins = pd.cut(
        missing["pct"],
        bins=[0, 5, 20, 50, 100],
        labels=["0–5%", "5–20%", "20–50%", ">50%"],
    )
    miss_bins.value_counts().sort_index().plot(
        kind="bar", ax=axes[1], color=["#2E86AB", "#F4A261", "#E9C46A", "#E94F37"], edgecolor="white"
    )
    axes[1].set_title("Distribution of missing rates (columns with gaps)")
    axes[1].set_ylabel("Number of columns")
    axes[1].set_xlabel("Missing rate band")
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=0)

    plt.suptitle("Missing value analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig("02_missing_values")
    return missing


# ---------------------------------------------------------------------------
# 3. Target distribution
# ---------------------------------------------------------------------------
def target_distribution(df: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("3. TARGET DISTRIBUTION")
    print("=" * 72)
    counts = df["TARGET"].value_counts().sort_index()
    rate = df["TARGET"].mean()
    print(f"  Class 0 (no difficulty):  {counts.get(0, 0):>10,}  ({1-rate:.2%})")
    print(f"  Class 1 (difficulties):   {counts.get(1, 0):>10,}  ({rate:.2%})")
    print(f"  Imbalance ratio:          1 : {counts.get(0,1)/max(counts.get(1,1),1):.1f}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    bars = axes[0].bar(
        [TARGET_LABELS[i] for i in counts.index],
        counts.values,
        color=[PALETTE[i] for i in counts.index],
        edgecolor="white",
        width=0.55,
    )
    axes[0].set_ylabel("Number of applications")
    axes[0].set_title("Target class counts")
    for bar, v in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v + 2000, f"{v:,}", ha="center", fontsize=10)

    axes[1].pie(
        counts,
        labels=[TARGET_LABELS[i] for i in counts.index],
        autopct="%1.1f%%",
        colors=[PALETTE[i] for i in counts.index],
        explode=(0, 0.06),
        startangle=90,
    )
    axes[1].set_title("Share of portfolio")

    contract = df.groupby("NAME_CONTRACT_TYPE")["TARGET"].mean().sort_values(ascending=False)
    contract.plot(kind="barh", ax=axes[2], color="#264653", edgecolor="white")
    axes[2].set_xlabel("Default rate")
    axes[2].set_title("Default rate by contract type")
    axes[2].axvline(rate, color="#E94F37", ls="--", label=f"Overall {rate:.1%}")
    axes[2].legend(loc="lower right")

    plt.suptitle("Target variable (payment difficulties)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig("03_target_distribution")


# ---------------------------------------------------------------------------
# 4. Income analysis
# ---------------------------------------------------------------------------
def income_analysis(df: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("4. INCOME ANALYSIS")
    print("=" * 72)
    for label, g in df.groupby("TARGET"):
        print(f"  Median income (TARGET={label}): ${g['AMT_INCOME_TOTAL'].median():,.0f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    plot_df = df[df["AMT_INCOME_TOTAL"] < df["AMT_INCOME_TOTAL"].quantile(0.99)]
    sns.histplot(
        data=plot_df,
        x="AMT_INCOME_TOTAL",
        hue="TARGET",
        hue_order=[0, 1],
        palette=PALETTE,
        kde=True,
        ax=axes[0, 0],
        bins=50,
        alpha=0.55,
        multiple="layer",
    )
    axes[0, 0].set_xlabel("Total income")
    axes[0, 0].set_title("Income distribution by target (99th pct cap)")
    axes[0, 0].legend(title="Target", labels=[TARGET_LABELS[0], TARGET_LABELS[1]])

    sns.boxplot(
        data=df,
        x="TARGET",
        y="AMT_INCOME_TOTAL",
        hue="TARGET",
        palette=PALETTE_LIST,
        legend=False,
        ax=axes[0, 1],
        showfliers=False,
    )
    axes[0, 1].set_xticks([0, 1])
    axes[0, 1].set_xticklabels([TARGET_LABELS[0], TARGET_LABELS[1]])
    axes[0, 1].set_title("Income by repayment outcome")

    income_type = (
        df.groupby("NAME_INCOME_TYPE")["TARGET"]
        .agg(["mean", "count"])
        .query("count >= 500")
        .sort_values("mean", ascending=False)
    )
    income_type["mean"].plot(kind="barh", ax=axes[1, 0], color="#457B9D", edgecolor="white")
    axes[1, 0].set_xlabel("Default rate")
    axes[1, 0].set_title("Default rate by income type (n≥500)")

    sns.violinplot(
        data=df[df["NAME_INCOME_TYPE"].isin(income_type.index[:6])],
        y="NAME_INCOME_TYPE",
        x="AMT_INCOME_TOTAL",
        hue="TARGET",
        palette=PALETTE,
        split=True,
        density_norm="count",
        ax=axes[1, 1],
    )
    axes[1, 1].set_xlabel("Income")
    axes[1, 1].set_title("Income spread — top income categories")
    axes[1, 1].legend(title="Target", loc="lower right")

    plt.suptitle("Income analysis", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig("04_income_analysis")


# ---------------------------------------------------------------------------
# 5. Credit amount analysis
# ---------------------------------------------------------------------------
def credit_amount_analysis(df: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("5. CREDIT AMOUNT ANALYSIS")
    print("=" * 72)
    df = df.copy()
    df["CREDIT_INCOME_RATIO"] = credit_income_ratio(df)
    print(f"  Median credit:              ${df['AMT_CREDIT'].median():,.0f}")
    print(f"  Median annuity:             ${df['AMT_ANNUITY'].median():,.0f}")
    print(f"  Median credit/income ratio: {df['CREDIT_INCOME_RATIO'].median():.2f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, col, title in zip(
        axes[0],
        ["AMT_CREDIT", "AMT_ANNUITY"],
        ["Credit amount", "Annuity (installment)"],
    ):
        cap = df[col].quantile(0.99)
        sns.histplot(
            data=df[df[col] <= cap],
            x=col,
            hue="TARGET",
            hue_order=[0, 1],
            palette=PALETTE,
            kde=True,
            ax=ax,
            bins=45,
            alpha=0.55,
        )
        ax.set_title(f"{title} by target")
        ax.legend(title="Target", labels=[TARGET_LABELS[0], TARGET_LABELS[1]])

    sns.scatterplot(
        data=df.sample(min(8000, len(df)), random_state=42),
        x="AMT_INCOME_TOTAL",
        y="AMT_CREDIT",
        hue="TARGET",
        palette=PALETTE,
        alpha=0.35,
        s=12,
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("Credit vs income (sample)")
    axes[1, 0].legend(title="Target", loc="upper left")

    ratio_cap = df["CREDIT_INCOME_RATIO"].quantile(0.99)
    sns.boxplot(
        data=df[df["CREDIT_INCOME_RATIO"] <= ratio_cap],
        x="TARGET",
        y="CREDIT_INCOME_RATIO",
        hue="TARGET",
        palette=PALETTE_LIST,
        legend=False,
        ax=axes[1, 1],
        showfliers=False,
    )
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_xticklabels([TARGET_LABELS[0], TARGET_LABELS[1]])
    axes[1, 1].set_ylabel("Credit / income")
    axes[1, 1].set_title("Leverage ratio by outcome")
    axes[1, 1].axhline(1, color="gray", ls=":", lw=1)

    plt.suptitle("Credit & leverage analysis", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig("05_credit_analysis")


# ---------------------------------------------------------------------------
# 6. Correlation heatmap
# ---------------------------------------------------------------------------
def correlation_heatmap(df: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("6. CORRELATION HEATMAP")
    print("=" * 72)
    key_cols = [
        "TARGET",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "CNT_CHILDREN",
        "CNT_FAM_MEMBERS",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "REGION_RATING_CLIENT",
    ]
    available = [c for c in key_cols if c in df.columns]
    corr = df[available].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
    )
    ax.set_title("Correlation matrix — key credit variables", pad=12)
    plt.tight_layout()
    save_fig("06_correlation_heatmap")

    target_corr = corr["TARGET"].drop("TARGET").sort_values(key=abs, ascending=False)
    print("  Strongest correlations with TARGET:")
    for col, r in target_corr.head(5).items():
        print(f"    {col:25s}  r = {r:+.3f}")


# ---------------------------------------------------------------------------
# 7. Default vs non-default comparison
# ---------------------------------------------------------------------------
def default_comparison(df: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("7. DEFAULT VS NON-DEFAULT COMPARISON")
    print("=" * 72)
    df = add_age_years(df)

    compare_cols = [
        ("AGE_YEARS", "Age (years)"),
        ("AMT_INCOME_TOTAL", "Income"),
        ("AMT_CREDIT", "Credit amount"),
        ("EXT_SOURCE_1", "EXT_SOURCE_1"),
        ("EXT_SOURCE_2", "EXT_SOURCE_2"),
        ("EXT_SOURCE_3", "EXT_SOURCE_3"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for ax, (col, title) in zip(axes, compare_cols):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        plot_data = df[[col, "TARGET"]].dropna()
        if col == "AMT_INCOME_TOTAL":
            cap = plot_data[col].quantile(0.99)
            plot_data = plot_data[plot_data[col] <= cap]
        sns.kdeplot(
            data=plot_data,
            x=col,
            hue="TARGET",
            hue_order=[0, 1],
            palette=PALETTE,
            fill=True,
            alpha=0.4,
            ax=ax,
            linewidth=2,
        )
        ax.set_title(title)
        ax.legend(title="", labels=[TARGET_LABELS[0], TARGET_LABELS[1]])

    plt.suptitle("Feature distributions: default vs non-default", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig("07_default_comparison")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    gender = pd.crosstab(df["CODE_GENDER"], df["TARGET"], normalize="index")[1]
    gender.plot(kind="bar", ax=axes[0], color=["#2E86AB", "#E94F37"], edgecolor="white")
    axes[0].set_ylabel("Default rate")
    axes[0].set_title("Default rate by gender")
    axes[0].set_xlabel("")
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=0)

    edu = (
        df.groupby("NAME_EDUCATION_TYPE")["TARGET"]
        .mean()
        .sort_values(ascending=True)
    )
    edu.plot(kind="barh", ax=axes[1], color="#1D3557", edgecolor="white")
    axes[1].set_xlabel("Default rate")
    axes[1].set_title("Default rate by education")

    plt.suptitle("Demographic & segment default rates", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig("08_segment_comparison")


# ---------------------------------------------------------------------------
# 8. Business insights (computed from data)
# ---------------------------------------------------------------------------
def business_insights(df: pd.DataFrame) -> list[str]:
    print("\n" + "=" * 72)
    print("8. BUSINESS INSIGHTS")
    print("=" * 72)

    df = add_age_years(df)
    df["CREDIT_INCOME_RATIO"] = credit_income_ratio(df)
    default_rate = df["TARGET"].mean()

    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    ext_mean = df[ext_cols].mean(axis=1)
    low_ext = ext_mean < ext_mean.quantile(0.25)
    high_ext = ext_mean >= ext_mean.quantile(0.75)
    dr_low_ext = df.loc[low_ext, "TARGET"].mean()
    dr_high_ext = df.loc[high_ext, "TARGET"].mean()

    young = df["AGE_YEARS"] < 35
    dr_young = df.loc[young, "TARGET"].mean()
    dr_old = df.loc[~young, "TARGET"].mean()

    ratio_q = df["CREDIT_INCOME_RATIO"].quantile([0.25, 0.75])
    low_ratio = df["CREDIT_INCOME_RATIO"] <= ratio_q.iloc[0]
    high_ratio = df["CREDIT_INCOME_RATIO"] >= ratio_q.iloc[1]
    dr_low_ratio = df.loc[low_ratio, "TARGET"].mean()
    dr_high_ratio = df.loc[high_ratio, "TARGET"].mean()

    unemployed_mask = df["DAYS_EMPLOYED"] == 365243  # Home Credit sentinel
    if unemployed_mask.sum() > 100:
        dr_unemp = df.loc[unemployed_mask, "TARGET"].mean()
        unemp_insight = (
            f"Clients flagged as unemployed (special DAYS_EMPLOYED code) show a "
            f"default rate of {dr_unemp:.1%} vs portfolio average {default_rate:.1%}."
        )
    else:
        unemp_insight = (
            f"Unemployment flag is rare; portfolio default rate is {default_rate:.1%} "
            f"with severe class imbalance (roughly 1:{int((1-default_rate)/default_rate)} negative:positive)."
        )

    cash = df[df["NAME_CONTRACT_TYPE"] == "Cash loans"]["TARGET"].mean()
    rev = df[df["NAME_CONTRACT_TYPE"] == "Revolving loans"]["TARGET"].mean()

    insights = [
        f"Portfolio default rate is {default_rate:.2%} (~{int(default_rate*len(df)):,} "
        f"distressed loans of {len(df):,}), indicating strong class imbalance for modeling.",
        f"External bureau scores are highly discriminative: bottom-quartile combined EXT_SOURCE "
        f"clients default at {dr_low_ext:.1%} vs {dr_high_ext:.1%} in the top quartile "
        f"— prioritize score completion and imputation in underwriting.",
        f"Younger applicants (<35 years) default at {dr_young:.1%} compared with "
        f"{dr_old:.1%} for older clients — age-adjusted pricing or limits may reduce losses.",
        f"Credit-to-income ratio varies widely (median {df['CREDIT_INCOME_RATIO'].median():.1f}x): "
        f"bottom-quartile leverage defaults at {dr_low_ratio:.1%} vs {dr_high_ratio:.1%} "
        f"in the top quartile — combine leverage with bureau scores, not income alone.",
        f"Cash loans default at {cash:.1%} vs revolving loans at {rev:.1%}; product-specific "
        f"models and monitoring beat a single portfolio-wide scorecard.",
        unemp_insight,
    ]

    for i, text in enumerate(insights[:6], 1):
        wrapped = textwrap.fill(text, width=68)
        print(f"\n  Insight {i}:\n  {wrapped.replace(chr(10), chr(10) + '  ')}")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    body = "\n\n".join(f"{i}. {t}" for i, t in enumerate(insights[:6], 1))
    ax.text(
        0.02,
        0.98,
        "Business insights — Home Credit application train",
        fontsize=14,
        fontweight="bold",
        va="top",
    )
    ax.text(0.02, 0.88, body, fontsize=10.5, va="top", wrap=True, linespacing=1.6)
    plt.tight_layout()
    save_fig("09_business_insights")
    return insights


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ensure_dir(OUTPUT_DIR)
    print("Loading application_train.csv ...")
    df = load_application_train()
    df["TARGET"] = df["TARGET"].astype(int)
    print(f"Loaded {len(df):,} rows × {df.shape[1]} columns")

    dataset_overview(df)
    missing_value_analysis(df)
    target_distribution(df)
    income_analysis(df)
    credit_amount_analysis(df)
    correlation_heatmap(df)
    default_comparison(df)
    business_insights(df)

    print("\n" + "=" * 72)
    print(f"EDA complete. Figures saved to: {OUTPUT_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
