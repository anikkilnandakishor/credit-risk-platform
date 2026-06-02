"""
Local fallback chatbot: keyword matching → predefined SQL → business answers.

Used when a question matches a known pattern, or when LLM APIs are unavailable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from src.talk_to_data.query_runner import run_query
from src.talk_to_data.security import SecurityError, layer1_validate_input, layer3_validate_sql
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", " ", text)
    return re.sub(r"\s+", " ", text)


def _has_any(text: str, *phrases: str) -> bool:
    return any(p in text for p in phrases)


def _has_all(text: str, *phrases: str) -> bool:
    return all(p in text for p in phrases)


@dataclass(frozen=True)
class LocalQueryTemplate:
    """Predefined question pattern and SQL."""

    query_id: str
    description: str
    sql: str
    matcher: Callable[[str], bool]
    answer_builder: Callable[[pd.DataFrame], str]


# ---------------------------------------------------------------------------
# Predefined SQL (validated read-only)
# ---------------------------------------------------------------------------
SQL_HIGH_RISK_COUNT = """
SELECT COUNT(*) AS high_risk_count
FROM customers
WHERE TARGET = 1
LIMIT 500
""".strip()

SQL_AVG_INCOME_DEFAULTERS = """
SELECT ROUND(AVG(AMT_INCOME_TOTAL), 2) AS avg_income_defaulters
FROM customers
WHERE TARGET = 1 AND AMT_INCOME_TOTAL IS NOT NULL
LIMIT 500
""".strip()

SQL_AGE_GROUP_DEFAULT = """
SELECT
  CASE
    WHEN (-DAYS_BIRTH / 365.25) < 30 THEN 'Under 30'
    WHEN (-DAYS_BIRTH / 365.25) < 40 THEN '30-39'
    WHEN (-DAYS_BIRTH / 365.25) < 50 THEN '40-49'
    WHEN (-DAYS_BIRTH / 365.25) < 60 THEN '50-59'
    ELSE '60+'
  END AS age_group,
  ROUND(100.0 * AVG(TARGET), 2) AS default_rate_pct,
  COUNT(*) AS n_customers
FROM customers
GROUP BY age_group
ORDER BY default_rate_pct DESC
LIMIT 1
""".strip()

SQL_RISKIEST_LOAN_TYPE = """
SELECT
  NAME_CONTRACT_TYPE AS loan_type,
  ROUND(100.0 * AVG(TARGET), 2) AS default_rate_pct,
  COUNT(*) AS n_customers
FROM customers
WHERE NAME_CONTRACT_TYPE IS NOT NULL
GROUP BY NAME_CONTRACT_TYPE
ORDER BY default_rate_pct DESC
LIMIT 1
""".strip()

SQL_PORTFOLIO_SUMMARY = """
SELECT
  COUNT(*) AS total_customers,
  SUM(TARGET) AS defaulters,
  ROUND(100.0 * AVG(TARGET), 2) AS default_rate_pct,
  ROUND(AVG(AMT_INCOME_TOTAL), 2) AS avg_income,
  ROUND(AVG(AMT_CREDIT), 2) AS avg_credit,
  ROUND(AVG(AMT_ANNUITY), 2) AS avg_annuity
FROM customers
LIMIT 500
""".strip()


def _answer_high_risk_count(df: pd.DataFrame) -> str:
    n = int(df.iloc[0, 0])
    return (
        f"There are **{n:,}** high-risk customers in the portfolio "
        f"(clients with payment difficulties, TARGET = 1)."
    )


def _answer_avg_income_defaulters(df: pd.DataFrame) -> str:
    v = float(df.iloc[0, 0])
    return (
        f"The average income among defaulters is **${v:,.2f}** "
        f"(mean AMT_INCOME_TOTAL where TARGET = 1)."
    )


def _answer_age_group(df: pd.DataFrame) -> str:
    row = df.iloc[0]
    return (
        f"The age group with the highest default rate is **{row['age_group']}** "
        f"at **{row['default_rate_pct']}%** "
        f"(based on {int(row['n_customers']):,} customers in that band)."
    )


def _answer_riskiest_loan(df: pd.DataFrame) -> str:
    row = df.iloc[0]
    return (
        f"The riskiest loan type is **{row['loan_type']}** "
        f"with a default rate of **{row['default_rate_pct']}%** "
        f"({int(row['n_customers']):,} applications)."
    )


def _answer_portfolio_summary(df: pd.DataFrame) -> str:
    r = df.iloc[0]
    return (
        f"**Portfolio risk summary:** {int(r['total_customers']):,} customers; "
        f"{int(r['defaulters']):,} with payment difficulties "
        f"({r['default_rate_pct']}% default rate). "
        f"Average income ${r['avg_income']:,.0f}, "
        f"average credit ${r['avg_credit']:,.0f}, "
        f"average installment ${r['avg_annuity']:,.0f}."
    )


def _match_portfolio_summary(text: str) -> bool:
    return _has_any(text, "portfolio", "summary", "overview", "snapshot") and _has_any(
        text, "risk", "default", "summary", "overview", "portfolio"
    )


def _match_high_risk_count(text: str) -> bool:
    if _match_portfolio_summary(text):
        return False
    return _has_any(
        text,
        "how many high risk",
        "how many defaulter",
        "how many default",
        "number of defaulter",
        "count of defaulter",
        "high risk customer",
        "high-risk customer",
        "payment difficult",
    ) or (
        _has_any(text, "how many", "count", "number of")
        and _has_any(text, "high risk", "high-risk", "defaulter", "default")
        and not _has_any(text, "income", "age", "loan type", "contract")
    )


def _match_avg_income_defaulters(text: str) -> bool:
    return _has_any(text, "income", "salary", "earning") and _has_any(
        text, "defaulter", "default", "high risk", "payment difficult"
    ) and _has_any(text, "average", "avg", "mean")


def _match_age_group(text: str) -> bool:
    return _has_any(text, "age", "young", "older", "elder") and _has_any(
        text, "default rate", "highest", "most risky", "riskiest", "worst"
    )


def _match_riskiest_loan(text: str) -> bool:
    return _has_any(
        text,
        "loan type",
        "contract type",
        "cash loan",
        "revolving",
        "product",
        "loan",
    ) and _has_any(text, "riskiest", "highest risk", "highest default", "most risky", "worst")


LOCAL_TEMPLATES: list[LocalQueryTemplate] = [
    LocalQueryTemplate(
        "portfolio_summary",
        "Portfolio risk summary",
        SQL_PORTFOLIO_SUMMARY,
        _match_portfolio_summary,
        _answer_portfolio_summary,
    ),
    LocalQueryTemplate(
        "high_risk_count",
        "Count of high-risk customers",
        SQL_HIGH_RISK_COUNT,
        _match_high_risk_count,
        _answer_high_risk_count,
    ),
    LocalQueryTemplate(
        "avg_income_defaulters",
        "Average income of defaulters",
        SQL_AVG_INCOME_DEFAULTERS,
        _match_avg_income_defaulters,
        _answer_avg_income_defaulters,
    ),
    LocalQueryTemplate(
        "age_group_default",
        "Age group with highest default rate",
        SQL_AGE_GROUP_DEFAULT,
        _match_age_group,
        _answer_age_group,
    ),
    LocalQueryTemplate(
        "riskiest_loan_type",
        "Riskiest loan / contract type",
        SQL_RISKIEST_LOAN_TYPE,
        _match_riskiest_loan,
        _answer_riskiest_loan,
    ),
]


def match_local_template(question: str) -> LocalQueryTemplate | None:
    """Return the first predefined template that matches the question."""
    text = _normalize(question)
    for template in LOCAL_TEMPLATES:
        if template.matcher(text):
            logger.info("Local chatbot matched template: %s", template.query_id)
            return template
    return None


def answer_from_local_template(
    question: str,
    template: LocalQueryTemplate | None = None,
) -> tuple[str, str, pd.DataFrame, str]:
    """
    Run a predefined query and build a business answer.

    Returns (sanitized_question, sql, dataframe, answer).
    """
    sanitized = layer1_validate_input(question)
    template = template or match_local_template(sanitized)
    if template is None:
        raise ValueError("No local template matched this question.")

    safe_sql = layer3_validate_sql(template.sql)
    df = run_query(safe_sql)
    answer = template.answer_builder(df)
    answer = f"{answer}\n\n*(Answered from local knowledge base — no API call.)*"
    return sanitized, safe_sql, df, answer


def try_local_chat(question: str) -> tuple[str, str, pd.DataFrame, str] | None:
    """
    Attempt to answer using the local chatbot.

    Returns None if no predefined pattern matches.
    """
    try:
        template = match_local_template(question)
        if template is None:
            return None
        return answer_from_local_template(question, template)
    except SecurityError:
        raise
    except Exception as exc:
        logger.warning("Local chatbot failed: %s", exc)
        return None
