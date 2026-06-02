"""
End-to-end NL-to-SQL pipeline with local fallback and 5-layer security.

Usage:
    python -m src.talk_to_data.nl_engine "How many high-risk customers exist?"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import pandas as pd

from src.data.database import CreditRiskDatabase
from src.talk_to_data.answer_generator import generate_business_answer
from src.talk_to_data.local_chatbot import try_local_chat
from src.talk_to_data.llm_client import (
    LLMNotConfiguredError,
    LLMQuotaError,
    LLMServiceError,
)
from src.talk_to_data.nl_to_sql import nl_to_sql
from src.talk_to_data.query_runner import get_schema_description
from src.talk_to_data.security import (
    SecurityError,
    layer3_validate_sql,
    layer4_execute_query,
    secure_question_pipeline,
    secure_results,
)
from src.utils.config import DB_PATH as CONFIG_DB_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TalkToDataResult:
    """Response from a natural language data question."""

    question: str
    sql: str
    data: pd.DataFrame
    answer: str
    source: str = "llm"  # "local" | "llm" | "local_fallback"
    security_passed: bool = True
    security_layers: list[str] = field(
        default_factory=lambda: [
            "input_validation",
            "prompt_injection_guard",
            "sql_hardening",
            "safe_execution",
            "output_sanitization",
        ]
    )

    def __str__(self) -> str:
        return (
            f"Question: {self.question}\n\n"
            f"SQL:\n{self.sql}\n\n"
            f"Answer ({self.source}):\n{self.answer}"
        )


def ensure_database_ready() -> None:
    """Verify customers table exists and has data."""
    if not CONFIG_DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found at {CONFIG_DB_PATH}. "
            "Run: python -m src.data.database --load"
        )
    db = CreditRiskDatabase()
    try:
        n = db.table_row_count()
    except Exception as exc:
        raise RuntimeError(
            f"Customers table not ready. Run: python -m src.data.database --load\n"
            f"Original error: {exc}"
        ) from exc
    if n == 0:
        raise RuntimeError(
            "Customers table is empty. Run: python -m src.data.database --load"
        )


def _ask_via_llm(
    sanitized_question: str,
    delimited_question: str,
    use_llm_answer: bool,
) -> TalkToDataResult:
    schema = get_schema_description()
    raw_sql = nl_to_sql(
        sanitized_question,
        schema,
        delimited_question=delimited_question,
    )
    safe_sql = layer3_validate_sql(raw_sql)
    df = layer4_execute_query(safe_sql)
    answer = generate_business_answer(
        sanitized_question,
        safe_sql,
        df,
        use_llm=use_llm_answer,
    )
    answer, df, safe_sql = secure_results(answer, df, safe_sql)
    return TalkToDataResult(
        question=sanitized_question,
        sql=safe_sql,
        data=df,
        answer=answer,
        source="llm",
    )


def ask(
    question: str,
    use_llm_answer: bool = True,
    prefer_local: bool = True,
) -> TalkToDataResult:
    """
    Answer a business question: local predefined SQL first, then LLM NL→SQL.

    Never raises LLM errors to callers when a local path or friendly message suffices.
    """
    ensure_database_ready()

    # Layer 1 validation always
    try:
        sanitized_question, delimited_question, _audit = secure_question_pipeline(question)
    except SecurityError:
        raise

    # 1) Local predefined queries (keyword → SQL)
    if prefer_local:
        local = try_local_chat(sanitized_question)
        if local is not None:
            sq, sql, df, answer = local
            answer, df, sql = secure_results(answer, df, sql)
            return TalkToDataResult(
                question=sq,
                sql=sql,
                data=df,
                answer=answer,
                source="local",
                security_layers=["input_validation", "sql_hardening", "safe_execution", "output_sanitization"],
            )

    # 2) LLM path with graceful degradation
    try:
        return _ask_via_llm(sanitized_question, delimited_question, use_llm_answer)
    except LLMQuotaError as exc:
        logger.warning("LLM quota/rate error: %s", exc)
        retry = try_local_chat(sanitized_question)
        if retry is not None:
            sq, sql, df, answer = retry
            answer, df, sql = secure_results(answer, df, sql)
            answer = (
                f"{answer}\n\n*(LLM quota exceeded — answered from local knowledge base.)*"
            )
            return TalkToDataResult(
                question=sq,
                sql=sql,
                data=df,
                answer=answer,
                source="local_fallback",
            )
        answer, df, sql = secure_results(
            "The AI service quota was exceeded and this question is not in the local "
            "knowledge base. Try one of the example questions or rephrase.",
            pd.DataFrame(),
            "",
        )
        return TalkToDataResult(
            question=sanitized_question,
            sql="",
            data=df,
            answer=answer,
            source="local_fallback",
        )
    except LLMNotConfiguredError:
        logger.info("No LLM configured; local-only mode")
        retry = try_local_chat(sanitized_question)
        if retry is not None:
            sq, sql, df, answer = retry
            answer, df, sql = secure_results(answer, df, sql)
            return TalkToDataResult(
                question=sq,
                sql=sql,
                data=df,
                answer=answer,
                source="local",
            )
        answer, df, _ = secure_results(
            "No API key is configured and this question is not recognized locally. "
            "Add OPENAI_API_KEY or GEMINI_API_KEY to .env, or try an example question.",
            pd.DataFrame(),
            "",
        )
        return TalkToDataResult(
            question=sanitized_question,
            sql="",
            data=df,
            answer=answer,
            source="local",
        )
    except (LLMServiceError, Exception) as exc:
        logger.exception("LLM pipeline failed: %s", exc)
        retry = try_local_chat(sanitized_question)
        if retry is not None:
            sq, sql, df, answer = retry
            answer, df, sql = secure_results(answer, df, sql)
            answer = f"{answer}\n\n*(AI unavailable — local fallback used.)*"
            return TalkToDataResult(
                question=sq,
                sql=sql,
                data=df,
                answer=answer,
                source="local_fallback",
            )
        answer, df, _ = secure_results(
            f"I could not complete your request. Error: {exc}. "
            "Try an example question or check the database is loaded.",
            pd.DataFrame(),
            "",
        )
        return TalkToDataResult(
            question=sanitized_question,
            sql="",
            data=df,
            answer=answer,
            source="local_fallback",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions about Home Credit data")
    parser.add_argument("question", nargs="?", help="Natural language question")
    parser.add_argument("--no-llm-answer", action="store_true")
    parser.add_argument("--examples", action="store_true")
    args = parser.parse_args()

    examples = [
        "How many high-risk customers exist?",
        "What is the average income of defaulters?",
        "Which age group has highest default rate?",
        "Which loan type is riskiest?",
        "Give me a portfolio risk summary.",
    ]
    questions = examples if args.examples else ([args.question] if args.question else [])
    if not questions:
        parser.error("Provide a question or use --examples")

    for q in questions:
        print("\n" + "=" * 72)
        try:
            result = ask(q, use_llm_answer=not args.no_llm_answer)
            print(result)
            if not result.data.empty:
                print("\nData preview:")
                print(result.data.to_string(index=False))
        except SecurityError as exc:
            print(f"SECURITY BLOCK: {exc}")
        except Exception as exc:
            print(f"ERROR: {exc}")
        print("=" * 72)


if __name__ == "__main__":
    main()
