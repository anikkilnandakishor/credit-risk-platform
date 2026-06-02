"""Convert natural language questions to SQL using OpenAI or Gemini."""

from src.talk_to_data.llm_client import complete
from src.talk_to_data.prompt_templates import (
    build_nl_to_sql_system_prompt,
    build_nl_to_sql_user_prompt,
)
from src.talk_to_data.security import PROMPT_INJECTION_SYSTEM_GUARD
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _clean_sql(raw: str) -> str:
    """Strip markdown fences and whitespace from model output."""
    sql = raw.strip()
    if sql.lower().startswith("```"):
        lines = sql.splitlines()
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        sql = "\n".join(lines).strip()
    return sql.rstrip(";")


def nl_to_sql(question: str, schema: str, *, delimited_question: str | None = None) -> str:
    """
    Generate SQLite from a natural language question.

    Uses OpenAI or Gemini based on LLM_PROVIDER and available API keys.
    `delimited_question` should come from layer 2 security wrapping when available.
    """
    user_question = delimited_question or question
    system_prompt = build_nl_to_sql_system_prompt(PROMPT_INJECTION_SYSTEM_GUARD)
    user_prompt = build_nl_to_sql_user_prompt(user_question, schema)
    raw = complete(system_prompt, user_prompt, temperature=0.0)
    sql = _clean_sql(raw)
    logger.info("Generated SQL: %s", sql[:300])
    return sql
