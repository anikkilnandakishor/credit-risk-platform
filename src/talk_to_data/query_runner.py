"""Safe SQL execution for talk-to-data."""

import pandas as pd

from src.data.database import CreditRiskDatabase, validate_select_sql
from src.talk_to_data.security import SecurityError, layer3_validate_sql, layer4_execute_query
from src.utils.logger import get_logger

logger = get_logger(__name__)


def validate_sql(sql: str) -> str:
    """Ensure query passes layer-3 SQL hardening."""
    try:
        return layer3_validate_sql(sql)
    except SecurityError:
        raise
    except ValueError as exc:
        raise SecurityError(f"SQL rejected: {exc}") from exc


def run_query(sql: str, database_url: str | None = None) -> pd.DataFrame:
    """Execute validated SQL via layer-4 safe execution."""
    _ = database_url
    return layer4_execute_query(sql)


def run_query_basic(sql: str) -> pd.DataFrame:
    """Execute with basic validation only (internal / tests)."""
    safe = validate_select_sql(sql)
    return CreditRiskDatabase().execute_query(safe)


def get_schema_description(database_url: str | None = None) -> str:
    """Return table/column listing for prompt context."""
    _ = database_url
    return CreditRiskDatabase().get_schema_description()
