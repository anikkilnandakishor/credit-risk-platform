"""
SQLite database module for the Home Credit Default Risk dataset.

Usage:
    python -m src.data.database --load
    python -m src.data.database --load --max-rows 10000
    python -m src.data.database --load --max-rows 0          # full dataset
    python -m src.data.database --query "SELECT COUNT(*) FROM customers"
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import pandas as pd
import sqlparse
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine

from src.data.loader import load_csv
from src.utils.config import APPLICATION_TRAIN, DB_PATH
from src.utils.helpers import ensure_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

CUSTOMERS_TABLE = "customers"
DEFAULT_MAX_ROWS = 10_000
INSERT_CHUNK_SIZE = 2_000

FORBIDDEN_SQL = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)


def _progress(msg: str) -> None:
    """Print user-visible progress (CLI) and log."""
    print(msg)
    logger.info(msg.strip())


def get_engine(db_path: Path | None = None, *, fast_bulk: bool = False) -> Engine:
    """Create a SQLAlchemy engine for credit_risk.db."""
    path = db_path or DB_PATH
    ensure_dir(path.parent)
    url = f"sqlite:///{path.resolve()}"
    engine = create_engine(url)

    if fast_bulk:

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_conn, _connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA temp_store=MEMORY")
            cursor.execute("PRAGMA cache_size=-64000")
            cursor.close()

    return engine


def validate_select_sql(sql: str) -> str:
    """Ensure SQL is a read-only SELECT statement."""
    stripped = sql.strip().rstrip(";")
    if FORBIDDEN_SQL.search(stripped):
        raise ValueError("Only SELECT queries are allowed.")
    parsed = sqlparse.parse(stripped)
    if not parsed:
        raise ValueError("Empty or invalid SQL.")
    stmt_type = parsed[0].get_type()
    if stmt_type and stmt_type.upper() != "SELECT" and stmt_type != "UNKNOWN":
        raise ValueError(f"Query type '{stmt_type}' is not allowed. Use SELECT only.")
    return stripped


def _read_training_csv(csv_path: Path, max_rows: int | None) -> pd.DataFrame:
    """Read CSV with pandas nrows for fast development loads."""
    nrows = None if max_rows is None or max_rows <= 0 else max_rows
    if nrows:
        _progress(f"[1/4] Reading first {nrows:,} rows from {csv_path.name} …")
    else:
        _progress(f"[1/4] Reading full file {csv_path.name} (this may take a while) …")

    t0 = time.perf_counter()
    df = load_csv(csv_path, nrows=nrows)
    elapsed = time.perf_counter() - t0
    _progress(
        f"      → {len(df):,} rows × {len(df.columns)} columns loaded in {elapsed:.1f}s"
    )
    return df


def _bulk_insert(
    df: pd.DataFrame,
    engine: Engine,
    table: str,
    if_exists: str,
    chunksize: int = INSERT_CHUNK_SIZE,
) -> None:
    """Optimized chunked insert into SQLite."""
    total_chunks = max(1, (len(df) + chunksize - 1) // chunksize)
    _progress(
        f"[3/4] Inserting into '{table}' ({len(df):,} rows, "
        f"chunksize={chunksize:,}, ~{total_chunks} chunks) …"
    )
    t0 = time.perf_counter()
    df.to_sql(
        table,
        engine,
        if_exists=if_exists,
        index=False,
        chunksize=chunksize,
        method="multi",
    )
    elapsed = time.perf_counter() - t0
    rate = len(df) / elapsed if elapsed > 0 else 0
    _progress(f"      → Insert complete in {elapsed:.1f}s ({rate:,.0f} rows/s)")


class CreditRiskDatabase:
    """Manage the Home Credit SQLite database (credit_risk.db)."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DB_PATH
        self.engine = get_engine(self.db_path)

    def create_customers_table(
        self,
        if_exists: str = "replace",
        max_rows: int | None = DEFAULT_MAX_ROWS,
    ) -> None:
        """Create the customers table from application_train.csv."""
        self.load_application_train(if_exists=if_exists, max_rows=max_rows)

    def load_application_train(
        self,
        csv_path: Path | None = None,
        if_exists: str = "replace",
        max_rows: int | None = DEFAULT_MAX_ROWS,
        chunksize: int = INSERT_CHUNK_SIZE,
    ) -> int:
        """
        Load application_train.csv into the customers table.

        Parameters
        ----------
        max_rows : int or None
            Max rows to read from CSV. None or <=0 reads the entire file.
        chunksize : int
            Rows per INSERT batch (SQLite optimization).

        Returns number of rows loaded.
        """
        csv_path = csv_path or APPLICATION_TRAIN
        _progress(f"Target database: {self.db_path.resolve()}")

        bulk_engine = get_engine(self.db_path, fast_bulk=True)
        df = _read_training_csv(csv_path, max_rows)

        _progress("[2/4] Preparing SQLite connection (WAL + bulk pragmas) …")
        _bulk_insert(df, bulk_engine, CUSTOMERS_TABLE, if_exists, chunksize=chunksize)

        _progress("[4/4] Creating indexes …")
        self.engine = bulk_engine
        self._create_indexes()

        _progress(f"Done — {len(df):,} rows in '{CUSTOMERS_TABLE}' @ {self.db_path.name}")
        return len(df)

    def _create_indexes(self) -> None:
        """Add indexes for common analytics queries."""
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    f"CREATE INDEX IF NOT EXISTS idx_{CUSTOMERS_TABLE}_target "
                    f"ON {CUSTOMERS_TABLE}(TARGET)"
                )
            )
            conn.execute(
                text(
                    f"CREATE INDEX IF NOT EXISTS idx_{CUSTOMERS_TABLE}_income "
                    f"ON {CUSTOMERS_TABLE}(AMT_INCOME_TOTAL)"
                )
            )

    def execute_query(
        self,
        sql: str,
        params: dict | None = None,
        allow_write: bool = False,
    ) -> pd.DataFrame:
        """Execute SQL safely and return results as a pandas DataFrame."""
        if not allow_write:
            sql = validate_select_sql(sql)

        logger.info("Executing query on %s", self.db_path.name)
        with self.engine.connect() as conn:
            return pd.read_sql(text(sql), conn, params=params or {})

    def table_row_count(self, table: str = CUSTOMERS_TABLE) -> int:
        """Return row count for a table."""
        df = self.execute_query(f"SELECT COUNT(*) AS n FROM {table}")
        return int(df["n"].iloc[0])

    def get_schema_description(self, table: str = CUSTOMERS_TABLE) -> str:
        """Return human-readable schema for NL-to-SQL prompts."""
        with self.engine.connect() as conn:
            cols = pd.read_sql(text(f"PRAGMA table_info({table})"), conn)
        if cols.empty:
            return f"Table '{table}' not found. Run load_application_train() first."
        col_list = ", ".join(f"{r['name']} ({r['type']})" for _, r in cols.iterrows())
        return f"- {table}: {col_list}"


def load_csv_to_database(
    csv_path: Path | None = None,
    db_path: Path | None = None,
    if_exists: str = "replace",
    max_rows: int | None = DEFAULT_MAX_ROWS,
) -> int:
    """Convenience function to load application_train.csv into credit_risk.db."""
    db = CreditRiskDatabase(db_path=db_path)
    return db.load_application_train(
        csv_path=csv_path,
        if_exists=if_exists,
        max_rows=max_rows,
    )


def run_query(
    sql: str,
    db_path: Path | None = None,
    params: dict | None = None,
) -> pd.DataFrame:
    """Execute a safe SELECT query and return a DataFrame."""
    return CreditRiskDatabase(db_path=db_path).execute_query(sql, params=params)


def main() -> None:
    parser = argparse.ArgumentParser(description="Home Credit SQLite database tools")
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load application_train.csv into customers table",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=APPLICATION_TRAIN,
        help="CSV path (default: data/application_train.csv)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DB_PATH,
        help="Database path (default: data/credit_risk.db)",
    )
    parser.add_argument(
        "--if-exists",
        choices=["fail", "replace", "append"],
        default="replace",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help=(
            f"Max CSV rows to load (default: {DEFAULT_MAX_ROWS:,}). "
            "Use 0 to load the full dataset."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=INSERT_CHUNK_SIZE,
        help=f"SQLite insert batch size (default: {INSERT_CHUNK_SIZE:,})",
    )
    parser.add_argument("--query", type=str, help="Run a SELECT query")
    args = parser.parse_args()

    max_rows = None if args.max_rows == 0 else args.max_rows
    db = CreditRiskDatabase(db_path=args.db)

    if args.load or not args.query:
        db.load_application_train(
            csv_path=args.csv,
            if_exists=args.if_exists,
            max_rows=max_rows,
            chunksize=args.chunk_size,
        )

    if args.query:
        result = db.execute_query(args.query)
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
