"""Application configuration loaded from environment variables."""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # optional; use system env vars or: pip install python-dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SQL_DIR = PROJECT_ROOT / "sql"

APP_ENV = os.getenv("APP_ENV", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

DB_PATH = DATA_DIR / "credit_risk.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # openai | gemini

MODEL_PATH = Path(os.getenv("MODEL_PATH", MODELS_DIR / "model.pkl"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

# Home Credit Default Risk filenames
APPLICATION_TRAIN = DATA_DIR / "application_train.csv"
APPLICATION_TEST = DATA_DIR / "application_test.csv"
TARGET_COLUMN = "TARGET"
