"""
src/data_loader.py
------------------
Handles all data ingestion: CSV loading for local dev,
and PostgreSQL querying for production pipeline.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    RAW_DATA_FILE,
    PROCESSED_DATA_FILE,
    DATABASE_URL,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)


# ── CSV loaders ────────────────────────────────────────────────────────────

def load_raw_sessions(path: Path = RAW_DATA_FILE) -> pd.DataFrame:
    """Load raw session CSV."""
    logger.info(f"Loading raw sessions from {path}")
    df = pd.read_csv(path)
    logger.info(f"  Rows: {len(df):,}  |  Columns: {df.shape[1]}")
    return df


def load_processed_data(path: Path = PROCESSED_DATA_FILE) -> pd.DataFrame:
    """Load feature-engineered dataset."""
    logger.info(f"Loading processed data from {path}")
    df = pd.read_csv(path)
    logger.info(f"  Rows: {len(df):,}  |  Conversion rate: {df[TARGET_COLUMN].mean():.2%}")
    return df


# ── SQL loaders ────────────────────────────────────────────────────────────

def get_engine():
    """Create SQLAlchemy engine from config."""
    return create_engine(DATABASE_URL)


def load_from_sql(query: str) -> pd.DataFrame:
    """
    Execute an arbitrary SQL query and return a DataFrame.

    Parameters
    ----------
    query : str
        Raw SQL string to execute.

    Returns
    -------
    pd.DataFrame
    """
    engine = get_engine()
    logger.info("Executing SQL query...")
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)
    logger.info(f"  Returned {len(df):,} rows")
    return df


def load_session_features_from_db() -> pd.DataFrame:
    """Load pre-computed session features from the session_features view."""
    query = (Path(__file__).parent.parent / "sql" / "session_features.sql").read_text()
    return load_from_sql(query)


# ── Validation ─────────────────────────────────────────────────────────────

def validate_schema(df: pd.DataFrame, required_cols: list[str]) -> None:
    """Assert required columns exist; raise ValueError otherwise."""
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    logger.info("Schema validation passed.")


def summarize(df: pd.DataFrame) -> None:
    """Print a quick summary of the DataFrame."""
    print(f"\n{'='*55}")
    print(f"Shape         : {df.shape}")
    print(f"Conversion rate: {df[TARGET_COLUMN].mean():.2%}" if TARGET_COLUMN in df.columns else "")
    print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"Dtypes:\n{df.dtypes.value_counts().to_string()}")
    print("="*55)
