"""
src/feature_engineering.py
--------------------------
All feature construction and preprocessing logic.
Returns a scikit-learn Pipeline-compatible ColumnTransformer.
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    BINARY_FEATURES,
    ALL_FEATURES,
    TARGET_COLUMN,
    PROCESSED_DATA_FILE,
)

logger = logging.getLogger(__name__)


# ── Derived Feature Construction ───────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct derived behavioral features on top of raw session data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw sessions DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional engineered columns.
    """
    df = df.copy()

    # Engagement intensity: pages per minute
    df["engagement_rate"] = np.where(
        df["session_duration_min"] > 0,
        df["pages_viewed"] / df["session_duration_min"],
        0.0,
    )

    # Cart commitment ratio: adds relative to pages viewed
    df["cart_commitment_ratio"] = np.where(
        df["pages_viewed"] > 0,
        df["cart_add_count"] / df["pages_viewed"],
        0.0,
    )

    # Purchase recency: bin into segments for tree-based models
    df["recency_segment"] = pd.cut(
        df["purchase_recency_days"],
        bins=[0, 7, 30, 90, 180, 365, 1000],
        labels=["<7d", "7-30d", "30-90d", "90-180d", "180-365d", "no_history"],
        right=True,
    ).astype(str)

    # High-intent flag: cart add + reviewed product details
    df["high_intent"] = (
        (df["cart_add_count"] > 0) & (df["product_detail_views"] >= 2)
    ).astype(int)

    # Revenue potential proxy
    df["revenue_potential"] = (
        df["cart_add_count"] * df["avg_order_value"].clip(lower=0)
    )

    # Friction signal: searched but didn't add to cart
    df["search_no_cart"] = (
        (df["search_query_count"] > 0) & (df["cart_add_count"] == 0)
    ).astype(int)

    # Session value index (composite)
    df["session_value_index"] = (
        0.3 * df["engagement_rate"].clip(0, 10) / 10
        + 0.3 * df["cart_commitment_ratio"].clip(0, 1)
        + 0.2 * df["viewed_reviews"]
        + 0.2 * df["is_returning_user"]
    )

    logger.info(f"Engineered 7 new features. Total columns: {df.shape[1]}")
    return df


def get_feature_columns() -> dict:
    """Return updated feature groups after engineering."""
    extra_numeric = [
        "engagement_rate",
        "cart_commitment_ratio",
        "revenue_potential",
        "session_value_index",
    ]
    extra_categorical = ["recency_segment"]
    extra_binary = ["high_intent", "search_no_cart"]

    return {
        "numeric":     NUMERIC_FEATURES + extra_numeric,
        "categorical": CATEGORICAL_FEATURES + extra_categorical,
        "binary":      BINARY_FEATURES + extra_binary,
    }


# ── Preprocessing Pipeline ─────────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    """
    Construct a ColumnTransformer that handles:
      - Numeric features: median imputation + standard scaling
      - Categorical features: most-frequent imputation + one-hot encoding
      - Binary features: pass-through (already 0/1)
    """
    feature_cols = get_feature_columns()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    binary_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num",  numeric_pipeline,     feature_cols["numeric"]),
            ("cat",  categorical_pipeline, feature_cols["categorical"]),
            ("bin",  binary_pipeline,      feature_cols["binary"]),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return preprocessor


# ── X / y Split ────────────────────────────────────────────────────────────

def split_features_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) from the processed dataframe."""
    feature_cols = get_feature_columns()
    all_cols = (
        feature_cols["numeric"]
        + feature_cols["categorical"]
        + feature_cols["binary"]
    )
    X = df[[c for c in all_cols if c in df.columns]]
    y = df[TARGET_COLUMN]
    return X, y


# ── Save Processed Data ────────────────────────────────────────────────────

def save_processed(df: pd.DataFrame, path: Path = PROCESSED_DATA_FILE) -> None:
    df.to_csv(path, index=False)
    logger.info(f"Processed data saved → {path}")


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from data_loader import load_raw_sessions
    df_raw = load_raw_sessions()
    df_eng = engineer_features(df_raw)
    save_processed(df_eng)
    print(df_eng.describe())
