"""
config.py
---------
Central configuration for the Online Purchase Prediction project.
Load environment variables via .env for database credentials.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for d in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Database ───────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     os.getenv("DB_PORT", "5432"),
    "dbname":   os.getenv("DB_NAME", "ecommerce"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}

DATABASE_URL = (
    f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
)

# ── Data ───────────────────────────────────────────────────────────────────
RAW_DATA_FILE       = DATA_RAW / "sessions.csv"
PROCESSED_DATA_FILE = DATA_PROCESSED / "sessions_features.csv"
TARGET_COLUMN       = "converted"
RANDOM_STATE        = 42
TEST_SIZE           = 0.2
VAL_SIZE            = 0.1

# ── Feature Groups ─────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "session_duration_min",
    "pages_viewed",
    "cart_add_count",
    "cart_abandon_rate",
    "purchase_recency_days",
    "visit_frequency_30d",
    "avg_order_value",
    "product_detail_views",
    "search_query_count",
    "time_on_cart_page_min",
]

CATEGORICAL_FEATURES = [
    "device_type",
    "traffic_source",
    "time_of_day",
    "category_affinity",
]

BINARY_FEATURES = [
    "is_returning_user",
    "used_promo_code",
    "viewed_reviews",
    "added_to_wishlist",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

# ── Model ──────────────────────────────────────────────────────────────────
BASELINE_MODEL_PATH = MODELS_DIR / "baseline_dt.pkl"
FINAL_MODEL_PATH    = MODELS_DIR / "random_forest_tuned.pkl"
METADATA_PATH       = MODELS_DIR / "model_metadata.json"

# Hyperparameter grid for RandomForestClassifier
RF_PARAM_GRID = {
    "classifier__n_estimators":      [100, 200, 300],
    "classifier__max_depth":         [None, 10, 20, 30],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf":  [1, 2, 4],
    "classifier__max_features":      ["sqrt", "log2"],
    "classifier__class_weight":      ["balanced", None],
}

CV_FOLDS    = 5
SCORING     = "roc_auc"
N_ITER      = 30          # RandomizedSearchCV iterations
