"""
src/model.py
------------
End-to-end training pipeline:
  1. Load processed data
  2. Train/val/test split
  3. Fit baseline DecisionTree
  4. Fit tuned RandomForest via RandomizedSearchCV
  5. Serialize models + metadata
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    BASELINE_MODEL_PATH,
    FINAL_MODEL_PATH,
    METADATA_PATH,
    CV_FOLDS,
    N_ITER,
    RANDOM_STATE,
    RF_PARAM_GRID,
    SCORING,
    TEST_SIZE,
    VAL_SIZE,
)
from data_loader import load_processed_data
from feature_engineering import build_preprocessor, engineer_features, split_features_target
from evaluate import compute_metrics, log_metrics

logger = logging.getLogger(__name__)


# ── Data Split ────────────────────────────────────────────────────────────

def make_splits(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
):
    """
    Stratified three-way split: train / validation / test.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First cut off test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    # Then split train into train + val
    val_fraction_of_trainval = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_fraction_of_trainval,
        random_state=RANDOM_STATE,
        stratify=y_trainval,
    )
    logger.info(
        f"Split sizes  train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Baseline Model ────────────────────────────────────────────────────────

def train_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor,
) -> Pipeline:
    """
    Fit a simple DecisionTree as the baseline.
    Returns a full sklearn Pipeline (preprocessor + classifier).
    """
    logger.info("Training baseline DecisionTree...")
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=50,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])
    pipe.fit(X_train, y_train)
    logger.info("Baseline training complete.")
    return pipe


# ── Tuned RandomForest ────────────────────────────────────────────────────

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor,
) -> Pipeline:
    """
    RandomizedSearchCV over RandomForestClassifier.
    Returns the best pipeline found.
    """
    logger.info(f"Running RandomizedSearchCV ({N_ITER} iterations, {CV_FOLDS}-fold CV)...")
    t0 = time.time()

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        pipe,
        param_distributions=RF_PARAM_GRID,
        n_iter=N_ITER,
        scoring=SCORING,
        cv=cv,
        refit=True,
        verbose=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    elapsed = time.time() - t0
    logger.info(
        f"Best CV {SCORING}: {search.best_score_:.4f}  "
        f"(elapsed {elapsed:.0f}s)\n"
        f"Best params: {search.best_params_}"
    )
    return search.best_estimator_


# ── Gradient Boosting (Comparison) ───────────────────────────────────────

def train_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor,
) -> Pipeline:
    """Fit a GradientBoostingClassifier for comparison."""
    logger.info("Training GradientBoostingClassifier...")
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=RANDOM_STATE,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


# ── Probability Calibration ───────────────────────────────────────────────

def calibrate(model: Pipeline, X_val: pd.DataFrame, y_val: pd.Series) -> CalibratedClassifierCV:
    """
    Wrap a fitted pipeline in Platt scaling (sigmoid) calibration.
    Improves reliability of predicted probabilities.
    """
    calibrated = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    calibrated.fit(X_val, y_val)
    return calibrated


# ── Serialization ─────────────────────────────────────────────────────────

def save_model(model, path: Path) -> None:
    joblib.dump(model, path)
    logger.info(f"Model saved → {path}")


def save_metadata(records: dict, path: Path = METADATA_PATH) -> None:
    with open(path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    logger.info(f"Metadata saved → {path}")


# ── Main Pipeline ─────────────────────────────────────────────────────────

def run_pipeline() -> dict:
    """
    Execute the full training pipeline and return a metadata dict.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )

    # 1. Load data
    from data_loader import load_raw_sessions
    df_raw = load_raw_sessions()
    df = engineer_features(df_raw)
    X, y = split_features_target(df)

    # 2. Split
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X, y)

    # 3. Build preprocessor (fit only on train)
    preprocessor = build_preprocessor()

    # ── Baseline ──
    baseline = train_baseline(X_train, y_train, preprocessor)
    save_model(baseline, BASELINE_MODEL_PATH)
    baseline_metrics = compute_metrics(baseline, X_test, y_test, label="Baseline DT")
    log_metrics(baseline_metrics)

    # ── Tuned RF ──
    rf_tuned = train_random_forest(X_train, y_train, preprocessor)

    # ── Calibrate ──
    rf_calibrated = calibrate(rf_tuned, X_val, y_val)
    save_model(rf_calibrated, FINAL_MODEL_PATH)
    rf_metrics = compute_metrics(rf_calibrated, X_test, y_test, label="RF Tuned + Calibrated")
    log_metrics(rf_metrics)

    # ── GB comparison ──
    gb = train_gradient_boosting(X_train, y_train, preprocessor)
    gb_metrics = compute_metrics(gb, X_test, y_test, label="GradientBoosting")
    log_metrics(gb_metrics)

    # 4. Save metadata
    metadata = {
        "run_timestamp": datetime.now().isoformat(),
        "n_train": len(X_train),
        "n_val":   len(X_val),
        "n_test":  len(X_test),
        "baseline": baseline_metrics,
        "rf_tuned": rf_metrics,
        "gradient_boosting": gb_metrics,
    }
    save_metadata(metadata)

    print("\n" + "="*55)
    print("PIPELINE COMPLETE")
    print(f"  Baseline AUC : {baseline_metrics['roc_auc']:.4f}")
    print(f"  RF Tuned AUC : {rf_metrics['roc_auc']:.4f}")
    print(f"  GB AUC       : {gb_metrics['roc_auc']:.4f}")
    print("="*55)

    return metadata


if __name__ == "__main__":
    run_pipeline()
