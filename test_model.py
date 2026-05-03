"""
tests/test_model.py
-------------------
Integration tests for the training and prediction pipeline.
Run with: pytest tests/test_model.py -v
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.feature_engineering import engineer_features, build_preprocessor, split_features_target
from src.model import train_baseline, make_splits
from src.evaluate import compute_metrics


def make_sample_df(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "session_id":             [f"sess_{i}" for i in range(n)],
        "user_id":                rng.integers(1, 1000, n),
        "session_duration_min":   rng.uniform(0.5, 30, n),
        "pages_viewed":           rng.integers(1, 20, n),
        "cart_add_count":         rng.integers(0, 5, n),
        "product_detail_views":   rng.integers(0, 10, n),
        "search_query_count":     rng.integers(0, 5, n),
        "time_on_cart_page_min":  rng.uniform(0, 10, n),
        "cart_abandon_rate":      rng.uniform(0, 1, n),
        "purchase_recency_days":  rng.choice([999, 7, 30, 90], size=n),
        "visit_frequency_30d":    rng.integers(0, 10, n),
        "avg_order_value":        rng.uniform(0, 200, n),
        "device_type":            rng.choice(["mobile", "desktop", "tablet"], n),
        "traffic_source":         rng.choice(["organic", "paid_search", "email"], n),
        "time_of_day":            rng.choice(["morning", "afternoon", "evening", "night"], n),
        "category_affinity":      rng.choice(["electronics", "clothing", "home_garden"], n),
        "is_returning_user":      rng.integers(0, 2, n),
        "used_promo_code":        rng.integers(0, 2, n),
        "viewed_reviews":         rng.integers(0, 2, n),
        "added_to_wishlist":      rng.integers(0, 2, n),
        "converted":              rng.integers(0, 2, n),
    })


@pytest.fixture(scope="module")
def prepared_data():
    df = engineer_features(make_sample_df())
    X, y = split_features_target(df)
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X, y)
    pre = build_preprocessor()
    return X_train, X_val, X_test, y_train, y_val, y_test, pre


class TestTrainBaseline:

    def test_baseline_trains(self, prepared_data):
        X_train, _, _, y_train, _, _, pre = prepared_data
        model = train_baseline(X_train, y_train, pre)
        assert model is not None

    def test_baseline_predicts(self, prepared_data):
        X_train, _, X_test, y_train, _, _, pre = prepared_data
        model = train_baseline(X_train, y_train, pre)
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)

    def test_baseline_probabilities_valid(self, prepared_data):
        X_train, _, X_test, y_train, _, _, pre = prepared_data
        model = train_baseline(X_train, y_train, pre)
        probs = model.predict_proba(X_test)
        assert probs.shape[1] == 2
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_baseline_predictions_binary(self, prepared_data):
        X_train, _, X_test, y_train, _, _, pre = prepared_data
        model = train_baseline(X_train, y_train, pre)
        preds = model.predict(X_test)
        assert set(preds).issubset({0, 1})


class TestComputeMetrics:

    def test_metrics_keys(self, prepared_data):
        X_train, _, X_test, y_train, _, y_test, pre = prepared_data
        model = train_baseline(X_train, y_train, pre)
        metrics = compute_metrics(model, X_test, y_test)
        required = {"roc_auc", "precision", "recall", "f1", "avg_precision", "brier_score"}
        assert required.issubset(metrics.keys())

    def test_roc_auc_in_range(self, prepared_data):
        X_train, _, X_test, y_train, _, y_test, pre = prepared_data
        model = train_baseline(X_train, y_train, pre)
        metrics = compute_metrics(model, X_test, y_test)
        assert 0.0 <= metrics["roc_auc"] <= 1.0

    def test_brier_score_non_negative(self, prepared_data):
        X_train, _, X_test, y_train, _, y_test, pre = prepared_data
        model = train_baseline(X_train, y_train, pre)
        metrics = compute_metrics(model, X_test, y_test)
        assert metrics["brier_score"] >= 0


class TestMakeSplits:

    def test_split_sizes(self):
        df = engineer_features(make_sample_df(1000))
        X, y = split_features_target(df)
        X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X, y)
        total = len(X_train) + len(X_val) + len(X_test)
        assert total == len(X)

    def test_no_leakage(self):
        df = engineer_features(make_sample_df(500))
        X, y = split_features_target(df)
        X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X, y)
        train_idx = set(X_train.index)
        val_idx   = set(X_val.index)
        test_idx  = set(X_test.index)
        assert len(train_idx & val_idx)  == 0, "Train/val overlap"
        assert len(train_idx & test_idx) == 0, "Train/test overlap"
        assert len(val_idx  & test_idx)  == 0, "Val/test overlap"

    def test_class_balance_preserved(self):
        """Stratified split should keep similar class ratios."""
        df = engineer_features(make_sample_df(1000))
        X, y = split_features_target(df)
        X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X, y)
        base_rate = y.mean()
        for split_y, name in [(y_train, "train"), (y_val, "val"), (y_test, "test")]:
            rate = split_y.mean()
            assert abs(rate - base_rate) < 0.05, \
                f"Class imbalance in {name}: {rate:.3f} vs base {base_rate:.3f}"
