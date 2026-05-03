"""
tests/test_features.py
----------------------
Unit tests for the feature engineering pipeline.
Run with: pytest tests/test_features.py -v
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.feature_engineering import engineer_features, build_preprocessor, split_features_target


# ── Fixtures ───────────────────────────────────────────────────────────────

def make_sample_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
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


# ── engineer_features ──────────────────────────────────────────────────────

class TestEngineerFeatures:

    def test_returns_dataframe(self):
        df = make_sample_df()
        result = engineer_features(df)
        assert isinstance(result, pd.DataFrame)

    def test_does_not_modify_input(self):
        df = make_sample_df()
        cols_before = set(df.columns)
        _ = engineer_features(df)
        assert set(df.columns) == cols_before, "Input DataFrame was mutated"

    def test_new_columns_present(self):
        df = engineer_features(make_sample_df())
        expected = [
            "engagement_rate", "cart_commitment_ratio", "recency_segment",
            "high_intent", "revenue_potential", "search_no_cart", "session_value_index",
        ]
        for col in expected:
            assert col in df.columns, f"Missing engineered column: {col}"

    def test_engagement_rate_non_negative(self):
        df = engineer_features(make_sample_df())
        assert (df["engagement_rate"] >= 0).all()

    def test_cart_commitment_ratio_bounded(self):
        df = engineer_features(make_sample_df())
        assert (df["cart_commitment_ratio"] >= 0).all()

    def test_high_intent_binary(self):
        df = engineer_features(make_sample_df())
        assert df["high_intent"].isin([0, 1]).all()

    def test_search_no_cart_binary(self):
        df = engineer_features(make_sample_df())
        assert df["search_no_cart"].isin([0, 1]).all()

    def test_session_value_index_range(self):
        df = engineer_features(make_sample_df())
        assert (df["session_value_index"] >= 0).all()
        assert (df["session_value_index"] <= 1.01).all()  # allow tiny float error

    def test_row_count_preserved(self):
        df = make_sample_df(100)
        df_eng = engineer_features(df)
        assert len(df_eng) == 100

    def test_zero_duration_session(self):
        """Sessions with 0 duration should yield engagement_rate == 0."""
        df = make_sample_df(10)
        df.loc[0, "session_duration_min"] = 0.0
        df_eng = engineer_features(df)
        assert df_eng.loc[0, "engagement_rate"] == 0.0

    def test_no_cart_session(self):
        """Sessions with no cart activity → revenue_potential == 0."""
        df = make_sample_df(10)
        df.loc[0, "cart_add_count"] = 0
        df_eng = engineer_features(df)
        assert df_eng.loc[0, "revenue_potential"] == 0.0

    def test_recency_segment_values(self):
        valid = {"<7d", "7-30d", "30-90d", "90-180d", "180-365d", "no_history"}
        df_eng = engineer_features(make_sample_df())
        assert set(df_eng["recency_segment"].unique()).issubset(valid)


# ── build_preprocessor ────────────────────────────────────────────────────

class TestPreprocessor:

    def test_fit_transform_no_error(self):
        df = engineer_features(make_sample_df())
        X, y = split_features_target(df)
        pre = build_preprocessor()
        X_t = pre.fit_transform(X)
        assert X_t.shape[0] == len(df)

    def test_output_no_nans(self):
        df = engineer_features(make_sample_df())
        # Introduce some NaNs
        df.loc[5, "session_duration_min"] = np.nan
        df.loc[10, "device_type"]         = np.nan
        X, y = split_features_target(df)
        pre  = build_preprocessor()
        X_t  = pre.fit_transform(X)
        assert not np.isnan(X_t).any(), "Preprocessor output contains NaNs"

    def test_unseen_categories(self):
        """OneHotEncoder should handle unknown categories gracefully."""
        df_train = engineer_features(make_sample_df(150))
        df_test  = engineer_features(make_sample_df(50))
        df_test.loc[0, "device_type"] = "smartwatch"  # unseen category

        X_train, _ = split_features_target(df_train)
        X_test,  _ = split_features_target(df_test)

        pre = build_preprocessor()
        pre.fit(X_train)
        X_t = pre.transform(X_test)
        assert not np.isnan(X_t).any()


# ── split_features_target ─────────────────────────────────────────────────

class TestSplitFeaturesTarget:

    def test_returns_tuple(self):
        df = engineer_features(make_sample_df())
        result = split_features_target(df)
        assert len(result) == 2

    def test_target_is_series(self):
        df = engineer_features(make_sample_df())
        _, y = split_features_target(df)
        assert hasattr(y, "iloc"), "y should be a pandas Series"

    def test_X_no_target_column(self):
        df = engineer_features(make_sample_df())
        X, _ = split_features_target(df)
        assert "converted" not in X.columns
