"""
data/generate_data.py
---------------------
Generates a realistic synthetic e-commerce session dataset that mirrors
the schema of the anonymized dataset used in this project.

Run:
    python data/generate_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_STATE = 42
N_SAMPLES    = 50_000
OUTPUT_PATH  = Path(__file__).parent / "raw" / "sessions.csv"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(RANDOM_STATE)


def generate_sessions(n: int = N_SAMPLES) -> pd.DataFrame:
    """Simulate browsing sessions with realistic correlations."""

    # ── User & Session Identity ──────────────────────────────────────────
    user_ids    = rng.integers(1000, 999_999, size=n)
    session_ids = [f"sess_{i:07d}" for i in range(n)]

    # ── Behavioral covariates ────────────────────────────────────────────
    is_returning = rng.choice([0, 1], size=n, p=[0.45, 0.55])
    device_type  = rng.choice(
        ["mobile", "desktop", "tablet"], size=n, p=[0.52, 0.38, 0.10]
    )
    traffic_source = rng.choice(
        ["organic", "paid_search", "email", "direct", "social"],
        size=n, p=[0.30, 0.25, 0.20, 0.15, 0.10],
    )
    time_of_day = rng.choice(
        ["morning", "afternoon", "evening", "night"],
        size=n, p=[0.20, 0.30, 0.35, 0.15],
    )
    category_affinity = rng.choice(
        ["electronics", "clothing", "home_garden", "beauty", "sports", "books"],
        size=n,
    )

    # ── Session metrics (correlated with conversion) ─────────────────────
    session_duration_min = np.clip(
        rng.gamma(shape=2, scale=4, size=n) + is_returning * 2, 0.5, 90
    )
    pages_viewed = np.clip(
        rng.poisson(lam=4 + is_returning * 2, size=n), 1, 40
    ).astype(int)
    cart_add_count = np.clip(
        rng.poisson(lam=pages_viewed / 5, size=n), 0, 10
    ).astype(int)
    product_detail_views = np.clip(
        rng.poisson(lam=pages_viewed * 0.6, size=n), 0, 30
    ).astype(int)
    search_query_count = np.clip(
        rng.poisson(lam=1.5, size=n), 0, 15
    ).astype(int)
    time_on_cart_page_min = np.where(
        cart_add_count > 0,
        np.clip(rng.exponential(scale=3, size=n), 0, 20),
        0.0,
    )

    # ── Historical user-level metrics ────────────────────────────────────
    purchase_recency_days = np.where(
        is_returning,
        np.clip(rng.exponential(scale=30, size=n), 1, 365),
        np.full(n, 999),          # 999 = no prior purchase
    )
    visit_frequency_30d = np.clip(
        rng.poisson(lam=3 + is_returning * 3, size=n), 0, 30
    ).astype(int)
    avg_order_value = np.where(
        is_returning,
        np.clip(rng.lognormal(mean=4.0, sigma=0.6, size=n), 5, 500),
        0.0,
    )
    cart_abandon_rate = np.clip(
        rng.beta(a=2, b=3, size=n) + (1 - is_returning) * 0.2, 0, 1
    )

    # ── Interaction flags ─────────────────────────────────────────────────
    used_promo_code   = (rng.random(n) < 0.18).astype(int)
    viewed_reviews    = (rng.random(n) < 0.40 + 0.15 * (cart_add_count > 0)).astype(int)
    added_to_wishlist = (rng.random(n) < 0.12).astype(int)

    # ── Conversion label (logistic-like probability) ──────────────────────
    log_odds = (
        -3.0
        + 0.8  * is_returning
        + 0.05 * session_duration_min
        + 0.12 * pages_viewed
        + 0.40 * cart_add_count
        - 0.30 * cart_abandon_rate
        - 0.002 * purchase_recency_days
        + 0.10 * visit_frequency_30d
        + 0.003 * avg_order_value
        + 0.25 * used_promo_code
        + 0.20 * viewed_reviews
        + 0.15 * added_to_wishlist
        + 0.30 * (traffic_source == "email")
        + 0.15 * (device_type == "desktop")
        + 0.10 * (time_of_day == "evening")
        + rng.normal(0, 0.5, size=n)   # noise
    )
    prob_convert = 1 / (1 + np.exp(-log_odds))
    converted    = (rng.random(n) < prob_convert).astype(int)

    df = pd.DataFrame({
        "session_id":             session_ids,
        "user_id":                user_ids,
        "session_duration_min":   session_duration_min.round(2),
        "pages_viewed":           pages_viewed,
        "cart_add_count":         cart_add_count,
        "product_detail_views":   product_detail_views,
        "search_query_count":     search_query_count,
        "time_on_cart_page_min":  time_on_cart_page_min.round(2),
        "cart_abandon_rate":      cart_abandon_rate.round(4),
        "purchase_recency_days":  purchase_recency_days.round(1),
        "visit_frequency_30d":    visit_frequency_30d,
        "avg_order_value":        avg_order_value.round(2),
        "device_type":            device_type,
        "traffic_source":         traffic_source,
        "time_of_day":            time_of_day,
        "category_affinity":      category_affinity,
        "is_returning_user":      is_returning,
        "used_promo_code":        used_promo_code,
        "viewed_reviews":         viewed_reviews,
        "added_to_wishlist":      added_to_wishlist,
        "converted":              converted,
    })

    print(f"Generated {n:,} sessions  |  Conversion rate: {converted.mean():.2%}")
    return df


if __name__ == "__main__":
    df = generate_sessions()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved → {OUTPUT_PATH}")
