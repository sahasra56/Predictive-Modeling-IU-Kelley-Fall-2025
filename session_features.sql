-- sql/session_features.sql
-- Joins session-level events with user-level history
-- to produce the model feature vector for each session.
-- This query powers both batch scoring and nightly retraining.

WITH session_base AS (
    SELECT
        s.session_id,
        s.user_id,
        s.session_duration_min,
        s.pages_viewed,
        s.product_detail_views,
        s.search_query_count,
        s.cart_add_count,
        s.time_on_cart_page_min,
        s.device_type,
        s.traffic_source,
        s.time_of_day,
        s.category_affinity,
        s.used_promo_code::INT       AS used_promo_code,
        s.viewed_reviews::INT        AS viewed_reviews,
        s.added_to_wishlist::INT     AS added_to_wishlist,
        s.converted::INT             AS converted
    FROM sessions s
    WHERE s.session_start_ts >= NOW() - INTERVAL '90 days'
),

user_history AS (
    SELECT
        u.user_id,
        u.is_returning_user::INT     AS is_returning_user,
        u.purchase_recency_days,
        u.visit_frequency_30d,
        u.avg_order_value,
        u.cart_abandon_rate
    FROM users u
)

SELECT
    sb.session_id,
    sb.user_id,

    -- ── Session features ──────────────────────────────────────────────────
    sb.session_duration_min,
    sb.pages_viewed,
    sb.product_detail_views,
    sb.search_query_count,
    sb.cart_add_count,
    sb.time_on_cart_page_min,
    sb.device_type,
    sb.traffic_source,
    sb.time_of_day,
    sb.category_affinity,
    sb.used_promo_code,
    sb.viewed_reviews,
    sb.added_to_wishlist,

    -- ── User history features ─────────────────────────────────────────────
    COALESCE(uh.is_returning_user,     0)   AS is_returning_user,
    COALESCE(uh.purchase_recency_days, 999) AS purchase_recency_days,
    COALESCE(uh.visit_frequency_30d,   0)   AS visit_frequency_30d,
    COALESCE(uh.avg_order_value,       0)   AS avg_order_value,
    COALESCE(uh.cart_abandon_rate,     0)   AS cart_abandon_rate,

    -- ── Derived features (computed in SQL for efficiency) ─────────────────
    CASE
        WHEN sb.session_duration_min > 0
        THEN sb.pages_viewed / sb.session_duration_min
        ELSE 0
    END AS engagement_rate,

    CASE
        WHEN sb.pages_viewed > 0
        THEN sb.cart_add_count::FLOAT / sb.pages_viewed
        ELSE 0
    END AS cart_commitment_ratio,

    CASE
        WHEN sb.cart_add_count > 0 AND sb.product_detail_views >= 2
        THEN 1 ELSE 0
    END AS high_intent,

    CASE
        WHEN sb.search_query_count > 0 AND sb.cart_add_count = 0
        THEN 1 ELSE 0
    END AS search_no_cart,

    -- ── Label ─────────────────────────────────────────────────────────────
    sb.converted

FROM session_base sb
LEFT JOIN user_history uh ON sb.user_id = uh.user_id

ORDER BY sb.session_id;
