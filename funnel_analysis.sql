-- sql/funnel_analysis.sql
-- Conversion funnel metrics by traffic source, device, and time of day.
-- Powers the Tableau funnel dashboard.

-- ── 1. Overall Funnel ────────────────────────────────────────────────────
SELECT
    COUNT(*)                                            AS total_sessions,
    SUM(CASE WHEN pages_viewed  >= 2  THEN 1 END)      AS reached_browse,
    SUM(CASE WHEN product_detail_views >= 1 THEN 1 END) AS reached_product_page,
    SUM(CASE WHEN cart_add_count >= 1 THEN 1 END)       AS reached_cart,
    SUM(CASE WHEN converted = TRUE THEN 1 END)          AS purchased,

    ROUND(100.0 * SUM(CASE WHEN pages_viewed >= 2 THEN 1 END)
          / NULLIF(COUNT(*), 0), 2)                     AS pct_browse,
    ROUND(100.0 * SUM(CASE WHEN cart_add_count >= 1 THEN 1 END)
          / NULLIF(COUNT(*), 0), 2)                     AS pct_reached_cart,
    ROUND(100.0 * SUM(CASE WHEN converted = TRUE THEN 1 END)
          / NULLIF(COUNT(*), 0), 2)                     AS overall_cvr
FROM sessions
WHERE session_start_ts >= NOW() - INTERVAL '30 days';


-- ── 2. Conversion by Traffic Source ──────────────────────────────────────
SELECT
    traffic_source,
    COUNT(*)                                                          AS sessions,
    ROUND(100.0 * SUM(converted::INT) / NULLIF(COUNT(*), 0), 2)      AS cvr_pct,
    ROUND(AVG(session_duration_min), 2)                               AS avg_duration_min,
    ROUND(AVG(cart_add_count),       2)                               AS avg_cart_adds
FROM sessions
WHERE session_start_ts >= NOW() - INTERVAL '30 days'
GROUP BY traffic_source
ORDER BY cvr_pct DESC;


-- ── 3. Conversion by Device Type ─────────────────────────────────────────
SELECT
    device_type,
    COUNT(*)                                                          AS sessions,
    ROUND(100.0 * SUM(converted::INT) / NULLIF(COUNT(*), 0), 2)      AS cvr_pct,
    ROUND(AVG(pages_viewed),           2)                             AS avg_pages,
    ROUND(AVG(session_duration_min),   2)                             AS avg_duration_min
FROM sessions
WHERE session_start_ts >= NOW() - INTERVAL '30 days'
GROUP BY device_type
ORDER BY sessions DESC;


-- ── 4. Conversion by Time of Day ─────────────────────────────────────────
SELECT
    time_of_day,
    COUNT(*)                                                          AS sessions,
    ROUND(100.0 * SUM(converted::INT) / NULLIF(COUNT(*), 0), 2)      AS cvr_pct
FROM sessions
WHERE session_start_ts >= NOW() - INTERVAL '30 days'
GROUP BY time_of_day
ORDER BY
    CASE time_of_day
        WHEN 'morning'   THEN 1
        WHEN 'afternoon' THEN 2
        WHEN 'evening'   THEN 3
        WHEN 'night'     THEN 4
    END;


-- ── 5. Drop-off Points (Cart Abandonment) ─────────────────────────────────
SELECT
    CASE
        WHEN cart_add_count = 0                          THEN '1_no_cart'
        WHEN cart_add_count > 0 AND converted = FALSE    THEN '2_cart_abandoned'
        WHEN converted = TRUE                            THEN '3_purchased'
    END                                                        AS funnel_stage,
    COUNT(*)                                                   AS sessions,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)         AS pct_of_total
FROM sessions
WHERE session_start_ts >= NOW() - INTERVAL '30 days'
GROUP BY 1
ORDER BY 1;


-- ── 6. Model Prediction Performance Monitoring ────────────────────────────
-- Tracks model accuracy over time for the dashboard.
SELECT
    DATE_TRUNC('day', predicted_at)::DATE   AS prediction_date,
    model_version,
    COUNT(*)                                AS scored_sessions,
    ROUND(AVG(prob_purchase),         4)    AS avg_predicted_prob,
    ROUND(AVG(actual_converted::INT), 4)    AS actual_cvr,
    ROUND(AVG(
        CASE WHEN predicted_converted = actual_converted THEN 1 ELSE 0 END
    ), 4)                                   AS accuracy
FROM predictions
WHERE predicted_at >= NOW() - INTERVAL '30 days'
  AND actual_converted IS NOT NULL
GROUP BY 1, 2
ORDER BY 1 DESC;
