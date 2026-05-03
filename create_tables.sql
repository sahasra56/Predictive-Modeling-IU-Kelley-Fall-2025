-- sql/create_tables.sql
-- Schema for the e-commerce session analytics database.

-- ── Sessions (raw event log) ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sessions (
    session_id              VARCHAR(20)   PRIMARY KEY,
    user_id                 INTEGER       NOT NULL,
    session_start_ts        TIMESTAMPTZ   NOT NULL,
    session_end_ts          TIMESTAMPTZ,
    session_duration_min    NUMERIC(8,2),
    pages_viewed            SMALLINT,
    product_detail_views    SMALLINT,
    search_query_count      SMALLINT,
    cart_add_count          SMALLINT,
    time_on_cart_page_min   NUMERIC(6,2)  DEFAULT 0,
    device_type             VARCHAR(10),   -- 'mobile','desktop','tablet'
    traffic_source          VARCHAR(20),   -- 'organic','paid_search','email','direct','social'
    time_of_day             VARCHAR(12),   -- 'morning','afternoon','evening','night'
    category_affinity       VARCHAR(30),
    used_promo_code         BOOLEAN       DEFAULT FALSE,
    viewed_reviews          BOOLEAN       DEFAULT FALSE,
    added_to_wishlist       BOOLEAN       DEFAULT FALSE,
    converted               BOOLEAN       NOT NULL DEFAULT FALSE,
    created_at              TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

-- ── Users (aggregate history) ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    user_id                 INTEGER       PRIMARY KEY,
    is_returning_user       BOOLEAN       NOT NULL DEFAULT FALSE,
    purchase_recency_days   NUMERIC(6,1)  DEFAULT 999,  -- 999 = no history
    visit_frequency_30d     SMALLINT      DEFAULT 0,
    avg_order_value         NUMERIC(10,2) DEFAULT 0,
    cart_abandon_rate       NUMERIC(5,4)  DEFAULT 0,
    total_purchases         INTEGER       DEFAULT 0,
    created_at              TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

-- ── Predictions (model output log) ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id           SERIAL        PRIMARY KEY,
    session_id              VARCHAR(20)   REFERENCES sessions(session_id),
    model_version           VARCHAR(50),
    prob_purchase           NUMERIC(6,4),
    predicted_converted     BOOLEAN,
    threshold_used          NUMERIC(4,2)  DEFAULT 0.50,
    actual_converted        BOOLEAN,
    predicted_at            TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

-- ── Indexes ────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_sessions_user_id       ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_converted     ON sessions(converted);
CREATE INDEX IF NOT EXISTS idx_sessions_start_ts      ON sessions(session_start_ts DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_traffic       ON sessions(traffic_source);
CREATE INDEX IF NOT EXISTS idx_predictions_session    ON predictions(session_id);
CREATE INDEX IF NOT EXISTS idx_predictions_model_ver  ON predictions(model_version);
