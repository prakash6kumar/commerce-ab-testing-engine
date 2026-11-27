DROP TABLE IF EXISTS exp_session_metrics;
DROP TABLE IF EXISTS exp_user_metrics;
DROP TABLE IF EXISTS exp_clean_exposures;

-- 1. Clean exposures: remove cross-variant contaminated users and deduplicate to first exposure
CREATE TABLE exp_clean_exposures AS
WITH user_variant_counts AS (
    SELECT
        user_id,
        COUNT(DISTINCT variant) AS distinct_variants
    FROM exposures
    GROUP BY user_id
),
ranked_exposures AS (
    SELECT
        e.user_id,
        e.merchant_id,
        e.variant,
        e.exposed_at,
        e.device_type,
        e.browser,
        ROW_NUMBER() OVER (
            PARTITION BY e.user_id
            ORDER BY e.exposed_at ASC, e.exposure_id ASC
        ) AS rn
    FROM exposures e
    INNER JOIN user_variant_counts uvc
        ON e.user_id = uvc.user_id
    WHERE uvc.distinct_variants = 1
)
SELECT
    user_id,
    merchant_id,
    variant,
    exposed_at,
    device_type,
    browser,
    (device_type = 'mobile' AND browser = 'safari') AS is_mobile_safari
FROM ranked_exposures
WHERE rn = 1;

CREATE INDEX idx_clean_exp_user ON exp_clean_exposures(user_id);

-- 2. User-level analysis table (1 row per clean exposed user)
CREATE TABLE exp_user_metrics AS
WITH pre_sessions AS (
    SELECT
        user_id,
        COUNT(*) AS pre_sessions,
        COUNT(*) FILTER (WHERE reached_checkout) AS pre_checkout_sessions,
        COUNT(*) FILTER (WHERE converted) AS pre_converted_sessions
    FROM sessions
    WHERE period = 'pre'
    GROUP BY user_id
),
pre_orders AS (
    SELECT
        user_id,
        COUNT(*) AS pre_orders,
        SUM(gmv_usd) AS pre_gmv_usd
    FROM orders
    WHERE period = 'pre'
    GROUP BY user_id
),
exp_sessions AS (
    SELECT
        s.user_id,
        COUNT(*) AS exp_sessions,
        COUNT(*) FILTER (WHERE s.reached_checkout) AS exp_checkout_sessions,
        COUNT(*) FILTER (WHERE s.converted) AS exp_converted_sessions,
        AVG(s.checkout_latency_ms) FILTER (WHERE s.reached_checkout) AS avg_checkout_latency_ms
    FROM sessions s
    INNER JOIN exp_clean_exposures e
        ON s.user_id = e.user_id
    WHERE s.period = 'experiment'
      AND s.session_start >= e.exposed_at
    GROUP BY s.user_id
),
exp_orders AS (
    SELECT
        o.user_id,
        COUNT(*) AS exp_orders,
        SUM(o.gmv_usd) AS exp_gmv_usd,
        COUNT(*) FILTER (WHERE o.is_refunded) AS exp_refunded_orders
    FROM orders o
    INNER JOIN exp_clean_exposures e
        ON o.user_id = e.user_id
    WHERE o.period = 'experiment'
      AND o.order_timestamp >= e.exposed_at
    GROUP BY o.user_id
)
SELECT
    e.user_id,
    e.merchant_id,
    m.merchant_tier,
    m.category AS merchant_category,
    m.shipping_complexity,
    e.variant,
    e.exposed_at,
    e.device_type,
    e.browser,
    e.is_mobile_safari,

    -- Pre-experiment covariates (for balance checks & CUPED)
    COALESCE(ps.pre_sessions, 0) AS pre_sessions,
    COALESCE(ps.pre_checkout_sessions, 0) AS pre_checkout_sessions,
    COALESCE(ps.pre_converted_sessions, 0) AS pre_converted_sessions,
    COALESCE(po.pre_orders, 0) AS pre_orders,
    COALESCE(po.pre_gmv_usd, 0.0)::NUMERIC(12, 2) AS pre_gmv_usd,
    CASE WHEN COALESCE(po.pre_orders, 0) > 0 THEN 1 ELSE 0 END AS pre_converted,

    -- Post-exposure experiment metrics
    COALESCE(es.exp_sessions, 0) AS exp_sessions,
    COALESCE(es.exp_checkout_sessions, 0) AS exp_checkout_sessions,
    COALESCE(es.exp_converted_sessions, 0) AS exp_converted_sessions,
    ROUND(es.avg_checkout_latency_ms, 2) AS avg_checkout_latency_ms,
    COALESCE(eo.exp_orders, 0) AS exp_orders,
    COALESCE(eo.exp_gmv_usd, 0.0)::NUMERIC(12, 2) AS exp_gmv_usd,
    COALESCE(eo.exp_refunded_orders, 0) AS exp_refunded_orders,
    CASE WHEN COALESCE(eo.exp_orders, 0) > 0 THEN 1 ELSE 0 END AS exp_converted
FROM exp_clean_exposures e
INNER JOIN merchants m
    ON e.merchant_id = m.merchant_id
LEFT JOIN pre_sessions ps
    ON e.user_id = ps.user_id
LEFT JOIN pre_orders po
    ON e.user_id = po.user_id
LEFT JOIN exp_sessions es
    ON e.user_id = es.user_id
LEFT JOIN exp_orders eo
    ON e.user_id = eo.user_id;

CREATE INDEX idx_exp_user_variant ON exp_user_metrics(variant);
CREATE INDEX idx_exp_user_tier ON exp_user_metrics(merchant_tier);

-- 3. Session-level analysis table (post-exposure checkout sessions for unit-mismatch & weekly novelty)
CREATE TABLE exp_session_metrics AS
SELECT
    s.session_id,
    s.user_id,
    s.merchant_id,
    m.merchant_tier,
    e.variant,
    e.device_type,
    e.browser,
    e.is_mobile_safari,
    s.session_start,
    LEAST(
        4,
        GREATEST(
            1,
            FLOOR(
                EXTRACT(EPOCH FROM (s.session_start - TIMESTAMP '2026-09-29 00:00:00')) / 604800
            )::INT + 1
        )
    ) AS exp_week,
    s.checkout_latency_ms,
    s.converted::INT AS converted
FROM sessions s
INNER JOIN exp_clean_exposures e
    ON s.user_id = e.user_id
INNER JOIN merchants m
    ON s.merchant_id = m.merchant_id
WHERE s.period = 'experiment'
  AND s.reached_checkout = TRUE
  AND s.session_start >= e.exposed_at;

CREATE INDEX idx_exp_sess_variant_week ON exp_session_metrics(variant, exp_week);