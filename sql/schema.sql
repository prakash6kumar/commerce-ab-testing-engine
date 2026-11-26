DROP TABLE IF EXISTS orders CASCADE;
DROP TABLE IF EXISTS sessions CASCADE;
DROP TABLE IF EXISTS exposures CASCADE;
DROP TABLE IF EXISTS users CASCADE;
DROP TABLE IF EXISTS merchants CASCADE;

CREATE TABLE merchants (
    merchant_id INT PRIMARY KEY,
    merchant_tier VARCHAR(20) NOT NULL CHECK (merchant_tier IN ('small', 'medium', 'enterprise')),
    category VARCHAR(50) NOT NULL,
    shipping_complexity INT NOT NULL CHECK (shipping_complexity BETWEEN 1 AND 5)
);

CREATE TABLE users (
    user_id INT PRIMARY KEY,
    merchant_id INT NOT NULL REFERENCES merchants(merchant_id),
    device_type VARCHAR(20) NOT NULL CHECK (device_type IN ('mobile', 'desktop')),
    browser VARCHAR(20) NOT NULL CHECK (browser IN ('safari', 'chrome', 'other')),
    signup_date DATE NOT NULL
);

CREATE TABLE exposures (
    exposure_id BIGSERIAL PRIMARY KEY,
    user_id INT NOT NULL REFERENCES users(user_id),
    merchant_id INT NOT NULL REFERENCES merchants(merchant_id),
    variant VARCHAR(20) NOT NULL CHECK (variant IN ('control', 'treatment')),
    exposed_at TIMESTAMP NOT NULL,
    device_type VARCHAR(20) NOT NULL,
    browser VARCHAR(20) NOT NULL
);

CREATE TABLE sessions (
    session_id BIGINT PRIMARY KEY,
    user_id INT NOT NULL REFERENCES users(user_id),
    merchant_id INT NOT NULL REFERENCES merchants(merchant_id),
    session_start TIMESTAMP NOT NULL,
    period VARCHAR(20) NOT NULL CHECK (period IN ('pre', 'experiment')),
    reached_checkout BOOLEAN NOT NULL,
    checkout_latency_ms NUMERIC(10, 2),
    converted BOOLEAN NOT NULL
);

CREATE TABLE orders (
    order_id BIGINT PRIMARY KEY,
    session_id BIGINT NOT NULL REFERENCES sessions(session_id),
    user_id INT NOT NULL REFERENCES users(user_id),
    merchant_id INT NOT NULL REFERENCES merchants(merchant_id),
    order_timestamp TIMESTAMP NOT NULL,
    period VARCHAR(20) NOT NULL CHECK (period IN ('pre', 'experiment')),
    gmv_usd NUMERIC(12, 2) NOT NULL,
    is_refunded BOOLEAN NOT NULL
);

-- Indexes to speed up analytical joins
CREATE INDEX idx_exposures_user ON exposures(user_id);
CREATE INDEX idx_sessions_user_period ON sessions(user_id, period);
CREATE INDEX idx_orders_user_period ON orders(user_id, period);