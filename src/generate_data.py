import os
from pathlib import Path
import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

SEED = 42
N_MERCHANTS = 200
N_USERS = 50_000

PRE_START = pd.Timestamp("2026-09-01 00:00:00")
EXP_START = pd.Timestamp("2026-09-29 00:00:00")
PERIOD_DAYS = 28

RAW_DIR = Path("data/raw")
SCHEMA_PATH = Path("sql/schema.sql")

DB_CONFIG = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", 5432)),
    "dbname": os.getenv("PGDATABASE", "experimentation"),
    "user": os.getenv("PGUSER", "exp_user"),
    "password": os.getenv("PGPASSWORD", "exp_password"),
}


def generate_merchants(rng: np.random.Generator) -> pd.DataFrame:
    tiers = rng.choice(
        ["small", "medium", "enterprise"],
        size=N_MERCHANTS,
        p=[0.70, 0.225, 0.075],
    )
    categories = rng.choice(
        ["apparel", "home_goods", "electronics", "beauty", "specialty_food"],
        size=N_MERCHANTS,
    )
    complexity_map = {"small": (1, 2), "medium": (2, 4), "enterprise": (4, 5)}
    complexity = [
        int(rng.integers(complexity_map[t][0], complexity_map[t][1] + 1))
        for t in tiers
    ]
    return pd.DataFrame(
        {
            "merchant_id": np.arange(1, N_MERCHANTS + 1),
            "merchant_tier": tiers,
            "category": categories,
            "shipping_complexity": complexity,
        }
    )


def generate_users(rng: np.random.Generator, merchants: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Traffic weight by merchant tier
    tier_weights = merchants["merchant_tier"].map(
        {"small": 1.0, "medium": 3.5, "enterprise": 14.0}
    ).values
    tier_probs = tier_weights / tier_weights.sum()

    merchant_ids = rng.choice(merchants["merchant_id"].values, size=N_USERS, p=tier_probs)
    device_type = rng.choice(["mobile", "desktop"], size=N_USERS, p=[0.62, 0.38])

    # Browser depends on device
    browser = np.where(
        device_type == "mobile",
        rng.choice(["safari", "chrome", "other"], size=N_USERS, p=[0.52, 0.40, 0.08]),
        rng.choice(["chrome", "safari", "other"], size=N_USERS, p=[0.65, 0.20, 0.15]),
    )

    signup_offsets = rng.integers(30, 365, size=N_USERS)
    signup_dates = (PRE_START - pd.to_timedelta(signup_offsets, unit="D")).date

    users_df = pd.DataFrame(
        {
            "user_id": np.arange(1, N_USERS + 1),
            "merchant_id": merchant_ids,
            "device_type": device_type,
            "browser": browser,
            "signup_date": signup_dates,
        }
    )

    # Hidden latent user traits (used to generate realistic correlation & CUPED signal)
    latent_df = users_df.merge(
        merchants[["merchant_id", "merchant_tier"]], on="merchant_id", how="left"
    )
    latent_df["true_variant"] = rng.choice(["control", "treatment"], size=N_USERS, p=[0.5, 0.5])
    # Persistent user propensity for conversion and spending (creates CUPED correlation)
    latent_df["conv_propensity"] = rng.normal(0.0, 0.45, size=N_USERS)
    latent_df["spend_multiplier"] = np.exp(rng.normal(0.0, 0.55, size=N_USERS))
    latent_df["activity_rate"] = rng.gamma(shape=1.6, scale=1.2, size=N_USERS)

    return users_df, latent_df


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_period_activity(
    rng: np.random.Generator,
    latent_users: pd.DataFrame,
    period: str,
    start_ts: pd.Timestamp,
    session_id_start: int,
    order_id_start: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Number of sessions per user in this 28-day period (at least 1 in experiment so users get exposed)
    raw_counts = rng.poisson(latent_users["activity_rate"].values)
    session_counts = np.maximum(raw_counts, 1) if period == "experiment" else raw_counts
    session_counts = np.minimum(session_counts, 12)

    user_idx = np.repeat(np.arange(len(latent_users)), session_counts)
    n_sessions = len(user_idx)

    u_id = latent_users["user_id"].values[user_idx]
    m_id = latent_users["merchant_id"].values[user_idx]
    tiers = latent_users["merchant_tier"].values[user_idx]
    variants = latent_users["true_variant"].values[user_idx]
    conv_prop = latent_users["conv_propensity"].values[user_idx]
    spend_mult = latent_users["spend_multiplier"].values[user_idx]

    # Random timestamps across the 28-day window
    sec_offsets = rng.integers(0, PERIOD_DAYS * 86400, size=n_sessions)
    session_start = start_ts + pd.to_timedelta(sec_offsets, unit="s")
    week_num = (sec_offsets // (7 * 86400)) + 1  # 1, 2, 3, 4

    # Probability of reaching checkout
    reach_prob = np.where(tiers == "enterprise", 0.55, np.where(tiers == "medium", 0.50, 0.45))
    reached_checkout = rng.random(n_sessions) < reach_prob

    # Ensure every user has at least one checkout session during experiment for clean exposure triggering
    if period == "experiment":
        first_session_mask = np.zeros(n_sessions, dtype=bool)
        _, first_indices = np.unique(u_id, return_index=True)
        first_session_mask[first_indices] = True
        reached_checkout = reached_checkout | first_session_mask

    # Checkout latency (Guardrail metric: Trap 4b)
    base_latency = np.where(tiers == "enterprise", 540.0, np.where(tiers == "medium", 470.0, 410.0))
    latency_noise = rng.lognormal(mean=0.0, sigma=0.22, size=n_sessions)
    latency = base_latency * latency_noise

    if period == "experiment":
        is_treat = variants == "treatment"
        # Treatment adds small overhead for small/medium, severe latency spike on enterprise
        latency_penalty = np.where(
            tiers == "enterprise", 380.0, np.where(tiers == "medium", 45.0, 25.0)
        )
        latency = np.where(is_treat, latency + latency_penalty, latency)

    checkout_latency_ms = np.where(reached_checkout, np.round(latency, 2), np.nan)

    # Conversion probability (Trap 3: user-correlated + Trap 4a: novelty decay)
    # Base log-odds around -1.65 (~16% checkout conversion)
    log_odds = -1.65 + conv_prop + np.where(tiers == "enterprise", 0.15, 0.0)

    if period == "experiment":
        is_treat = variants == "treatment"
        # Novelty decay over weeks 1 to 4
        week_lift = np.select(
            [week_num == 1, week_num == 2, week_num == 3, week_num == 4],
            [0.24, 0.14, 0.09, 0.08],
            default=0.08,
        )
        # Enterprise latency hurts conversion lift on enterprise stores
        tier_adjustment = np.where(tiers == "enterprise", -0.07, 0.03)
        log_odds = np.where(is_treat, log_odds + week_lift + tier_adjustment, log_odds)

    conv_prob = _sigmoid(log_odds)
    converted = reached_checkout & (rng.random(n_sessions) < conv_prob)

    sessions_df = pd.DataFrame(
        {
            "session_id": np.arange(session_id_start, session_id_start + n_sessions),
            "user_id": u_id,
            "merchant_id": m_id,
            "session_start": session_start,
            "period": period,
            "reached_checkout": reached_checkout,
            "checkout_latency_ms": checkout_latency_ms,
            "converted": converted,
        }
    )

    # Generate orders for converted sessions (Trap 2: heavy-tailed GMV)
    conv_mask = converted
    n_orders = int(conv_mask.sum())
    ord_tiers = tiers[conv_mask]
    ord_spend_mult = spend_mult[conv_mask]
    ord_variants = variants[conv_mask]

    # Base AOV by tier with heavy right tail
    base_aov = np.where(ord_tiers == "enterprise", 195.0, np.where(ord_tiers == "medium", 85.0, 44.0))
    order_noise = rng.lognormal(mean=0.0, sigma=0.45, size=n_orders)
    gmv = base_aov * ord_spend_mult * order_noise

    # Slight AOV bump in treatment for small/medium (upfront free-shipping clarity encourages adding items)
    if period == "experiment":
        aov_lift = np.where((ord_variants == "treatment") & (ord_tiers != "enterprise"), 1.03, 1.0)
        gmv = gmv * aov_lift

    # Occasional whale orders on enterprise (heavy tail)
    whale_mask = (ord_tiers == "enterprise") & (rng.random(n_orders) < 0.015)
    gmv = np.where(whale_mask, gmv * rng.uniform(4.0, 8.0, size=n_orders), gmv)

    # Refund guardrail: baseline 4%, slight uptick to 5.4% on Enterprise treatment
    refund_prob = np.full(n_orders, 0.04)
    if period == "experiment":
        refund_prob = np.where(
            (ord_variants == "treatment") & (ord_tiers == "enterprise"), 0.054, refund_prob
        )
    is_refunded = rng.random(n_orders) < refund_prob

    # Order happens 1 to 10 minutes after session start
    order_ts = session_start[conv_mask] + pd.to_timedelta(
        rng.integers(60, 600, size=n_orders), unit="s"
    )

    orders_df = pd.DataFrame(
        {
            "order_id": np.arange(order_id_start, order_id_start + n_orders),
            "session_id": sessions_df.loc[conv_mask, "session_id"].values,
            "user_id": u_id[conv_mask],
            "merchant_id": m_id[conv_mask],
            "order_timestamp": order_ts,
            "period": period,
            "gmv_usd": np.round(gmv, 2),
            "is_refunded": is_refunded,
        }
    )

    return sessions_df, orders_df


def generate_exposures(
    rng: np.random.Generator,
    latent_users: pd.DataFrame,
    exp_sessions: pd.DataFrame,
) -> pd.DataFrame:
    # First checkout session in the experiment period triggers exposure
    checkout_sessions = exp_sessions[exp_sessions["reached_checkout"]].sort_values("session_start")
    first_exposures = checkout_sessions.groupby("user_id", as_index=False).first()

    exposures = first_exposures[["user_id", "merchant_id", "session_start"]].rename(
        columns={"session_start": "exposed_at"}
    )
    exposures = exposures.merge(
        latent_users[["user_id", "true_variant", "device_type", "browser"]],
        on="user_id",
        how="left",
    ).rename(columns={"true_variant": "variant"})

    # TRAP 1: Telemetry bug drops 10% of Treatment exposures on Mobile Safari
    bug_mask = (
        (exposures["variant"] == "treatment")
        & (exposures["device_type"] == "mobile")
        & (exposures["browser"] == "safari")
        & (rng.random(len(exposures)) < 0.10)
    )
    exposures = exposures[~bug_mask].copy()

    # Realistic log noise: ~0.3% duplicate exposure events
    dup_sample = exposures.sample(frac=0.003, random_state=SEED).copy()
    dup_sample["exposed_at"] = dup_sample["exposed_at"] + pd.Timedelta(seconds=45)

    # Realistic log noise: ~0.2% cross-variant contamination (user saw both variants)
    contam_sample = exposures.sample(frac=0.002, random_state=SEED + 1).copy()
    contam_sample["variant"] = np.where(
        contam_sample["variant"] == "control", "treatment", "control"
    )
    contam_sample["exposed_at"] = contam_sample["exposed_at"] + pd.Timedelta(days=2)

    exposures = pd.concat([exposures, dup_sample, contam_sample], ignore_index=True)
    exposures = exposures.sort_values("exposed_at").reset_index(drop=True)

    return exposures[["user_id", "merchant_id", "variant", "exposed_at", "device_type", "browser"]]


def load_to_postgres(csv_paths: dict[str, Path]) -> None:
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            print("Applying sql/schema.sql...")
            cur.execute(SCHEMA_PATH.read_text())

            # Load in foreign-key dependency order
            load_order = [
                ("merchants", "merchant_id, merchant_tier, category, shipping_complexity"),
                ("users", "user_id, merchant_id, device_type, browser, signup_date"),
                ("exposures", "user_id, merchant_id, variant, exposed_at, device_type, browser"),
                ("sessions", "session_id, user_id, merchant_id, session_start, period, reached_checkout, checkout_latency_ms, converted"),
                ("orders", "order_id, session_id, user_id, merchant_id, order_timestamp, period, gmv_usd, is_refunded"),
            ]

            for table, cols in load_order:
                path = csv_paths[table]
                with open(path, "r", encoding="utf-8") as f:
                    cur.copy_expert(
                        f"COPY {table} ({cols}) FROM STDIN WITH (FORMAT CSV, HEADER TRUE, NULL '')",
                        f,
                    )
                cur.execute(f"SELECT COUNT(*) FROM {table};")
                count = cur.fetchone()[0]
                print(f"Loaded {table:<10}: {count:,} rows")

        conn.commit()
        print("PostgreSQL load complete.")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("Generating merchants and users...")
    merchants_df = generate_merchants(rng)
    users_df, latent_users = generate_users(rng, merchants_df)

    print("Generating pre-experiment period activity...")
    pre_sessions, pre_orders = generate_period_activity(
        rng, latent_users, "pre", PRE_START, session_id_start=1, order_id_start=1
    )

    print("Generating experiment period activity...")
    exp_sessions, exp_orders = generate_period_activity(
        rng,
        latent_users,
        "experiment",
        EXP_START,
        session_id_start=len(pre_sessions) + 1,
        order_id_start=len(pre_orders) + 1,
    )

    print("Generating exposure logs (with mobile Safari telemetry bug)...")
    exposures_df = generate_exposures(rng, latent_users, exp_sessions)

    sessions_df = pd.concat([pre_sessions, exp_sessions], ignore_index=True)
    orders_df = pd.concat([pre_orders, exp_orders], ignore_index=True)

    tables = {
        "merchants": merchants_df,
        "users": users_df,
        "exposures": exposures_df,
        "sessions": sessions_df,
        "orders": orders_df,
    }

    csv_paths: dict[str, Path] = {}
    for name, df in tables.items():
        path = RAW_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
        csv_paths[name] = path
        print(f"Wrote {path} ({len(df):,} rows)")

    load_to_postgres(csv_paths)


if __name__ == "__main__":
    main()