import os
from pathlib import Path
import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from scipy import stats
from statsmodels.stats.multitest import multipletests

load_dotenv()

OUTPUT_DIR = Path("data/processed")

DB_CONFIG = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", 5432)),
    "dbname": os.getenv("PGDATABASE", "experimentation"),
    "user": os.getenv("PGUSER", "exp_user"),
    "password": os.getenv("PGPASSWORD", "exp_password"),
}


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def query_df(conn, sql: str) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql)
        cols = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)


def load_clean_user_metrics(conn) -> pd.DataFrame:
    sql = """
    SELECT
        user_id,
        merchant_tier,
        variant,
        pre_sessions::FLOAT AS pre_sessions,
        pre_converted::FLOAT AS pre_converted,
        pre_gmv_usd::FLOAT AS pre_gmv_usd,
        exp_checkout_sessions::FLOAT AS exp_checkout_sessions,
        exp_converted_sessions::FLOAT AS exp_converted_sessions,
        exp_orders::FLOAT AS exp_orders,
        exp_refunded_orders::FLOAT AS exp_refunded_orders,
        exp_gmv_usd::FLOAT AS exp_gmv_usd,
        exp_converted::FLOAT AS exp_converted,
        avg_checkout_latency_ms::FLOAT AS avg_checkout_latency_ms
    FROM exp_user_metrics
    WHERE NOT is_mobile_safari;
    """
    return query_df(conn, sql)


def welch_ttest(t_vals: np.ndarray, c_vals: np.ndarray, alpha: float = 0.05) -> dict:
    t_vals = t_vals[~np.isnan(t_vals)]
    c_vals = c_vals[~np.isnan(c_vals)]

    mean_t = float(np.mean(t_vals))
    mean_c = float(np.mean(c_vals))
    diff = mean_t - mean_c
    rel_lift = (diff / mean_c) if mean_c != 0 else np.nan

    var_t = float(np.var(t_vals, ddof=1))
    var_c = float(np.var(c_vals, ddof=1))
    se = float(np.sqrt(var_t / len(t_vals) + var_c / len(c_vals)))

    # Welch-Satterthwaite degrees of freedom
    num = (var_t / len(t_vals) + var_c / len(c_vals)) ** 2
    den = ((var_t / len(t_vals)) ** 2) / (len(t_vals) - 1) + (
        (var_c / len(c_vals)) ** 2
    ) / (len(c_vals) - 1)
    df = num / den

    t_stat = diff / se
    p_val = float(2 * stats.t.sf(abs(t_stat), df))
    t_crit = float(stats.t.ppf(1 - alpha / 2, df))

    return {
        "control_mean": mean_c,
        "treatment_mean": mean_t,
        "abs_diff": diff,
        "rel_lift": rel_lift,
        "se": se,
        "ci_low": diff - t_crit * se,
        "ci_high": diff + t_crit * se,
        "p_value": p_val,
    }


def cuped_ttest(
    df: pd.DataFrame, y_col: str, x_col: str, alpha: float = 0.05
) -> dict:
    """Applies pooled-theta CUPED adjustment and runs Welch's t-test."""
    y = df[y_col].values.astype(float)
    x = df[x_col].values.astype(float)

    # Pooled theta avoids introducing bias between variants
    cov_xy = np.cov(y, x, ddof=1)[0, 1]
    var_x = np.var(x, ddof=1)
    theta = cov_xy / var_x if var_x > 0 else 0.0

    y_cuped = y - theta * (x - np.mean(x))

    var_raw = float(np.var(y, ddof=1))
    var_adj = float(np.var(y_cuped, ddof=1))
    var_reduction = 1.0 - (var_adj / var_raw) if var_raw > 0 else 0.0

    is_treat = (df["variant"] == "treatment").values
    res = welch_ttest(y_cuped[is_treat], y_cuped[~is_treat], alpha=alpha)
    res["theta"] = float(theta)
    res["var_reduction"] = float(var_reduction)
    return res


def compute_mde(
    std: float, n_per_group: int, alpha: float = 0.05, power: float = 0.80
) -> float:
    """Two-sample absolute Minimum Detectable Effect."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    return float((z_alpha + z_beta) * np.sqrt(2 * (std ** 2) / n_per_group))


def delta_method_ratio_test(
    df: pd.DataFrame, num_col: str, den_col: str, alpha: float = 0.05
) -> dict:
    """
    Tests difference in ratio metric R = sum(num) / sum(den) between treatment
    and control using the Delta Method (valid under user-level randomization).
    """
    results = {}
    for variant in ["control", "treatment"]:
        sub = df[df["variant"] == variant]
        n = len(sub)
        num = sub[num_col].values.astype(float)
        den = sub[den_col].values.astype(float)

        mean_num = np.mean(num)
        mean_den = np.mean(den)
        ratio = mean_num / mean_den

        var_num = np.var(num, ddof=1)
        var_den = np.var(den, ddof=1)
        cov_nd = np.cov(num, den, ddof=1)[0, 1]

        # Delta method variance of ratio estimator
        var_ratio = (
            var_num - 2 * ratio * cov_nd + (ratio ** 2) * var_den
        ) / (n * (mean_den ** 2))
        results[variant] = {"ratio": float(ratio), "var": float(var_ratio)}

    r_c = results["control"]["ratio"]
    r_t = results["treatment"]["ratio"]
    diff = r_t - r_c
    se = float(np.sqrt(results["control"]["var"] + results["treatment"]["var"]))
    z_stat = diff / se
    p_val = float(2 * stats.norm.sf(abs(z_stat)))
    z_crit = float(stats.norm.ppf(1 - alpha / 2))

    return {
        "control_mean": r_c,
        "treatment_mean": r_t,
        "abs_diff": diff,
        "rel_lift": diff / r_c if r_c != 0 else np.nan,
        "se": se,
        "ci_low": diff - z_crit * se,
        "ci_high": diff + z_crit * se,
        "p_value": p_val,
    }


def naive_session_ztest(
    df: pd.DataFrame, num_col: str, den_col: str, alpha: float = 0.05
) -> dict:
    """Naive binomial z-test treating every session as independent (WRONG under user randomization)."""
    ctrl = df[df["variant"] == "control"]
    treat = df[df["variant"] == "treatment"]

    n_c = float(ctrl[den_col].sum())
    n_t = float(treat[den_col].sum())
    p_c = float(ctrl[num_col].sum()) / n_c
    p_t = float(treat[num_col].sum()) / n_t

    diff = p_t - p_c
    se = float(np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t))
    z_stat = diff / se
    p_val = float(2 * stats.norm.sf(abs(z_stat)))
    z_crit = float(stats.norm.ppf(1 - alpha / 2))

    return {
        "control_mean": p_c,
        "treatment_mean": p_t,
        "abs_diff": diff,
        "rel_lift": diff / p_c if p_c != 0 else np.nan,
        "se": se,
        "ci_low": diff - z_crit * se,
        "ci_high": diff + z_crit * se,
        "p_value": p_val,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = get_connection()
    try:
        df = load_clean_user_metrics(conn)
    finally:
        conn.close()

    ctrl = df[df["variant"] == "control"]
    treat = df[df["variant"] == "treatment"]
    n_per_group = min(len(ctrl), len(treat))

    # --- 1. CUPED vs. NAIVE COMPARISON ---
    print("=== 1. PRIMARY METRICS: NAIVE WELCH'S T-TEST vs. CUPED ===")
    gmv_raw = welch_ttest(treat["exp_gmv_usd"].values, ctrl["exp_gmv_usd"].values)
    gmv_cuped = cuped_ttest(df, "exp_gmv_usd", "pre_gmv_usd")

    conv_raw = welch_ttest(treat["exp_converted"].values, ctrl["exp_converted"].values)
    conv_cuped = cuped_ttest(df, "exp_converted", "pre_converted")

    cuped_table = pd.DataFrame(
        [
            {
                "metric": "User Conversion (Raw)",
                "ctrl": f"{conv_raw['control_mean']:.4f}",
                "treat": f"{conv_raw['treatment_mean']:.4f}",
                "abs_diff": f"{conv_raw['abs_diff']:+.4f}",
                "se": f"{conv_raw['se']:.5f}",
                "p_value": f"{conv_raw['p_value']:.4e}",
                "var_red": "0.0%",
            },
            {
                "metric": "User Conversion (CUPED)",
                "ctrl": f"{conv_cuped['control_mean']:.4f}",
                "treat": f"{conv_cuped['treatment_mean']:.4f}",
                "abs_diff": f"{conv_cuped['abs_diff']:+.4f}",
                "se": f"{conv_cuped['se']:.5f}",
                "p_value": f"{conv_cuped['p_value']:.4e}",
                "var_red": f"{conv_cuped['var_reduction']:.1%}",
            },
            {
                "metric": "GMV per User (Raw)",
                "ctrl": f"${gmv_raw['control_mean']:.2f}",
                "treat": f"${gmv_raw['treatment_mean']:.2f}",
                "abs_diff": f"${gmv_raw['abs_diff']:+.2f}",
                "se": f"${gmv_raw['se']:.3f}",
                "p_value": f"{gmv_raw['p_value']:.4e}",
                "var_red": "0.0%",
            },
            {
                "metric": "GMV per User (CUPED)",
                "ctrl": f"${gmv_cuped['control_mean']:.2f}",
                "treat": f"${gmv_cuped['treatment_mean']:.2f}",
                "abs_diff": f"${gmv_cuped['abs_diff']:+.2f}",
                "se": f"${gmv_cuped['se']:.3f}",
                "p_value": f"{gmv_cuped['p_value']:.4e}",
                "var_red": f"{gmv_cuped['var_reduction']:.1%}",
            },
        ]
    )
    print(cuped_table.to_string(index=False))

    # --- 2. POWER ANALYSIS & RUNTIME SAVED ---
    print("\n=== 2. POWER ANALYSIS & MDE (80% Power, alpha = 0.05) ===")
    gmv_std_raw = float(np.std(ctrl["exp_gmv_usd"].values, ddof=1))
    gmv_std_cuped = gmv_std_raw * np.sqrt(1 - gmv_cuped["var_reduction"])
    mde_gmv_raw = compute_mde(gmv_std_raw, n_per_group)
    mde_gmv_cuped = compute_mde(gmv_std_cuped, n_per_group)
    days_saved = 28.0 * gmv_cuped["var_reduction"]

    conv_std_raw = float(np.std(ctrl["exp_converted"].values, ddof=1))
    mde_conv_raw = compute_mde(conv_std_raw, n_per_group)

    print(f"Sample size per group (clean): {n_per_group:,}")
    print(
        f"User Conversion MDE (Raw):     {mde_conv_raw:.4f} "
        f"({mde_conv_raw / conv_raw['control_mean']:.2%} relative)"
    )
    print(
        f"GMV per User MDE (Raw):        ${mde_gmv_raw:.2f} "
        f"({mde_gmv_raw / gmv_raw['control_mean']:.2%} relative)"
    )
    print(
        f"GMV per User MDE (CUPED):      ${mde_gmv_cuped:.2f} "
        f"({mde_gmv_cuped / gmv_raw['control_mean']:.2%} relative)"
    )
    print(
        f"CUPED GMV Variance Reduction:  {gmv_cuped['var_reduction']:.1%} "
        f"(~{days_saved:.1f} days of equivalent experiment runtime saved)"
    )

    # --- 3. UNIT MISMATCH: NAIVE SESSION Z-TEST vs. DELTA METHOD ---
    print("\n=== 3. UNIT MISMATCH: SESSION CONVERSION (NAIVE vs. DELTA METHOD) ===")
    sess_naive = naive_session_ztest(df, "exp_converted_sessions", "exp_checkout_sessions")
    sess_delta = delta_method_ratio_test(df, "exp_converted_sessions", "exp_checkout_sessions")
    se_inflation = sess_delta["se"] / sess_naive["se"]

    unit_table = pd.DataFrame(
        [
            {
                "method": "Naive Session Z-Test (Biased)",
                "ctrl_rate": f"{sess_naive['control_mean']:.4%}",
                "treat_rate": f"{sess_naive['treatment_mean']:.4%}",
                "abs_diff": f"{sess_naive['abs_diff']:+.4%}",
                "se": f"{sess_naive['se']:.6f}",
                "p_value": f"{sess_naive['p_value']:.4e}",
            },
            {
                "method": "Delta Method (User-Clustered)",
                "ctrl_rate": f"{sess_delta['control_mean']:.4%}",
                "treat_rate": f"{sess_delta['treatment_mean']:.4%}",
                "abs_diff": f"{sess_delta['abs_diff']:+.4%}",
                "se": f"{sess_delta['se']:.6f}",
                "p_value": f"{sess_delta['p_value']:.4e}",
            },
        ]
    )
    print(unit_table.to_string(index=False))
    print(
        f"SE Inflation Factor (Delta / Naive): {se_inflation:.3f}x "
        f"(Naive test understates SE by {(1 - 1/se_inflation):.1%})"
    )

    # --- 4. GUARDRAILS & BENJAMINI-HOCHBERG (FDR) CORRECTION ---
    print("\n=== 4. OVERALL METRIC FAMILY WITH BENJAMINI-HOCHBERG (FDR = 0.05) ===")
    latency_res = welch_ttest(
        treat["avg_checkout_latency_ms"].values,
        ctrl["avg_checkout_latency_ms"].values,
    )
    refund_res = delta_method_ratio_test(df, "exp_refunded_orders", "exp_orders")

    summary_rows = [
        {"metric": "User Conversion (CUPED)", "type": "Primary", **conv_cuped},
        {"metric": "GMV per User (CUPED)", "type": "Primary", **gmv_cuped},
        {"metric": "Session Conversion (Delta)", "type": "Primary", **sess_delta},
        {"metric": "Avg Checkout Latency (ms)", "type": "Guardrail", **latency_res},
        {"metric": "Order Refund Rate (Delta)", "type": "Guardrail", **refund_res},
    ]
    summary_df = pd.DataFrame(summary_rows)

    rejected, p_adj, _, _ = multipletests(
        summary_df["p_value"].values, alpha=0.05, method="fdr_bh"
    )
    summary_df["p_value_bh"] = p_adj
    summary_df["significant_bh"] = rejected

    display_df = summary_df[
        [
            "metric",
            "type",
            "control_mean",
            "treatment_mean",
            "abs_diff",
            "rel_lift",
            "p_value",
            "p_value_bh",
            "significant_bh",
        ]
    ].copy()
    display_df["control_mean"] = display_df["control_mean"].map(lambda v: f"{v:.4f}")
    display_df["treatment_mean"] = display_df["treatment_mean"].map(lambda v: f"{v:.4f}")
    display_df["abs_diff"] = display_df["abs_diff"].map(lambda v: f"{v:+.4f}")
    display_df["rel_lift"] = display_df["rel_lift"].map(lambda v: f"{v:+.2%}")
    display_df["p_value"] = display_df["p_value"].map(lambda v: f"{v:.4e}")
    display_df["p_value_bh"] = display_df["p_value_bh"].map(lambda v: f"{v:.4e}")

    print(display_df.to_string(index=False))

    out_path = OUTPUT_DIR / "day3_overall_results.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"\nSaved overall statistical results to {out_path}")


if __name__ == "__main__":
    main()