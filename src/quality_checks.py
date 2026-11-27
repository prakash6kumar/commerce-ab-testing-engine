import os
from pathlib import Path
import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from scipy import stats

load_dotenv()

SQL_PIPELINE_PATH = Path("sql/build_experiment_tables.sql")

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


def run_sql_pipeline(conn) -> None:
    print("Running sql/build_experiment_tables.sql...")
    with conn.cursor() as cur:
        cur.execute(SQL_PIPELINE_PATH.read_text())
    conn.commit()
    print("Analytical tables built: exp_clean_exposures, exp_user_metrics, exp_session_metrics.\n")


def check_exposure_hygiene(conn) -> None:
    sql = """
    WITH raw_stats AS (
        SELECT COUNT(*) AS raw_rows, COUNT(DISTINCT user_id) AS raw_unique_users
        FROM exposures
    ),
    contam AS (
        SELECT COUNT(*) AS contaminated_users
        FROM (
            SELECT user_id
            FROM exposures
            GROUP BY user_id
            HAVING COUNT(DISTINCT variant) > 1
        ) t
    ),
    clean_stats AS (
        SELECT COUNT(*) AS clean_users
        FROM exp_clean_exposures
    )
    SELECT
        r.raw_rows,
        r.raw_unique_users,
        (r.raw_rows - r.raw_unique_users) AS duplicate_or_extra_rows,
        c.contaminated_users,
        cl.clean_users
    FROM raw_stats r, contam c, clean_stats cl;
    """
    df = query_df(conn, sql)
    row = df.iloc[0]

    print("=== 1. EXPOSURE HYGIENE AUDIT ===")
    print(f"Raw exposure rows:            {int(row['raw_rows']):,}")
    print(f"Unique users in raw logs:     {int(row['raw_unique_users']):,}")
    print(f"Duplicate/extra rows removed: {int(row['duplicate_or_extra_rows']):,}")
    print(f"Contaminated users excluded:  {int(row['contaminated_users']):,} (saw both variants)")
    print(f"Clean exposed users kept:     {int(row['clean_users']):,}\n")


def srm_chi_square(n_control: int, n_treatment: int, expected_ratio: float = 0.5) -> dict:
    total = n_control + n_treatment
    expected = [total * expected_ratio, total * (1.0 - expected_ratio)]
    chi2, p_val = stats.chisquare(f_obs=[n_control, n_treatment], f_exp=expected)
    return {
        "control": int(n_control),
        "treatment": int(n_treatment),
        "treat_share": n_treatment / total,
        "chi2": float(chi2),
        "p_value": float(p_val),
        "status": "FAIL (SRM)" if p_val < 0.001 else "PASS",
    }


def check_srm(conn) -> None:
    print("=== 2. SAMPLE RATIO MISMATCH (SRM) CHECKS (alpha = 0.001) ===")

    # Overall SRM
    overall_df = query_df(
        conn,
        """
        SELECT variant, COUNT(*) AS n
        FROM exp_user_metrics
        GROUP BY variant;
        """,
    )
    counts = dict(zip(overall_df["variant"], overall_df["n"]))
    res = srm_chi_square(counts["control"], counts["treatment"])
    print(
        f"OVERALL: Control={res['control']:,} | Treatment={res['treatment']:,} "
        f"| Treat Share={res['treat_share']:.4%} | p={res['p_value']:.2e} -> {res['status']}"
    )

    # Segmented SRM by device + browser
    print("\nDrill-down by Device & Browser:")
    seg_df = query_df(
        conn,
        """
        SELECT
            device_type,
            browser,
            COUNT(*) FILTER (WHERE variant = 'control') AS n_control,
            COUNT(*) FILTER (WHERE variant = 'treatment') AS n_treatment
        FROM exp_user_metrics
        GROUP BY device_type, browser
        ORDER BY device_type, browser;
        """,
    )
    rows = []
    for _, r in seg_df.iterrows():
        s = srm_chi_square(r["n_control"], r["n_treatment"])
        rows.append(
            {
                "segment": f"{r['device_type']} / {r['browser']}",
                "control": s["control"],
                "treatment": s["treatment"],
                "treat_share": f"{s['treat_share']:.2%}",
                "p_value": f"{s['p_value']:.2e}",
                "status": s["status"],
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))

    # Post-isolation SRM (excluding mobile / safari)
    clean_df = query_df(
        conn,
        """
        SELECT variant, COUNT(*) AS n
        FROM exp_user_metrics
        WHERE NOT is_mobile_safari
        GROUP BY variant;
        """,
    )
    clean_counts = dict(zip(clean_df["variant"], clean_df["n"]))
    clean_res = srm_chi_square(clean_counts["control"], clean_counts["treatment"])
    print(
        f"\nPOST-ISOLATION (Excl. Mobile Safari): "
        f"Control={clean_res['control']:,} | Treatment={clean_res['treatment']:,} "
        f"| Treat Share={clean_res['treat_share']:.4%} | p={clean_res['p_value']:.4f} -> {clean_res['status']}\n"
    )


def check_covariate_balance(conn) -> None:
    print("=== 3. PRE-EXPERIMENT COVARIATE BALANCE (Excl. Mobile Safari) ===")
    df = query_df(
        conn,
        """
        SELECT
            variant,
            merchant_tier,
            pre_sessions::FLOAT AS pre_sessions,
            pre_checkout_sessions::FLOAT AS pre_checkout_sessions,
            pre_orders::FLOAT AS pre_orders,
            pre_gmv_usd::FLOAT AS pre_gmv_usd
        FROM exp_user_metrics
        WHERE NOT is_mobile_safari;
        """,
    )

    ctrl = df[df["variant"] == "control"]
    treat = df[df["variant"] == "treatment"]

    metrics = ["pre_sessions", "pre_checkout_sessions", "pre_orders", "pre_gmv_usd"]
    results = []
    for col in metrics:
        c_vals = ctrl[col].values
        t_vals = treat[col].values
        _, p_val = stats.ttest_ind(t_vals, c_vals, equal_var=False)
        pooled_sd = np.sqrt((np.var(c_vals, ddof=1) + np.var(t_vals, ddof=1)) / 2.0)
        smd = (np.mean(t_vals) - np.mean(c_vals)) / pooled_sd if pooled_sd > 0 else 0.0
        results.append(
            {
                "covariate": col,
                "control_mean": round(float(np.mean(c_vals)), 4),
                "treatment_mean": round(float(np.mean(t_vals)), 4),
                "smd": round(float(smd), 4),
                "p_value": round(float(p_val), 4),
                "status": "PASS" if (p_val >= 0.05 and abs(smd) < 0.05) else "WARN",
            }
        )
    print(pd.DataFrame(results).to_string(index=False))

    # Categorical balance: merchant_tier
    contingency = pd.crosstab(df["merchant_tier"], df["variant"])
    chi2, p_val, _, _ = stats.chi2_contingency(contingency)
    print(
        f"\nMerchant Tier Mix Balance (Chi-Square): chi2={chi2:.4f}, p={p_val:.4f} -> "
        f"{'PASS' if p_val >= 0.05 else 'WARN'}\n"
    )


def main() -> None:
    conn = get_connection()
    try:
        run_sql_pipeline(conn)
        check_exposure_hygiene(conn)
        check_srm(conn)
        check_covariate_balance(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()