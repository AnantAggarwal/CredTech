"""
credit_risk_pipeline.py
Credit risk scoring + SHAP explanations for given tickers.
- Reads all parquet files from parquet_glob
- Train: 2015-2020, Test: 2021-2025 (adjustable)
- Tries LightGBM, falls back to XGBoost if needed
- Returns per-ticker default probability (0=safe, 1=risky)
- Can be imported as a module: get_credit_scores(parquet_glob) -> {ticker: float}
- Or run standalone: python credit_risk_pipeline.py
"""

import os
import glob
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

# Try imports for model
def _get_model_backend():
    try:
        import lightgbm as lgb
        return "lightgbm", lgb
    except Exception:
        pass
    try:
        import xgboost as xgb
        return "xgboost", xgb
    except Exception:
        pass
    raise ImportError("Install lightgbm or xgboost (prefer lightgbm).")

# -------------------------
# Config — edit for standalone use
# -------------------------
DEFAULT_PARQUET_GLOB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "panel_*.parquet")
DATE_COL    = "date"
TICKER_COL  = "ticker"
TRAIN_YEARS = (2015, 2020)
TEST_YEARS  = (2021, 2025)
TOPK        = 6


# -------------------------
# Helpers
# -------------------------
def _clean_colname(c):
    s = str(c)
    s = s.replace("(", "").replace(")", "").replace("'", "").replace(",", "_").replace(" ", "")
    s = s.replace("__", "_")
    return s

def _first_existing(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

def _pct_change_grouped(df, by, col, periods=4):
    return df.groupby(by)[col].pct_change(periods=periods).replace([np.inf, -np.inf], np.nan)


# -------------------------
# Main importable function
# -------------------------
def get_credit_scores(parquet_glob=None):
    """
    Run the full credit risk ML pipeline on parquet panel files.

    Returns
    -------
    dict  {ticker: default_probability}
        default_probability ∈ [0, 1]  —  0 = very safe, 1 = very risky
    Returns an empty dict if parquet files are not found or the pipeline fails.
    """
    if parquet_glob is None:
        parquet_glob = DEFAULT_PARQUET_GLOB

    try:
        MODEL_BACKEND, lib = _get_model_backend()
    except ImportError as e:
        print(f"[credit_pipeline] {e}")
        return {}

    try:
        import shap
    except ImportError:
        print("[credit_pipeline] 'shap' not installed — install it: pip install shap")
        return {}

    # ------------------------------------------------------------------
    # 1) Load data from PostgreSQL
    # ------------------------------------------------------------------
    import psycopg2
    import json
    from dotenv import load_dotenv
    load_dotenv()
    db_url = os.environ.get("DATABASE_URL")
    
    data = pd.DataFrame()
    if db_url:
        try:
            conn = psycopg2.connect(db_url)
            query = "SELECT ticker, date, features FROM market_features"
            raw_df = pd.read_sql(query, conn)
            conn.close()
            
            if not raw_df.empty:
                # Highly memory-efficient JSON parsing to prevent OOM
                features_list = []
                for x in raw_df['features']:
                    if isinstance(x, str): features_list.append(json.loads(x))
                    elif x is None: features_list.append({})
                    else: features_list.append(x)
                
                features_df = pd.DataFrame(features_list)
                data = pd.concat([raw_df[['ticker', 'date']], features_df], axis=1)
                print(f"[credit_pipeline] Loaded {len(data)} rows from PostgreSQL")
            else:
                print("[credit_pipeline] No data found in PostgreSQL market_features")
        except Exception as e:
            print(f"[credit_pipeline] Error loading from PostgreSQL: {e}")
            
    # Fallback to Parquet if DB is empty or failed
    if data.empty:
        files = sorted(glob.glob(parquet_glob))
        if not files:
            print(f"[credit_pipeline] No data in DB and no parquet files found at: {parquet_glob}")
            return {}
    
        dfs = []
        for f in files:
            df = pd.read_parquet(f)
            if "ticker" not in df.columns and "Ticker" not in df.columns:
                ticker_name = os.path.basename(f).replace("panel_", "").split(".")[0]
                df["ticker"] = ticker_name
            dfs.append(df)
        data = pd.concat(dfs, ignore_index=True)

    # Clean column names
    data.columns = [_clean_colname(c) for c in data.columns]

    if DATE_COL not in data.columns:
        print(f"[credit_pipeline] '{DATE_COL}' column missing. Found: {list(data.columns)[:10]}")
        return {}

    data[DATE_COL] = pd.to_datetime(data[DATE_COL], errors="coerce")
    data = data.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    data["year"] = data[DATE_COL].dt.year

    if TICKER_COL not in data.columns:
        print(f"[credit_pipeline] '{TICKER_COL}' column not found.")
        return {}

    print(f"[credit_pipeline] Loaded {len(data)} rows, {len(data.columns)} cols")

    # ------------------------------------------------------------------
    # 2) Feature engineering
    # ------------------------------------------------------------------
    cols = set(data.columns)

    eps_candidates = [c for c in cols if "diluted_eps" in c or ("eps" in c and "diluted" in c)]
    eps_col = eps_candidates[0] if eps_candidates else _first_existing(cols, [
        "fin__fin__diluted_eps", "fin_fin_diluted_eps", "fin__fin__basic_eps"
    ])
    interest_exp_col = _first_existing(cols, [
        "fin__fin__interest_expense", "fin_fin_interest_expense", "fin__fin__interestexpense"
    ])
    pretax_col = _first_existing(cols, [
        "fin__fin__pretax_income", "fin_fin_pretax_income", "fin__fin__pretaxincome"
    ])
    netincome_col = _first_existing(cols, [
        "fin__fin__net_income", "fin_fin_net_income"
    ])
    total_debt_col = _first_existing(cols, [
        "fin__bs__total_debt", "fin_bs_total_debt", "fin__bs__totaldebt"
    ])
    common_equity_col = _first_existing(cols, [
        "fin__bs__common_stock_equity", "fin_bs_common_stock_equity", "fin__bs__commonstockequity"
    ])
    vol_col = _first_existing(cols, [
        "mkt__vol_21d", "mkt_vol_21d", "mkt__vol_63d", "mkt_vol_21d_mkt__"
    ])

    # Engineered features
    data["feat_debt_to_equity"] = np.nan
    if total_debt_col and common_equity_col:
        data["feat_debt_to_equity"] = data[total_debt_col] / (data[common_equity_col].abs() + 1e-9)

    data["feat_eps_growth"] = np.nan
    if eps_col:
        data["feat_eps_growth"] = _pct_change_grouped(data, "ticker", eps_col, periods=4)

    data["feat_interest_coverage"] = np.nan
    if pretax_col and interest_exp_col:
        data["feat_interest_coverage"] = (
            data[pretax_col].abs() / (data[interest_exp_col].abs() + 1e-9)
        )

    if vol_col:
        data["feat_volatility"] = data[vol_col]
    else:
        ret_cols = [c for c in cols if "mkt__ret" in c or "ret_21d" in c or "ret_63d" in c]
        if ret_cols:
            data["feat_volatility"] = (
                data.groupby("ticker")[ret_cols[0]]
                .rolling(window=21, min_periods=1).std()
                .reset_index(level=0, drop=True)
            )
        else:
            data["feat_volatility"] = np.nan

    # Copy a few market features
    market_cols = [c for c in cols if "mkt__adjclose" in c or "mkt__close" in c or "mkt__volume" in c]
    for m in market_cols[:3]:
        data[f"copy_{m}"] = data[m]

    # ------------------------------------------------------------------
    # 3) Proxy target label
    # ------------------------------------------------------------------
    if "target_default" not in data.columns:
        proxy = (data["feat_eps_growth"] < 0) & (data["feat_interest_coverage"] < 1.0)
        if netincome_col is not None:
            proxy = proxy & (data[netincome_col] < 0)
        data["target_default"] = proxy.astype(int)

    target_col = "target_default"
    print(f"[credit_pipeline] Proxy default rate: {data[target_col].mean():.3f}")

    # ------------------------------------------------------------------
    # 4) Train / test split
    # ------------------------------------------------------------------
    train = data[(data["year"] >= TRAIN_YEARS[0]) & (data["year"] <= TRAIN_YEARS[1])].copy()
    test  = data[(data["year"] >= TEST_YEARS[0])  & (data["year"] <= TEST_YEARS[1])].copy()

    if train.empty or test.empty:
        print("[credit_pipeline] Empty train or test sets — check data timeline.")
        return {}

    print(f"[credit_pipeline] Train: {len(train)} rows, Test: {len(test)} rows")

    # ------------------------------------------------------------------
    # 5) Feature matrix
    # ------------------------------------------------------------------
    meta_cols = {DATE_COL, "year", TICKER_COL, target_col}
    num_cols = [
        c for c in train.columns
        if c not in meta_cols and pd.api.types.is_numeric_dtype(train[c])
    ]
    ordered = [f for f in [
        "feat_debt_to_equity", "feat_eps_growth", "feat_interest_coverage", "feat_volatility"
    ] if f in num_cols]
    num_cols = ordered + [f for f in num_cols if f not in ordered]

    X_train = train[num_cols]
    y_train = train[target_col].astype(int)
    X_test  = test[num_cols]
    y_test  = test[target_col].astype(int)

    imp = SimpleImputer(strategy="median")
    X_train_imp = imp.fit_transform(X_train)
    X_test_imp  = imp.transform(X_test)

    # ------------------------------------------------------------------
    # 6) Train model
    # ------------------------------------------------------------------
    if MODEL_BACKEND == "lightgbm":
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=600, learning_rate=0.03, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            verbose=-1
        )
        model.fit(X_train_imp, y_train)
        y_proba = model.predict_proba(X_test_imp)[:, 1]
    else:
        import xgboost as xgb
        model = xgb.XGBClassifier(
            n_estimators=600, learning_rate=0.03, max_depth=6,
            use_label_encoder=False, eval_metric="logloss", random_state=42
        )
        model.fit(X_train_imp, y_train)
        y_proba = model.predict_proba(X_test_imp)[:, 1]

    print(f"[credit_pipeline] Trained {MODEL_BACKEND}")

    try:
        auc = roc_auc_score(y_test, y_proba)
        print(f"[credit_pipeline] Test AUC: {auc:.4f}")
    except Exception:
        pass

    test = test.assign(credit_risk_score=y_proba)

    # ------------------------------------------------------------------
    # 7) Per-ticker latest score
    # ------------------------------------------------------------------
    result = {}
    for ticker in test[TICKER_COL].unique():
        df_t = test[test[TICKER_COL] == ticker]
        if df_t.empty:
            continue
        idx = df_t[DATE_COL].idxmax()
        row = df_t.loc[idx]
        score = float(row["credit_risk_score"])
        result[ticker] = score

    print(f"[credit_pipeline] Scores: {result}")
    return result


# -------------------------
# Standalone entry point
# -------------------------
def main():
    import shap

    scores = get_credit_scores()
    if not scores:
        print("No scores computed.")
        return

    print("\n=== Per-ticker credit risk (default probability) ===")
    for ticker, prob in sorted(scores.items()):
        creditworthiness = (1 - prob) * 100
        print(f"  {ticker:<8} default_prob={prob:.4f}   creditworthiness={creditworthiness:.1f}/100")

    # Also save CSV for traceability
    rows = [{"ticker": t, "default_probability": p, "creditworthiness_score": (1-p)*100}
            for t, p in scores.items()]
    pd.DataFrame(rows).to_csv("credit_risk_scores_latest.csv", index=False)
    print("\nSaved to credit_risk_scores_latest.csv")


if __name__ == "__main__":
    main()
