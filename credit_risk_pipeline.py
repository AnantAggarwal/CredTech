# credit_risk_pipeline_tickers.py
"""
Credit risk scoring + SHAP explanations for given tickers.
- Reads all parquet files from PARQUET_GLOB
- Train: 2015-2020, Test: 2021-2025 (adjustable)
- Tries LightGBM, falls back to XGBoost if needed
- Prints per-ticker latest test-date score + top SHAP drivers
- Prints global feature importance
"""

import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

# Try imports for model
MODEL_BACKEND = None
try:
    import lightgbm as lgb
    MODEL_BACKEND = "lightgbm"
except Exception:
    try:
        import xgboost as xgb
        MODEL_BACKEND = "xgboost"
    except Exception:
        raise ImportError("Install lightgbm or xgboost to run this script (prefer lightgbm).")

import shap

# -------------------------
# Config - edit paths here
# -------------------------
PARQUET_GLOB = "/Users/vaibhavkalyan/Desktop/ParquetReader/*.parquet"
DATE_COL = "date"
TICKER_COL = "ticker"

# Training / Test split years (change if you want)
TRAIN_YEARS = (2015, 2020)
TEST_YEARS  = (2021, 2025)

# Top-K SHAP features to show per ticker
TOPK = 6

# -------------------------
# Helpers
# -------------------------
def clean_colname(c):
    s = str(c)
    # remove tuple cruft and spaces
    s = s.replace("(", "").replace(")", "").replace("'", "").replace(",", "_").replace(" ", "")
    s = s.replace("__", "_")
    return s

def first_existing(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

def pct_change_grouped(df, by, col, periods=4):
    return df.groupby(by)[col].pct_change(periods=periods).replace([np.inf, -np.inf], np.nan)

# -------------------------
# 1) Load parquet files
# -------------------------
files = sorted(glob.glob(PARQUET_GLOB))
if not files:
    raise FileNotFoundError(f"No parquet files found at: {PARQUET_GLOB}")

# read and concat
dfs = []
for f in files:
    df = pd.read_parquet(f)
    # attach ticker if not present: use filename (apple.parquet -> apple)
    if "ticker" not in df.columns and "Ticker" not in df.columns:
        ticker_name = os.path.basename(f).split(".")[0]
        df["ticker"] = ticker_name
    dfs.append(df)
data = pd.concat(dfs, ignore_index=True)

# Clean column names
data.columns = [clean_colname(c) for c in data.columns]

# ensure date column present/parsed
if DATE_COL not in data.columns:
    raise KeyError(f"'{DATE_COL}' column missing after cleaning. Found columns: {list(data.columns)[:20]}")
data[DATE_COL] = pd.to_datetime(data[DATE_COL], errors="coerce")
data = data.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
data["year"] = data[DATE_COL].dt.year

# ensure ticker exists
if TICKER_COL not in data.columns:
    raise KeyError(f"'{TICKER_COL}' column not found. Provide ticker column or include tickers in filenames.")

print(f"Loaded {len(files)} files -> total rows: {len(data)}, columns: {len(data.columns)}")
print("Sample tickers:", data[TICKER_COL].unique().tolist())

# -------------------------
# 2) Engineer features (the list the user provided + a few robust ones)
# -------------------------
# Use available columns - adapt to your specific cleaned names
# Provided fields (cleaned expected): mkt__adjclose_mkt__bac, mkt__ret_1d_mkt__, fin__fin__diluted_eps, fin__fin__interest_expense, fin__fin__pretax_income, fin__bs_total_debt, fin__bs_common_stock_equity
# We'll attempt common variants

cols = set(data.columns)

# find EPS column
eps_candidates = [c for c in cols if "diluted_eps" in c or "eps" in c and "diluted" in c]
eps_col = eps_candidates[0] if eps_candidates else first_existing(cols, ["fin__fin__diluted_eps", "fin_fin_diluted_eps", "fin__fin__basic_eps"])
# interest expense / pretax income
interest_exp_candidates = first_existing(cols, ["fin__fin__interest_expense", "fin_fin_interest_expense", "fin__fin__interestexpense"])
pretax_candidates = first_existing(cols, ["fin__fin__pretax_income", "fin_fin_pretax_income", "fin__fin__pretaxincome"])
netincome_candidates = first_existing(cols, ["fin__fin__net_income", "fin_fin_net_income"])

# debt & equity
total_debt_cand = first_existing(cols, ["fin__bs__total_debt", "fin_bs_total_debt", "fin__bs__totaldebt"])
common_equity_cand = first_existing(cols, ["fin__bs__common_stock_equity", "fin_bs_common_stock_equity", "fin__bs__commonstockequity"])

# volatility candidates
vol_cand = first_existing(cols, ["mkt__vol_21d", "mkt_vol_21d", "mkt__vol_63d", "mkt_vol_21d_mkt__"])

print("Detected columns (examples): eps =", eps_col, "interest_exp =", interest_exp_candidates, "pretax =", pretax_candidates)

# Create engineered features (if possible)
data["feat_debt_to_equity"] = np.nan
if total_debt_cand and common_equity_cand:
    data["feat_debt_to_equity"] = data[total_debt_cand] / (data[common_equity_cand].abs() + 1e-9)

# EPS growth (quarterly-ish proxy using 4-period pct change per ticker)
if eps_col:
    data["feat_eps_growth"] = pct_change_grouped(data, "ticker", eps_col, periods=4)
else:
    data["feat_eps_growth"] = np.nan

# Interest coverage approximation
data["feat_interest_coverage"] = np.nan
if pretax_candidates and interest_exp_candidates:
    data["feat_interest_coverage"] = data[pretax_candidates].abs() / (data[interest_exp_candidates].abs() + 1e-9)

# Volatility
if vol_cand:
    data["feat_volatility"] = data[vol_cand]
else:
    # fallback to returns std if ret columns exist
    ret_cols = [c for c in cols if "mkt__ret" in c or "ret_21d" in c or "ret_63d" in c]
    if ret_cols:
        # rolling vol by ticker: use 21-day returns sd (approx)
        data["feat_volatility"] = data.groupby("ticker")[ret_cols[0]].rolling(window=21, min_periods=1).std().reset_index(level=0, drop=True)
    else:
        data["feat_volatility"] = np.nan

# Also copy some straightforward market features if present
possible_market = [c for c in cols if "mkt__adjclose" in c or "mkt__close" in c or "mkt__volume" in c]
for m in possible_market[:3]:
    data[f"copy_{m}"] = data[m]

# -------------------------
# 3) Create proxy target label (no real default data)
#    Heuristic: default if (net income < 0) AND (eps growth < 0) AND (interest coverage < 1)
# -------------------------
if "target_default" not in data.columns:
    proxy = pd.Series(False, index=data.index)
    if netincome_candidates is not None:
        proxy = proxy | (data[netincome_candidates] < 0)
    # require two signals: negative eps growth and weak interest coverage
    proxy = (data["feat_eps_growth"] < 0) & (data["feat_interest_coverage"] < 1.0)
    # if net income exists, include it as additional filter (makes proxy stricter)
    if netincome_candidates is not None:
        proxy = proxy & (data[netincome_candidates] < 0)
    data["target_default"] = proxy.astype(int)

target_col = "target_default"
print("Proxy target created. Default rate (overall):", data[target_col].mean())

# -------------------------
# 4) Train/test split by year
# -------------------------
train = data[(data["year"] >= TRAIN_YEARS[0]) & (data["year"] <= TRAIN_YEARS[1])].copy()
test  = data[(data["year"] >= TEST_YEARS[0]) & (data["year"] <= TEST_YEARS[1])].copy()

if train.empty or test.empty:
    raise ValueError("Empty train or test sets. Check TRAIN_YEARS / TEST_YEARS and your data timeline.")

print(f"Train rows {len(train)} ({TRAIN_YEARS[0]}-{TRAIN_YEARS[1]}), Test rows {len(test)} ({TEST_YEARS[0]}-{TEST_YEARS[1]})")

# -------------------------
# 5) Prepare feature matrix
# -------------------------
meta_cols = {DATE_COL, "year", TICKER_COL, target_col}
# choose numeric features excluding metadata
num_cols = [c for c in train.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(train[c])]

# for reproducibility, move engineered features to front (if present)
ordered_feats = [f for f in ["feat_debt_to_equity", "feat_eps_growth", "feat_interest_coverage", "feat_volatility"] if f in num_cols]
other_feats = [f for f in num_cols if f not in ordered_feats]
num_cols = ordered_feats + other_feats

print("Using numeric feature count:", len(num_cols))
print("Top features used (first 12):", num_cols[:12])

X_train = train[num_cols]
y_train = train[target_col].astype(int)
X_test  = test[num_cols]
y_test  = test[target_col].astype(int)

# Impute missing with median
imp = SimpleImputer(strategy="median")
X_train_imp = imp.fit_transform(X_train)
X_test_imp  = imp.transform(X_test)

# -------------------------
# 6) Train model (LightGBM preferred; fallback to XGBoost)
# -------------------------
if MODEL_BACKEND == "lightgbm":
    model = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.03, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    model.fit(X_train_imp, y_train)
    # predict probabilities for class 1
    y_proba = model.predict_proba(X_test_imp)[:, 1]
    model_type = "LightGBM"
else:
    # XGBoost fallback
    model = xgb.XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=6, use_label_encoder=False, eval_metric="logloss", random_state=42)
    model.fit(X_train_imp, y_train)
    y_proba = model.predict_proba(X_test_imp)[:, 1]
    model_type = "XGBoost"

print(f"Trained model using {model_type}")

# Evaluate AUC (note: proxy labels may be noisy)
try:
    auc = roc_auc_score(y_test, y_proba)
    print(f"[AUC on test {TEST_YEARS[0]}-{TEST_YEARS[1]}]: {auc:.4f}")
except Exception as e:
    print("Unable to compute AUC (likely single-class in test). Error:", e)

# Attach scores to test dataframe
test = test.assign(credit_risk_score=y_proba)

# -------------------------
# 7) SHAP explanations
# -------------------------
explainer = shap.TreeExplainer(model)
# compute shap on test set (be careful with size)
shap_vals = explainer.shap_values(X_test_imp)
# shap_values for class 1 (if model returns list)
if isinstance(shap_vals, list):
    shap_for_class1 = shap_vals[1]
else:
    shap_for_class1 = shap_vals

# Build per-ticker report (use latest row in test window per ticker)
report_rows = []
tickers = test[TICKER_COL].unique()
for t in tickers:
    df_t = test[test[TICKER_COL] == t].copy()
    if df_t.empty:
        continue
    # pick latest date row for this ticker in test set
    idx = df_t[DATE_COL].idxmax()
    row = df_t.loc[idx]
    row_idx_in_test = df_t.index.get_loc(idx)  # position inside df_t, but we need absolute index in test
    # find absolute index in test (position relative to X_test_imp)
    absolute_positions = np.where(test.index == idx)[0]
    if len(absolute_positions) == 0:
        # fallback: choose last occurrence index in test
        absolute_pos = test.index.get_indexer_for([idx])[0]
    else:
        absolute_pos = absolute_positions[0]

    score = row["credit_risk_score"]
    # SHAP vector for that row
    shap_vec = shap_for_class1[absolute_pos]
    # map feature -> shap
    feat_shap = list(zip(num_cols, shap_vec))
    feat_shap_sorted = sorted(feat_shap, key=lambda x: abs(x[1]), reverse=True)[:TOPK]
    report_rows.append({
        "ticker": t,
        "date": row[DATE_COL],
        "score": float(score),
        "top_shap": feat_shap_sorted
    })

# Print human friendly report
print("\n=== Per-ticker latest test-date credit risk + SHAP drivers ===")
for r in report_rows:
    print(f"\n[{r['ticker']}] date={r['date'].date()}  score={r['score']:.3f}")
    print(" Top drivers (feature : shap_value) :")
    for f, s in r["top_shap"]:
        sign = "+" if s > 0 else "-"
        print(f"   {f:<35} {sign}{abs(s):.4f}")

# -------------------------
# 8) Global feature importance
# -------------------------
if MODEL_BACKEND == "lightgbm":
    importances = model.feature_importances_
elif MODEL_BACKEND == "xgboost":
    importances = model.feature_importances_
else:
    importances = None

if importances is not None:
    fi = pd.Series(importances, index=num_cols).sort_values(ascending=False)
    print("\n=== Global feature importance (top 15) ===")
    print(fi.head(15))
else:
    print("No feature importance available for this model type.")

# -------------------------
# 9) Save outputs
# -------------------------
out_scores = test[[TICKER_COL, DATE_COL, "credit_risk_score"]].copy()
out_scores.to_csv("credit_risk_scores_test.csv", index=False)
print("\nSaved test scores to credit_risk_scores_test.csv")

# Also make a tidy SHAP table for the rows we reported
rows = []
for r in report_rows:
    for f, s in r["top_shap"]:
        rows.append({"ticker": r["ticker"], "date": r["date"], "feature": f, "shap": float(s)})
pd.DataFrame(rows).to_csv("shap_top_features_per_ticker.csv", index=False)
print("Saved per-ticker top SHAP to shap_top_features_per_ticker.csv")

print("\nDone.")
