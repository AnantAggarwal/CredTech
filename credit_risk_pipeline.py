# credit_risk_pipeline.py
import os
import glob
import math
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import lightgbm as lgb
import shap

import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------
PARQUET_GLOB = "/Users/vaibhavkalyan/Desktop/ParquetReader/*.parquet"
DATE_COL = "date"
TICKER_COL = "ticker"
# If you already have a label, set it here (e.g., "Default").
# If not present, the script will create a proxy label.
TARGET_COL_CANDIDATES = ["Default", "default", "default_flag", "label"]

DELPHI_TICKER_CANDIDATES = {"DLPH", "DELPHI", "DELPHI AUTOMOTIVE", "DELPHI AUTOMOTIVE PLC", "APTV"}  # APTV is post-spinoff

# -----------------------------
# Helpers
# -----------------------------
def clean_colname(c):
    # turn tuples/ugly names like "('mkt__adj close','mkt__bac')" -> mkt__adjclose_mkt__bac
    s = str(c)
    s = s.replace("(", "").replace(")", "").replace("'", "").replace(",", "_").replace(" ", "")
    s = s.replace("__", "_")  # mild normalization
    return s

def first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def pct_change_grouped(df, by, col, periods=4):
    # quarterly-ish growth proxy: % change over the last ~4 periods per ticker
    s = df.groupby(by)[col].pct_change(periods=periods)
    return s.replace([np.inf, -np.inf], np.nan)

# -----------------------------
# 1) Load & unify data
# -----------------------------
files = glob.glob(PARQUET_GLOB)
if not files:
    raise FileNotFoundError(f"No parquet files found at: {PARQUET_GLOB}")

dfs = [pd.read_parquet(f) for f in files]
data = pd.concat(dfs, ignore_index=True)

# Clean columns
data.columns = [clean_colname(c) for c in data.columns]

# Parse date and derive year
if DATE_COL not in data.columns:
    # try common variants after cleaning
    date_guess = first_existing(data, ["Date", "DATE"])
    if not date_guess:
        raise KeyError("No 'date' column found. Please provide a date column.")
    data.rename(columns={date_guess: DATE_COL}, inplace=True)

data[DATE_COL] = pd.to_datetime(data[DATE_COL], errors="coerce")
data = data.dropna(subset=[DATE_COL]).sort_values(DATE_COL)
data["year"] = data[DATE_COL].dt.year

# Ensure ticker exists
if TICKER_COL not in data.columns:
    t_guess = first_existing(data, ["symbol", "SYMBOL"])
    if not t_guess:
        raise KeyError("No 'ticker' column found. Please include a ticker column.")
    data.rename(columns={t_guess: TICKER_COL}, inplace=True)

# -----------------------------
# 2) Derive features (if missing)
# -----------------------------
# Debt-to-Equity ≈ total_debt / common_stock_equity
if "fin_bs_total_debt" in data.columns and "fin_bs_common_stock_equity" in data.columns:
    data["feat_debt_to_equity"] = data["fin_bs_total_debt"] / (data["fin_bs_common_stock_equity"].abs() + 1e-6)
else:
    data["feat_debt_to_equity"] = np.nan

# EPS growth from diluted EPS (YoY-ish proxy using 4 periods)
eps_col = first_existing(data, ["fin_fin_diluted_eps", "fin_fin_basic_eps"])
if eps_col:
    data["feat_eps_growth"] = pct_change_grouped(data, TICKER_COL, eps_col, periods=4)
else:
    data["feat_eps_growth"] = np.nan

# Interest Coverage ≈ |pretax_income| / (|interest_expense| + 1e-6)
if "fin_fin_pretax_income" in data.columns and "fin_fin_interest_expense" in data.columns:
    data["feat_interest_coverage"] = (
        data["fin_fin_pretax_income"].abs() / (data["fin_fin_interest_expense"].abs() + 1e-6)
    )
else:
    data["feat_interest_coverage"] = np.nan

# Volatility (prefer 21d market vol if present)
vol_cols_pref = ["mkt__vol_21d", "mkt_vol_21d", "mkt__vol_63d", "mkt_vol_63d"]
vol_pick = first_existing(data, vol_cols_pref)
if vol_pick:
    data["feat_volatility"] = data[vol_pick]
else:
    data["feat_volatility"] = np.nan

# Current Ratio (if you happen to have it; if not, stays NaN)
# Placeholder: if you had current_assets/current_liabilities, compute here.

# -----------------------------
# 3) Target label
# -----------------------------
target_col = first_existing(data, TARGET_COL_CANDIDATES)
if target_col is None:
    # Build a proxy default label (simple heuristic):
    # Default = 1 if (net income < 0) AND (EPS growth < 0) AND (interest coverage < 1.0)
    ni_col = first_existing(data, ["fin_fin_net_income", "fin_fin_net_income_common_stockholders"])
    ic = data["feat_interest_coverage"]
    eg = data["feat_eps_growth"]

    if ni_col is None:
        # fall back to a lighter heuristic: (eps_growth < 0) & (interest_coverage < 1)
        proxy = (eg < 0) & (ic < 1.0)
    else:
        proxy = (data[ni_col] < 0) & (eg < 0) & (ic < 1.0)

    data["target_default"] = proxy.astype(int)
    target_col = "target_default"
    print("No explicit default label found — using a proxy target 'target_default'.")
else:
    print(f"Using provided label: {target_col}")

# -----------------------------
# 4) Train/Test split by year
#    Train: 2010–2020, Test: 2021–2022
# -----------------------------
train = data[(data["year"] >= 2010) & (data["year"] <= 2020)].copy()
test  = data[(data["year"] >= 2021) & (data["year"] <= 2022)].copy()

if train.empty or test.empty:
    raise ValueError("Train or test split is empty. Check your date coverage.")

# Feature set: all numeric cols except target + metadata
meta_cols = {DATE_COL, "year", TICKER_COL}
drop_cols = meta_cols.union({target_col})
num_cols = [c for c in data.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(data[c])]

X_train = train[num_cols]
y_train = train[target_col].astype(int)
X_test  = test[num_cols]
y_test  = test[target_col].astype(int)

# Impute & (optionally) scale
imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train)
X_test_imp  = imputer.transform(X_test)

# -----------------------------
# 5) Train LightGBM
# -----------------------------
model = lgb.LGBMClassifier(
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=63,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train_imp, y_train)

# Evaluate
y_proba = model.predict_proba(X_test_imp)[:, 1]
auc = roc_auc_score(y_test, y_proba)
print(f"[AUC] Validation (2021–2022): {auc:.4f}")

# -----------------------------
# 6) Score Delphi (if present)
# -----------------------------
# Try to find a Delphi row in the TEST set
test_tickers = test[TICKER_COL].astype(str).str.upper()
delphi_mask = test_tickers.isin({t.upper() for t in DELPHI_TICKER_CANDIDATES}) | test_tickers.str.contains("DELPHI", na=False)
if delphi_mask.any():
    delphi_idx = np.where(delphi_mask.values)[0][0]
    delphi_row = X_test_imp[delphi_idx:delphi_idx+1]
    delphi_score = model.predict_proba(delphi_row)[:, 1][0]
    print(f"[Delphi] Credit Risk Score: {delphi_score:.2f}  (i.e., {delphi_score*100:.0f}% risk)")
else:
    print("[Delphi] No Delphi ticker found in 2021–2022 test set. Available tickers example:",
          test[TICKER_COL].dropna().astype(str).unique()[:10])

# -----------------------------
# 7) SHAP explanations
# -----------------------------
# TreeExplainer on LightGBM
explainer = shap.TreeExplainer(model)
# Use a manageable sample for speed
sample_n = min(1000, X_test_imp.shape[0])
X_test_sample = X_test_imp[:sample_n]
shap_values = explainer.shap_values(X_test_sample)

# Textual explanation for Delphi (or first test row) — no plot needed to run headless
explain_idx = np.where(delphi_mask.values)[0][0] if delphi_mask.any() else 0
row_vals = explainer.shap_values(X_test_imp[explain_idx:explain_idx+1])

# Map SHAP back to feature names
if isinstance(row_vals, list):
    # binary classifier returns [for_class0, for_class1]
    row_shap = row_vals[1][0]
    base_value = explainer.expected_value[1]
else:
    row_shap = row_vals[0]
    base_value = explainer.expected_value

# Sort contributions
contrib = sorted(zip(num_cols, row_shap), key=lambda x: abs(x[1]), reverse=True)[:10]
print("\n[SHAP] Top local drivers for this row:")
for name, val in contrib:
    direction = "UP" if val > 0 else "DOWN"
    print(f"  {name}: {val:+.2f} → pushed risk {direction}")

# Optional: show a bar plot (uncomment if running interactively with a display)
# shap.initjs()
# shap.force_plot(base_value, row_shap, pd.Series(X_test.iloc[explain_idx], index=num_cols))
# shap.summary_plot(shap_values[1] if isinstance(shap_values, list) else shap_values, features=X_test.iloc[:sample_n], feature_names=num_cols)

# -----------------------------
# 8) Global Feature Importance
# -----------------------------
importances = model.feature_importances_
fi = pd.Series(importances, index=num_cols).sort_values(ascending=False)
print("\n[Global Feature Importance] Top 15:")
print(fi.head(15))

# -----------------------------
# 9) Plain-language mapping example
# -----------------------------
# Try to reference the engineered feature names if present
pl_feat = "feat_debt_to_equity"
if pl_feat in num_cols:
    d2e_value = X_test.iloc[explain_idx][pl_feat]
    shap_val = dict(contrib).get(pl_feat, np.nan)
    direction = "increases" if shap_val > 0 else "decreases"
    print(f"\n[Plain Language] Model: 'Debt-to-Equity = {d2e_value:.2f}' → SHAP shows it {direction} risk (contribution {shap_val:+.2f}).")
else:
    print("\n[Plain Language] Debt-to-Equity feature not available; computed columns are 'feat_*'. "
          "If you provide current assets/liabilities, I’ll derive Current Ratio too.")

# -----------------------------
# 10) Nice-to-have prints
# -----------------------------
print("\n[Data ranges]")
print("Train years:", int(train['year'].min()), "→", int(train['year'].max()))
print("Test years :", int(test['year'].min()),  "→", int(test['year'].max()))
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
