

import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler

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
# Config
# -------------------------
PARQUET_GLOB = "/Users/vaibhavkalyan/Desktop/ParquetReader/*.parquet"
DATE_COL = "date"
TICKER_COL = "ticker"

TRAIN_YEARS = (2015, 2020)
TEST_YEARS  = (2021, 2025)
TOPK = 6

# -------------------------
# Helpers
# -------------------------
def clean_colname(c):
    s = str(c)
    s = s.replace("(", "").replace(")", "").replace("'", "").replace(",", "_").replace(" ", "")
    s = s.replace("__", "_").strip("_")
    return s

def find_column(df, patterns):
    """Find first matching column from patterns"""
    cols = df.columns.tolist()
    for pattern in patterns:
        matches = [c for c in cols if pattern.lower() in c.lower()]
        if matches:
            return matches[0]
    return None

def safe_divide(numerator, denominator, default=np.nan):
    """Safe division with default value"""
    result = np.full_like(numerator, default, dtype=float)
    mask = (denominator != 0) & (~np.isnan(denominator)) & (~np.isnan(numerator))
    result[mask] = numerator[mask] / denominator[mask]
    return result

# -------------------------
# 1) Load and prepare data
# -------------------------
files = sorted(glob.glob(PARQUET_GLOB))
if not files:
    raise FileNotFoundError(f"No parquet files found at: {PARQUET_GLOB}")

print(f"Found {len(files)} files")

dfs = []
for f in files:
    df = pd.read_parquet(f)
    # Add ticker from filename if missing
    if "ticker" not in df.columns and "Ticker" not in df.columns:
        ticker_name = os.path.basename(f).split(".")[0].upper()
        df["ticker"] = ticker_name
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(data)} total rows")

# Clean column names
data.columns = [clean_colname(c) for c in data.columns]

# Handle date column
if DATE_COL not in data.columns:
    raise KeyError(f"'{DATE_COL}' column missing. Found: {list(data.columns)[:10]}")

data[DATE_COL] = pd.to_datetime(data[DATE_COL], errors="coerce")
data = data.dropna(subset=[DATE_COL]).sort_values([TICKER_COL, DATE_COL]).reset_index(drop=True)
data["year"] = data[DATE_COL].dt.year

print(f"After cleaning: {len(data)} rows, {len(data.columns)} columns")
print("Tickers:", sorted(data[TICKER_COL].unique()))
print("Date range:", data[DATE_COL].min(), "to", data[DATE_COL].max())

# -------------------------
# 2) Enhanced Feature Engineering
# -------------------------
print("\n=== Feature Engineering ===")

# Find key financial columns with flexible matching
revenue_col = find_column(data, ["total_revenue", "revenue", "sales"])
net_income_col = find_column(data, ["net_income"])
total_debt_col = find_column(data, ["total_debt", "debt"])
equity_col = find_column(data, ["common_stock_equity", "stockholders_equity", "equity"])
assets_col = find_column(data, ["total_assets"])
current_assets_col = find_column(data, ["current_assets"])
current_liab_col = find_column(data, ["current_liabilities"])
eps_col = find_column(data, ["diluted_eps", "eps"])
interest_exp_col = find_column(data, ["interest_expense"])
operating_income_col = find_column(data, ["operating_income"])

print(f"Key columns found:")
print(f"  Revenue: {revenue_col}")
print(f"  Net Income: {net_income_col}")
print(f"  Total Debt: {total_debt_col}")
print(f"  Equity: {equity_col}")
print(f"  EPS: {eps_col}")

# Market data columns
price_col = find_column(data, ["adjclose", "close"])
volume_col = find_column(data, ["volume"])

# Create engineered features
features_created = 0

# 1. Debt-to-Equity Ratio
if total_debt_col and equity_col:
    data["debt_to_equity"] = safe_divide(data[total_debt_col], data[equity_col].abs())
    features_created += 1
    print(f"Created debt_to_equity (non-null: {data['debt_to_equity'].notna().sum()})")

# 2. Current Ratio
if current_assets_col and current_liab_col:
    data["current_ratio"] = safe_divide(data[current_assets_col], data[current_liab_col])
    features_created += 1
    print(f"Created current_ratio (non-null: {data['current_ratio'].notna().sum()})")

# 3. ROA (Return on Assets)
if net_income_col and assets_col:
    data["roa"] = safe_divide(data[net_income_col], data[assets_col])
    features_created += 1
    print(f"Created roa (non-null: {data['roa'].notna().sum()})")

# 4. Profit Margin
if net_income_col and revenue_col:
    data["profit_margin"] = safe_divide(data[net_income_col], data[revenue_col])
    features_created += 1
    print(f"Created profit_margin (non-null: {data['profit_margin'].notna().sum()})")

# 5. Interest Coverage
if operating_income_col and interest_exp_col:
    data["interest_coverage"] = safe_divide(data[operating_income_col], data[interest_exp_col].abs())
    features_created += 1
    print(f"Created interest_coverage (non-null: {data['interest_coverage'].notna().sum()})")

# 6. Revenue and earnings growth (year-over-year)
for ticker in data[TICKER_COL].unique():
    mask = data[TICKER_COL] == ticker
    if revenue_col:
        data.loc[mask, "revenue_growth_yoy"] = data.loc[mask, revenue_col].pct_change(4)  # Quarterly data
    if net_income_col:
        data.loc[mask, "earnings_growth_yoy"] = data.loc[mask, net_income_col].pct_change(4)
    if eps_col:
        data.loc[mask, "eps_growth_yoy"] = data.loc[mask, eps_col].pct_change(4)

if revenue_col:
    features_created += 1
    print(f"Created revenue_growth_yoy (non-null: {data['revenue_growth_yoy'].notna().sum()})")

# 7. Price-based features (if available)
if price_col:
    for ticker in data[TICKER_COL].unique():
        mask = data[TICKER_COL] == ticker
        # Price volatility (rolling standard deviation)
        data.loc[mask, "price_volatility"] = data.loc[mask, price_col].rolling(window=21, min_periods=5).std()
        # Price momentum (3-month return)
        data.loc[mask, "price_momentum"] = data.loc[mask, price_col].pct_change(12)  # 3 months if quarterly
    
    features_created += 2
    print(f"Created price_volatility (non-null: {data['price_volatility'].notna().sum()})")

print(f"Total engineered features: {features_created}")

# -------------------------
# 3) FIXED Target Variable Creation for High-Quality Stocks
# -------------------------
print("\n=== Creating Relative Performance Target Variable ===")

# Create a composite financial health score
health_metrics = []

# Normalize metrics to 0-1 scale for combination
def normalize_metric(series, higher_is_better=True):
    """Normalize to 0-1 scale, handling outliers"""
    q1, q99 = series.quantile([0.01, 0.99])
    clipped = series.clip(q1, q99)
    
    if higher_is_better:
        normalized = (clipped - clipped.min()) / (clipped.max() - clipped.min() + 1e-8)
    else:
        normalized = (clipped.max() - clipped) / (clipped.max() - clipped.min() + 1e-8)
    
    return normalized

# 1. Profitability metrics
if 'roa' in data.columns:
    data['roa_normalized'] = normalize_metric(data['roa'], higher_is_better=True)
    health_metrics.append('roa_normalized')
    
if 'profit_margin' in data.columns:
    data['profit_margin_normalized'] = normalize_metric(data['profit_margin'], higher_is_better=True)
    health_metrics.append('profit_margin_normalized')

# 2. Leverage metrics  
if 'debt_to_equity' in data.columns:
    data['debt_to_equity_normalized'] = normalize_metric(data['debt_to_equity'], higher_is_better=False)
    health_metrics.append('debt_to_equity_normalized')

# 3. Liquidity metrics
if 'current_ratio' in data.columns:
    data['current_ratio_normalized'] = normalize_metric(data['current_ratio'], higher_is_better=True)
    health_metrics.append('current_ratio_normalized')

# 4. Growth metrics
if 'revenue_growth_yoy' in data.columns:
    # Handle negative growth appropriately
    growth_metric = data['revenue_growth_yoy'].fillna(0)
    # Convert to relative performance (above/below median)
    data['revenue_growth_normalized'] = (growth_metric > growth_metric.median()).astype(float)
    health_metrics.append('revenue_growth_normalized')

if 'earnings_growth_yoy' in data.columns:
    growth_metric = data['earnings_growth_yoy'].fillna(0)
    data['earnings_growth_normalized'] = (growth_metric > growth_metric.median()).astype(float)
    health_metrics.append('earnings_growth_normalized')

# 5. Market performance
if 'price_momentum' in data.columns:
    momentum_metric = data['price_momentum'].fillna(0)
    data['price_momentum_normalized'] = (momentum_metric > momentum_metric.median()).astype(float)
    health_metrics.append('price_momentum_normalized')

print(f"Health metrics used: {health_metrics}")

# Create composite health score
if health_metrics:
    data['health_score'] = data[health_metrics].fillna(0.5).mean(axis=1)
    print(f"Health score range: {data['health_score'].min():.3f} to {data['health_score'].max():.3f}")
else:
    print("Warning: No health metrics available, using revenue-based scoring")
    if revenue_col:
        data['health_score'] = normalize_metric(data[revenue_col], higher_is_better=True)
    else:
        data['health_score'] = np.random.random(len(data))  # Last resort

# Create target using multiple approaches
# Method 1: Time-based relative performance (worst performing quarters)
data['target_temporal'] = 0
for ticker in data[TICKER_COL].unique():
    mask = data[TICKER_COL] == ticker
    ticker_scores = data.loc[mask, 'health_score']
    # Mark bottom 20% of quarters for each company as "higher risk"
    threshold = ticker_scores.quantile(0.2)
    data.loc[mask & (data['health_score'] <= threshold), 'target_temporal'] = 1

# Method 2: Cross-sectional comparison (worst performers at each time point)
data['target_crosssectional'] = 0
for date in data[DATE_COL].dt.to_period('Q').unique():
    date_mask = data[DATE_COL].dt.to_period('Q') == date
    if date_mask.sum() > 1:  # Need multiple companies
        period_scores = data.loc[date_mask, 'health_score']
        threshold = period_scores.quantile(0.3)  # Bottom 30%
        data.loc[date_mask & (data['health_score'] <= threshold), 'target_crosssectional'] = 1

# Method 3: Absolute thresholds (very conservative for quality stocks)
data['target_absolute'] = 0
# Mark observations with multiple concerning signals
concerning_signals = 0

if 'roa' in data.columns:
    concerning_signals += (data['roa'] < data['roa'].quantile(0.1)).astype(int)

if 'profit_margin' in data.columns:
    concerning_signals += (data['profit_margin'] < data['profit_margin'].quantile(0.1)).astype(int)

if 'debt_to_equity' in data.columns:
    concerning_signals += (data['debt_to_equity'] > data['debt_to_equity'].quantile(0.9)).astype(int)

data['target_absolute'] = (concerning_signals >= 2).astype(int)

# Combine methods - use OR logic (any method flags as risky)
data['target_default'] = (
    (data['target_temporal'] == 1) | 
    (data['target_crosssectional'] == 1) | 
    (data['target_absolute'] == 1)
).astype(int)

# Report on different methods
print(f"\nTarget creation results:")
print(f"Temporal method: {data['target_temporal'].mean():.1%} ({data['target_temporal'].sum()} cases)")
print(f"Cross-sectional method: {data['target_crosssectional'].mean():.1%} ({data['target_crosssectional'].sum()} cases)")
print(f"Absolute method: {data['target_absolute'].mean():.1%} ({data['target_absolute'].sum()} cases)")
print(f"Combined target: {data['target_default'].mean():.1%} ({data['target_default'].sum()} cases)")

# If still no defaults, use a guaranteed method
if data['target_default'].sum() == 0:
    print("No defaults found with standard methods. Using bottom 10% by health score.")
    threshold = data['health_score'].quantile(0.1)
    data['target_default'] = (data['health_score'] <= threshold).astype(int)

default_rate = data['target_default'].mean()
print(f"\nFinal default rate: {default_rate:.1%} ({data['target_default'].sum()} higher-risk periods out of {len(data)})")

if default_rate == 0:
    print("ERROR: Still no target cases detected. Check data quality.")
    exit(1)

# -------------------------
# 4) Train/Test Split and Feature Selection
# -------------------------
train = data[(data["year"] >= TRAIN_YEARS[0]) & (data["year"] <= TRAIN_YEARS[1])].copy()
test  = data[(data["year"] >= TEST_YEARS[0]) & (data["year"] <= TEST_YEARS[1])].copy()

print(f"\nTrain: {len(train)} rows ({train['target_default'].mean():.1%} default rate)")
print(f"Test: {len(test)} rows ({test['target_default'].mean():.1%} default rate)")

# Select features (engineered + some market features)
engineered_features = [
    'debt_to_equity', 'current_ratio', 'roa', 'profit_margin', 'interest_coverage',
    'revenue_growth_yoy', 'earnings_growth_yoy', 'eps_growth_yoy', 
    'price_volatility', 'price_momentum', 'health_score'
]

# Add some raw financial features
meta_cols = {DATE_COL, "year", TICKER_COL, "target_default", "target_temporal", 
             "target_crosssectional", "target_absolute", "health_score"}
meta_cols.update([col for col in data.columns if col.endswith('_normalized')])

numeric_cols = [c for c in data.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(data[c])]

# Prioritize engineered features
feature_cols = [f for f in engineered_features if f in data.columns]
# Add some raw features that have good coverage
for col in numeric_cols:
    if len(feature_cols) < 50:  # Limit to prevent overfitting
        non_null_pct = data[col].notna().mean()
        if non_null_pct > 0.1:  # At least 10% coverage
            feature_cols.append(col)

print(f"Using {len(feature_cols)} features")
print(f"Top features: {feature_cols[:10]}")

# Prepare feature matrices
X_train = train[feature_cols]
y_train = train["target_default"]
X_test = test[feature_cols]
y_test = test["target_default"]

# Check for class balance
print(f"Train class distribution: {y_train.value_counts().to_dict()}")
print(f"Test class distribution: {y_test.value_counts().to_dict()}")

# Imputation and scaling
imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

# Scale features for stability
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)

# Update feature_cols to match the actual features used by the model
# Some features might be dropped during imputation if they're all NaN
n_features_after_processing = X_train_scaled.shape[1]
print(f"Features after imputation and scaling: {n_features_after_processing}")

# Keep track of which features survived the preprocessing
if n_features_after_processing < len(feature_cols):
    print(f"⚠️  Some features were dropped during preprocessing")
    print(f"Original features: {len(feature_cols)}, Final features: {n_features_after_processing}")
    # Truncate feature_cols to match the actual number of features
    feature_cols = feature_cols[:n_features_after_processing]
elif n_features_after_processing > len(feature_cols):
    print(f"⚠️  More features than expected after preprocessing")
    # This shouldn't happen with our current setup, but handle it
    feature_cols = feature_cols + [f"generated_feature_{i}" for i in range(len(feature_cols), n_features_after_processing)]

# -------------------------
# 5) Model Training
# -------------------------
print(f"\n=== Training {MODEL_BACKEND.upper()} Model ===")

if MODEL_BACKEND == "lightgbm":
    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight='balanced',  # Handle class imbalance
        random_state=42,
        verbosity=-1
    )
    model.fit(X_train_scaled, y_train)
    y_train_pred = model.predict_proba(X_train_scaled)[:, 1]
    y_test_pred = model.predict_proba(X_test_scaled)[:, 1]
else:
    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=len(y_train[y_train==0])/max(len(y_train[y_train==1]), 1),  # Handle imbalance
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train_scaled, y_train)
    y_train_pred = model.predict_proba(X_train_scaled)[:, 1]
    y_test_pred = model.predict_proba(X_test_scaled)[:, 1]

# Evaluation
try:
    train_auc = roc_auc_score(y_train, y_train_pred)
    test_auc = roc_auc_score(y_test, y_test_pred)
    print(f"Train AUC: {train_auc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
except Exception as e:
    print(f"Could not compute AUC: {e}")
    train_auc = test_auc = None

# -------------------------
# 6) SHAP Analysis
# -------------------------
print("\n=== SHAP Analysis ===")

# Use a subset for SHAP to avoid memory issues
shap_sample_size = min(500, len(X_test_scaled))
shap_indices = np.random.choice(len(X_test_scaled), shap_sample_size, replace=False)

try:
    # For LightGBM, use the original imputed data (not scaled) for better SHAP performance
    if MODEL_BACKEND == "lightgbm":
        explainer = shap.TreeExplainer(model)
        # LightGBM works better with original scale data for SHAP
        shap_values = explainer.shap_values(X_test_imp[shap_indices])
    else:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_scaled[shap_indices])

    # Handle SHAP output format - LightGBM returns single array for binary classification
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values_class1 = shap_values[1]  # Class 1 (higher risk)
    elif isinstance(shap_values, list) and len(shap_values) == 1:
        shap_values_class1 = shap_values[0]  # Single class output
    else:
        shap_values_class1 = shap_values  # Direct array
    
    shap_available = True
    print(f"SHAP analysis completed successfully with {shap_sample_size} samples")
    
except Exception as e:
    print(f"SHAP analysis failed: {e}")
    shap_available = False
    shap_values_class1 = np.zeros((len(shap_indices), len(feature_cols)))

# Per-ticker analysis (latest date)
print("\n=== Per-Ticker Credit Risk Report ===")
test_with_scores = test.copy()
test_with_scores['credit_risk_score'] = y_test_pred

ticker_reports = []
for ticker in sorted(test[TICKER_COL].unique()):
    ticker_data = test_with_scores[test_with_scores[TICKER_COL] == ticker]
    if ticker_data.empty:
        continue
    
    # Get latest observation
    latest_idx = ticker_data[DATE_COL].idxmax()
    latest_row = ticker_data.loc[latest_idx]
    
    # Find corresponding index in test set for SHAP
    test_pos = np.where(test.index == latest_idx)[0]
    if len(test_pos) > 0 and test_pos[0] < len(shap_indices) and shap_available:
        # Check if this test position is in our SHAP sample
        shap_idx = np.where(shap_indices == test_pos[0])[0]
        if len(shap_idx) > 0:
            # Get SHAP values for this observation
            shap_vec = shap_values_class1[shap_idx[0]]
            feature_importance = list(zip(feature_cols, shap_vec))
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            top_features = feature_importance[:TOPK]
        else:
            # This observation wasn't in SHAP sample, use model feature importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_importance = list(zip(feature_cols, importances))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                top_features = feature_importance[:TOPK]
            else:
                top_features = [(f, 0.0) for f in feature_cols[:TOPK]]
    else:
        # Use model feature importance as fallback
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = list(zip(feature_cols, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            top_features = feature_importance[:TOPK]
        else:
            top_features = [(f, 0.0) for f in feature_cols[:TOPK]]
    
    ticker_reports.append({
        'ticker': ticker,
        'date': latest_row[DATE_COL],
        'score': latest_row['credit_risk_score'],
        'actual_target': latest_row['target_default'],
        'health_score': latest_row.get('health_score', 0),
        'top_features': top_features
    })

# Print ticker reports
print("\n" + "="*80)
for report in ticker_reports:
    risk_level = "HIGH" if report['score'] > 0.7 else "MEDIUM" if report['score'] > 0.3 else "LOW"
    status_icon = "⚠️ " if report['actual_target'] else "✅ "
    
    print(f"\n{status_icon}[{report['ticker']}] {report['date'].strftime('%Y-%m-%d')}")
    print(f"   Risk Score: {report['score']:.3f} | Health Score: {report['health_score']:.3f} | Risk Level: {risk_level}")
    
    if report['actual_target']:
        print("   📈 Flagged as higher risk period in historical data")
    
    print("   🔍 Top risk factors:")
    for i, (feature, importance) in enumerate(report['top_features'], 1):
        direction = "↑" if importance > 0 else "↓" if importance < 0 else "→"
        abs_importance = abs(importance)
        print(f"      {i}. {feature:<25} {direction} Impact: {abs_importance:.4f}")

print("\n" + "="*80)

# Global feature importance
print("\n=== Global Feature Importance ===")
if hasattr(model, 'feature_importances_'):
    # Ensure we have the correct number of features after imputation/scaling
    n_features_model = len(model.feature_importances_)
    n_features_expected = len(feature_cols)
    
    print(f"Model features: {n_features_model}, Expected features: {n_features_expected}")
    print(f"Features after imputation: {X_train_scaled.shape[1]}")
    
    # Use the actual number of features from the trained model
    if n_features_model == X_train_scaled.shape[1]:
        # Features might have been dropped during imputation, use only the remaining ones
        active_feature_cols = feature_cols[:n_features_model] if n_features_model <= len(feature_cols) else feature_cols
        
        importance_df = pd.DataFrame({
            'feature': active_feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 15 Most Important Features:")
        print("-" * 50)
        for idx, row in importance_df.head(15).iterrows():
            print(f"{row['feature']:<30} {row['importance']:.4f}")
    else:
        print(f"⚠️  Feature count mismatch: model has {n_features_model} features, expected {n_features_expected}")
        print("Displaying raw feature importances:")
        for i, importance in enumerate(model.feature_importances_[:15]):
            feature_name = feature_cols[i] if i < len(feature_cols) else f"feature_{i}"
            print(f"{feature_name:<30} {importance:.4f}")

# Additional SHAP global importance if available
if shap_available and len(shap_values_class1) > 0:
    print("\n=== SHAP Global Feature Importance ===")
    try:
        feature_importance_shap = np.abs(shap_values_class1).mean(axis=0)
        n_shap_features = len(feature_importance_shap)
        
        # Match SHAP features with our feature names
        if n_shap_features <= len(feature_cols):
            shap_feature_names = feature_cols[:n_shap_features]
        else:
            shap_feature_names = feature_cols + [f"feature_{i}" for i in range(len(feature_cols), n_shap_features)]
        
        shap_importance_df = pd.DataFrame({
            'feature': shap_feature_names,
            'shap_importance': feature_importance_shap
        }).sort_values('shap_importance', ascending=False)
        
        print("\nTop 15 Features by SHAP Values:")
        print("-" * 50)
        for idx, row in shap_importance_df.head(15).iterrows():
            print(f"{row['feature']:<30} {row['shap_importance']:.4f}")
            
    except Exception as e:
        print(f"Could not compute SHAP global importance: {e}")
        print("SHAP values shape:", shap_values_class1.shape if hasattr(shap_values_class1, 'shape') else 'Unknown')

# Save results
output_df = test_with_scores[[TICKER_COL, DATE_COL, 'credit_risk_score', 'target_default', 'health_score']].copy()
output_file = 'enhanced_credit_risk_scores.csv'
output_df.to_csv(output_file, index=False)
print(f"\n📊 Results saved to {output_file}")

# Summary statistics by ticker
print("\n=== Summary Statistics by Ticker ===")
summary_stats = test_with_scores.groupby(TICKER_COL).agg({
    'credit_risk_score': ['mean', 'std', 'min', 'max'],
    'target_default': ['sum', 'mean'],
    'health_score': ['mean', 'std']
}).round(4)

summary_stats.columns = ['Risk_Mean', 'Risk_Std', 'Risk_Min', 'Risk_Max', 
                        'Risk_Periods', 'Risk_Rate', 'Health_Mean', 'Health_Std']
print(summary_stats)

print("\n" + "="*80)
print("=== FINAL SUMMARY ===")
print(f"🔬 Model: {MODEL_BACKEND.upper()}")
print(f"📈 Features Used: {len(feature_cols)}")
print(f"📊 Target Method: Relative Performance Analysis")
print(f"⚠️  Higher-Risk Rate: {default_rate:.1%}")
if test_auc:
    print(f"🎯 Test AUC Score: {test_auc:.4f}")
    performance = "Excellent" if test_auc > 0.8 else "Good" if test_auc > 0.7 else "Fair" if test_auc > 0.6 else "Poor"
    print(f"📊 Model Performance: {performance}")

print("\n✅ Credit Risk Model Training Complete!")
print(f"📁 Results exported to: {output_file}")
print("\n💡 Note: For blue-chip stocks (AAPL, AMZN, etc.), 'higher risk'")
print("   represents periods of relatively weaker performance rather")
print("   than absolute financial distress.")
print("="*80)