
#!/usr/bin/env python3
"""
CredTech Hackathon — Structured Data Pipeline (MVP)
Author: You :)

What this does (end-to-end):
1) Fetch structured data
   - Fundamentals (quarterly) from yfinance
   - Market data (daily OHLCV) from yfinance
   - Macro (monthly/weekly) from FRED (via pandas_datareader) — optional, will skip gracefully if unavailable
2) Engineer features
   - Financial ratios (liquidity, leverage, profitability, efficiency)
   - Growth metrics (QoQ, YoY where available)
   - Market signals (returns, rolling volatility)
   - Macro overlays (policy rate, CPI YoY), forward-filled to daily
3) Align to a daily panel (issuer × date × features) using a date spine and forward-filling
4) Save outputs to parquet/csv

Run:
  python build_structured_features.py --tickers AAPL MSFT GOOGL JPM --years 10 --outdir data_out
  # or
  python build_structured_features.py --tickers-file tickers.txt --years 10 --outdir data_out

Notes:
- Keep it simple & robust. Missing sources won't crash the pipeline.
- You can expand features easily in the dedicated sections below.
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

# Macro: optional
try:
    from pandas_datareader import data as pdr
    _HAS_FRED = True
except Exception:
    _HAS_FRED = False


# -----------------------------
# Utils
# -----------------------------

def log(msg: str):
    print(f"[pipeline] {msg}", flush=True)


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lower().strip().replace(" ", "_") for c in df.columns]
    return df


def safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    b = b.replace(0, np.nan)
    out = a / b
    return out.replace([np.inf, -np.inf], np.nan)


def make_daily_spine(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    # Changed from 'D' (Daily) to 'W-FRI' (Weekly) to reduce database size by 85% and prevent 512MB OOM
    return pd.date_range(start=start, end=end, freq="W-FRI")


def pct_change_safe(s: pd.Series, periods: int = 1) -> pd.Series:
    return s.pct_change(periods=periods).replace([np.inf, -np.inf], np.nan)


# -----------------------------
# Ingestion
# -----------------------------

def fetch_fundamentals_quarterly(ticker: str) -> Optional[pd.DataFrame]:
    """
    Returns a quarterly fundamentals dataframe (index=quarter end date).
    Columns: normalized.
    """
    try:
        t = yf.Ticker(ticker)
        fin = t.quarterly_financials.T
        bs = t.quarterly_balance_sheet.T
        cf = t.quarterly_cashflow.T
        if fin.empty and bs.empty and cf.empty:
            log(f"{ticker}: No quarterly fundamentals found.")
            return None

        # Align by union of indices
        idx = fin.index.union(bs.index).union(cf.index).sort_values()
        fin = fin.reindex(idx)
        bs = bs.reindex(idx)
        cf = cf.reindex(idx)

        df = pd.concat({"fin": fin, "bs": bs, "cf": cf}, axis=1)
        # Flatten MultiIndex columns if any
        df.columns = ["__".join([c for c in col if c]) if isinstance(col, tuple) else str(col) for col in df.columns]
        df = normalize_cols(df).sort_index()
        return df
    except Exception as e:
        log(f"{ticker}: fundamentals fetch error: {e}")
        return None


def fetch_market_daily(ticker: str, years: int = 10) -> Optional[pd.DataFrame]:
    """
    Returns daily OHLCV as DataFrame with columns: Open, High, Low, Close, Adj Close, Volume
    """
    try:
        period = f"{years}y"
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False)
        if df is None or df.empty:
            log(f"{ticker}: market data empty.")
            return None
        df = df.rename(columns=str.lower)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        log(f"{ticker}: market fetch error: {e}")
        return None


def fetch_macro_series(series_codes: List[str], start: str = "2000-01-01") -> Optional[pd.DataFrame]:
    """
    Fetch macro time series from FRED (via pandas_datareader).
    If not available or offline, returns None.
    """
    if not _HAS_FRED:
        log("FRED not available (pandas_datareader not installed). Skipping macro.")
        return None

    out = {}
    for code in series_codes:
        try:
            s = pdr.DataReader(code, "fred", start)
            s = s.rename(columns={code: code})
            out[code] = s[code]
        except Exception as e:
            log(f"Macro fetch failed for {code}: {e}")
    if not out:
        return None
    df = pd.concat(out, axis=1)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


# -----------------------------
# Feature Engineering
# -----------------------------

def first_existing(df: pd.DataFrame, candidates: list):
    for col in candidates:
        if col in df.columns:
            return df[col]
    return None

def engineer_fundamental_ratios(f: pd.DataFrame) -> pd.DataFrame:
    df = f.copy()

    # Balance sheet
    ca  = first_existing(df, ["bs__current_assets", "current_assets"])
    cl  = first_existing(df, ["bs__current_liabilities", "current_liabilities"])
    tl  = first_existing(df, ["bs__total_liab", "total_liab"])
    ta  = first_existing(df, ["bs__total_assets", "total_assets"])
    tse = first_existing(df, [
        "bs__total_stockholder_equity",
        "bs__stockholders_equity",
        "bs__total_equity_gross_minority_interest",
        "total_stockholder_equity",
        "stockholders_equity",
        "total_equity_gross_minority_interest",
    ])

    # Income statement
    ni = first_existing(df, ["fin__net_income", "net_income"])
    rev = first_existing(df, ["fin__total_revenue", "total_revenue"])
    ebit = first_existing(df, ["fin__ebit", "ebit"])
    interest_exp = first_existing(df, [
        "fin__interest_expense",
        "interest_expense",
        "fin__interest_expense_non_operating",
        "interest_expense_non_operating"
    ])

    # Cash flow
    ocf = first_existing(df, [
        "cf__total_cash_from_operating_activities",
        "total_cash_from_operating_activities"
    ])

    # Ratios & growth
    if ca is not None and cl is not None:
        df["rat_current_ratio"] = safe_divide(ca, cl)
    if tl is not None and tse is not None:
        df["rat_debt_to_equity"] = safe_divide(tl, tse)
    if ni is not None and rev is not None:
        df["rat_net_profit_margin"] = safe_divide(ni, rev)
    if ni is not None and ta is not None:
        df["rat_roa"] = safe_divide(ni, ta)
    if rev is not None and ta is not None:
        df["rat_asset_turnover"] = safe_divide(rev, ta)
    if ebit is not None and interest_exp is not None:
        df["rat_interest_coverage"] = safe_divide(ebit, interest_exp)
    if ocf is not None and tl is not None:
        df["rat_ocf_to_debt"] = safe_divide(ocf, tl)

    # Growth
    if rev is not None:
        df["g_qoq_revenue"] = pct_change_safe(rev, periods=1)
        df["g_yoy_revenue"] = pct_change_safe(rev, periods=4)
    if ni is not None:
        df["g_qoq_net_income"] = pct_change_safe(ni, periods=1)
        df["g_yoy_net_income"] = pct_change_safe(ni, periods=4)
    if tl is not None:
        df["g_qoq_total_debt"] = pct_change_safe(tl, periods=1)
        df["g_yoy_total_debt"] = pct_change_safe(tl, periods=4)

    return df

def engineer_market_features(mkt: pd.DataFrame) -> pd.DataFrame:
    df = mkt.copy()
    # Basic returns
    df["ret_1d"] = df["adj close"].pct_change()
    # Cumulative windowed returns
    for w in (5, 21, 63, 126, 252):
        df[f"ret_{w}d"] = df["adj close"].pct_change(periods=w)
    # Rolling volatility (close-to-close)
    for w in (21, 63, 126):
        df[f"vol_{w}d"] = df["ret_1d"].rolling(w).std() * np.sqrt(252)
    return df


def engineer_macro_features(macro: pd.DataFrame) -> pd.DataFrame:
    if macro is None or macro.empty:
        return macro
    df = macro.copy()
    # Example transforms: YoY for CPI, diffs for rate
    for col in df.columns:
        if "CPI" in col.upper():
            df[f"{col}_yoy"] = df[col].pct_change(periods=12)
        if "RATE" in col.upper() or "FEDFUNDS" in col.upper():
            df[f"{col}_chg_1m"] = df[col].diff(1)
    return df


# -----------------------------
# Merge / Panel builder
# -----------------------------

def build_daily_panel(ticker: str, years: int, macro_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    fund = fetch_fundamentals_quarterly(ticker)
    mkt = fetch_market_daily(ticker, years=years)

    if mkt is None:
        log(f"{ticker}: skipping due to missing market data.")
        return None

    # Engineer features
    if fund is not None:
        fund_feat = engineer_fundamental_ratios(fund)
    else:
        fund_feat = None
        log(f"{ticker}: fundamentals missing, continuing with market-only.")

    mkt_feat = engineer_market_features(mkt)

    # Date spine
    start = (mkt_feat.index.min() if mkt_feat is not None else None) or pd.Timestamp.today() - pd.Timedelta(days=365*years)
    end = (mkt_feat.index.max() if mkt_feat is not None else None) or pd.Timestamp.today()
    spine = pd.Index(make_daily_spine(start, end), name="date")

    # Reindex to daily
    if fund_feat is not None:
        # fundamentals are quarterly; upsample to daily with ffill
        fund_daily = fund_feat.reindex(spine).ffill()
    else:
        fund_daily = None

    if mkt_feat is not None:
        mkt_daily = mkt_feat.reindex(spine).ffill()
    else:
        mkt_daily = None

    # Macro (optional)
    if macro_df is not None and not macro_df.empty:
        macro_daily = macro_df.reindex(spine).ffill()
    else:
        macro_daily = None

    # Merge
    pieces = []
    if mkt_daily is not None:
        pieces.append(mkt_daily.add_prefix("mkt__"))
    if fund_daily is not None:
        pieces.append(fund_daily.add_prefix("fin__"))
    if macro_daily is not None:
        pieces.append(macro_daily.add_prefix("mac__"))

    if not pieces:
        return None

    panel = pd.concat(pieces, axis=1)
    panel["ticker"] = ticker
    panel.index.name = "date"
    return panel.reset_index()


# -----------------------------
# Main
# -----------------------------


def get_top_300_tickers():
    try:
        import pandas as pd
        import requests
        import io
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        html = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text
        df = pd.read_html(io.StringIO(html))[0]
        tickers = df['Symbol'].tolist()
        return tickers[:300]
    except Exception as e:
        log(f"Failed to fetch S&P 500: {e}")
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

def parse_args():
    ap = argparse.ArgumentParser(description="Build structured features panel (issuer × date)")
    ap.add_argument("--tickers", nargs="*", help="List of tickers (space-separated).", default=[])
    ap.add_argument("--tickers-file", type=str, help="Path to a text file with one ticker per line.", default=None)
    ap.add_argument("--years", type=int, default=10, help="How many years of daily market data to fetch.")
    ap.add_argument("--outdir", type=str, default="data_out", help="Output directory.")
    ap.add_argument("--macro", nargs="*", default=["FEDFUNDS", "CPIAUCSL"], help="FRED series codes to fetch.")
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Resolve tickers (Top 300)
    tickers = get_top_300_tickers()
    if args.tickers:
        tickers = args.tickers
    tickers = sorted(set([t.upper() for t in tickers]))

    # Fetch macro (optional)
    macro_df = None
    if args.macro:
        try:
            macro_df = fetch_macro_series(args.macro, start="2000-01-01")
            macro_df = engineer_macro_features(macro_df)
            if macro_df is None or macro_df.empty:
                log("Macro empty; continuing without macro.")
        except Exception as e:
            log(f"Macro fetch error (skipping): {e}")
            macro_df = None

    import json
    import psycopg2
    from psycopg2.extras import execute_batch
    from dotenv import load_dotenv
    load_dotenv()
    db_url = os.environ.get("DATABASE_URL")

    conn = None
    if db_url:
        try:
            conn = psycopg2.connect(db_url)
        except Exception as e:
            log(f"Failed to connect to PostgreSQL: {e}")

    # Build and save per-ticker panels immediately to save RAM
    for t in tickers:
        log(f"Building panel for {t} ...")
        panel_t = build_daily_panel(t, years=args.years, macro_df=macro_df)
        if panel_t is not None:
            # Flatten columns
            panel_t.columns = [str(c) for c in panel_t.columns]

            # Light post-processing
            for col in [c for c in panel_t.columns if str(c).startswith("fin__rat_")]:
                panel_t[col] = panel_t[col].clip(lower=-1000, upper=1000)
            
            # Save to PostgreSQL
            if conn is not None:
                try:
                    records = []
                    for _, row in panel_t.iterrows():
                        d = row['date']
                        feat_dict = row.drop(['ticker', 'date']).to_dict()
                        feat_dict = {k: (None if pd.isna(v) else v) for k, v in feat_dict.items()}
                        records.append((t, d.strftime('%Y-%m-%d'), json.dumps(feat_dict)))
                    
                    with conn.cursor() as cur:
                        execute_batch(cur, """
                            INSERT INTO market_features (ticker, date, features)
                            VALUES (%s, %s, %s)
                            ON CONFLICT (ticker, date) DO UPDATE 
                            SET features = EXCLUDED.features;
                        """, records)
                    conn.commit()
                    log(f"✅ Saved {t} to PostgreSQL.")
                except Exception as e:
                    log(f"Failed to save {t} to PostgreSQL: {e}")
                    conn.rollback()

            # Free memory immediately
            del panel_t

    if conn is not None:
        conn.close()
    log("Done.")

if __name__ == "__main__":
    main()
