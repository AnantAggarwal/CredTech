import os
import yfinance as yf
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")

def init_db():
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    with open("db/schema.sql", "r") as f:
        cur.execute(f.read())
    conn.commit()
    cur.close()
    conn.close()

def ingest_companies():
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN"]
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    for t in tickers:
        cur.execute("INSERT INTO companies (ticker, name) VALUES (%s, %s) ON CONFLICT DO NOTHING;", (t, t))
    conn.commit()
    cur.close()
    conn.close()

def compute_scores():
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN"]
    data = {}
    for t in tickers:
        hist = yf.Ticker(t).history(period="1mo")
        if len(hist) > 0:
            ret = (hist["Close"][-1] - hist["Close"][0]) / hist["Close"][0]
            data[t] = float(ret)
        else:
            data[t] = 0.0

    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    for t, s in data.items():
        cur.execute("SELECT id FROM companies WHERE ticker=%s;", (t,))
        cid = cur.fetchone()[0]
        cur.execute("INSERT INTO scores (company_id, score) VALUES (%s, %s);", (cid, s))
    conn.commit()
    cur.close()
    conn.close()

if __name__ == "__main__":
    init_db()
    ingest_companies()
    compute_scores()
    print("✅ Data ingested + scores computed.")
