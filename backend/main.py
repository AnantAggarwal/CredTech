from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")

app = FastAPI(title="CredTech API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_conn():
    return psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)


@app.get("/")
def root():
    return {"message": "CredTech API v2", "docs": "/docs"}


# ── Track B: updated to include nexscore and grade ───────────────────────────
@app.get("/companies")
def get_companies(limit: int = 100, offset: int = 0):
    """All companies with current credit_score, sentiment_score, nexscore, and grade."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, ticker, name, credit_score, sentiment_score,
                       nexscore, grade
                FROM companies
                ORDER BY ticker
                LIMIT %s OFFSET %s
            """, (limit, offset))
            rows = cur.fetchall()
            
    result = []
    for r in rows:
        d = dict(r)
        d["creditworthiness_display"] = round((1 - float(d["credit_score"])) * 100, 1) if d["credit_score"] is not None else 0
        d["sentiment_display"] = round(((float(d["sentiment_score"]) + 1) / 2) * 100, 1) if d["sentiment_score"] is not None else 0
        result.append(d)
        
    return result


@app.get("/scores")
def get_scores():
    """Latest 20 score records across all companies."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT c.ticker, s.credit_score, s.sentiment_score,
                       s.nexscore, s.grade, s.updated_at
                FROM scores s
                JOIN companies c ON s.company_id = c.id
                ORDER BY s.updated_at DESC
                LIMIT 20
            """)
            rows = cur.fetchall()
    return [dict(r) for r in rows]


@app.get("/company/{ticker}/history")
def get_company_history(ticker: str):
    """Full score history + current info for a single ticker."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, ticker, name, credit_score, sentiment_score,
                       sentiment_summary, nexscore, grade, analyst_note
                FROM companies
                WHERE ticker = %s
            """, (ticker.upper(),))
            company = cur.fetchone()

        if not company:
            raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found")

        with conn.cursor() as cur:
            cur.execute("""
                SELECT credit_score, sentiment_score, nexscore, grade, updated_at
                FROM scores
                WHERE company_id = %s
                ORDER BY updated_at DESC
            """, (company["id"],))
            history = cur.fetchall()

    return {
        "company_info": dict(company),
        "score_history": [dict(h) for h in history],
        "total_records": len(history),
    }


@app.get("/stats")
def get_stats():
    """Aggregate statistics across all companies."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*)                    AS total_companies,
                    AVG(credit_score)           AS avg_credit_score,
                    MIN(credit_score)           AS min_credit_score,
                    MAX(credit_score)           AS max_credit_score,
                    AVG(sentiment_score)        AS avg_sentiment_score,
                    MIN(sentiment_score)        AS min_sentiment_score,
                    MAX(sentiment_score)        AS max_sentiment_score,
                    AVG(nexscore)               AS avg_nexscore,
                    MIN(nexscore)               AS min_nexscore,
                    MAX(nexscore)               AS max_nexscore
                FROM companies
            """)
            row = cur.fetchone()

    stats = dict(row)
    def safe(v):
        return round(float(v), 4) if v is not None else None

    return {
        "total_companies": stats["total_companies"],
        "credit": {
            "raw_avg": safe(stats["avg_credit_score"]),
            "raw_min": safe(stats["min_credit_score"]),
            "raw_max": safe(stats["max_credit_score"]),
            "display_avg": round((1 - float(stats["avg_credit_score"])) * 100, 1)
                           if stats["avg_credit_score"] is not None else None,
        },
        "sentiment": {
            "raw_avg": safe(stats["avg_sentiment_score"]),
            "raw_min": safe(stats["min_sentiment_score"]),
            "raw_max": safe(stats["max_sentiment_score"]),
            "display_avg": round(float(stats["avg_sentiment_score"]) * 100, 1)
                           if stats["avg_sentiment_score"] is not None else None,
        },
        "nexscore": {
            "avg": safe(stats["avg_nexscore"]),
            "min": safe(stats["min_nexscore"]),
            "max": safe(stats["max_nexscore"]),
        },
    }


# ── Track B: new endpoints ────────────────────────────────────────────────────

@app.get("/leaderboard")
def get_leaderboard(limit: int = 100, offset: int = 0):
    """
    All companies ranked by nexscore DESC.
    Returns: rank, ticker, name, nexscore, grade, credit_display, sentiment_display.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    ticker,
                    name,
                    nexscore,
                    grade,
                    credit_score,
                    sentiment_score
                FROM companies
                ORDER BY nexscore DESC NULLS LAST
                LIMIT %s OFFSET %s
            """, (limit, offset))
            rows = cur.fetchall()

    result = []
    for rank, r in enumerate(rows, start=1):
        d = dict(r)
        # creditworthiness: (1 - raw_credit) * 100
        credit_display = round((1 - float(d["credit_score"])) * 100, 1) if d["credit_score"] is not None else None
        # sentiment display: shift from [-1,1] → [0,100]
        sentiment_display = round(((float(d["sentiment_score"]) + 1) / 2) * 100, 1) if d["sentiment_score"] is not None else None
        result.append({
            "rank": rank,
            "ticker": d["ticker"],
            "name": d["name"],
            "nexscore": d["nexscore"],
            "grade": d["grade"],
            "credit_display": credit_display,
            "sentiment_display": sentiment_display,
        })
    return result


@app.get("/company/{ticker}/news")
def get_company_news(ticker: str):
    """
    Top 5 most impactful headlines for a ticker from the most recent pipeline run.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            # Resolve ticker → company_id
            cur.execute(
                "SELECT id FROM companies WHERE ticker = %s",
                (ticker.upper(),)
            )
            company = cur.fetchone()
        if not company:
            raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found")

        company_id = company["id"]

        with conn.cursor() as cur:
            # Get the most recent run_id for this company
            cur.execute("""
                SELECT run_id
                FROM news_headlines
                WHERE company_id = %s
                ORDER BY created_at DESC
                LIMIT 1
            """, (company_id,))
            latest_run = cur.fetchone()

        if not latest_run:
            return {"ticker": ticker.upper(), "headlines": [], "run_id": None}

        run_id = latest_run["run_id"]

        with conn.cursor() as cur:
            cur.execute("""
                SELECT title, source, url, sentiment_label, sentiment_score, created_at
                FROM news_headlines
                WHERE company_id = %s AND run_id = %s
                ORDER BY ABS(sentiment_score) DESC
                LIMIT 5
            """, (company_id, run_id))
            headlines = cur.fetchall()

    return {
        "ticker": ticker.upper(),
        "run_id": run_id,
        "headlines": [dict(h) for h in headlines],
    }


@app.get("/company/{ticker}/nexscore")
def get_company_nexscore(ticker: str):
    """
    Returns nexscore detail for a ticker:
    {ticker, nexscore, grade, analyst_note, credit_display, sentiment_display, updated_at}
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT c.ticker, c.name, c.nexscore, c.grade,
                       c.analyst_note, c.credit_score, c.sentiment_score,
                       s.updated_at
                FROM companies c
                LEFT JOIN scores s ON c.id = s.company_id
                WHERE c.ticker = %s
                  AND (s.id IS NULL OR
                       s.id = (SELECT MAX(id) FROM scores WHERE company_id = c.id))
            """, (ticker.upper(),))
            row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found")

    d = dict(row)
    credit_display = round((1 - float(d["credit_score"])) * 100, 1) if d["credit_score"] is not None else None
    sentiment_display = round(((float(d["sentiment_score"]) + 1) / 2) * 100, 1) if d["sentiment_score"] is not None else None

    return {
        "ticker": d["ticker"],
        "name": d["name"],
        "nexscore": d["nexscore"],
        "grade": d["grade"],
        "analyst_note": d["analyst_note"],
        "credit_display": credit_display,
        "sentiment_display": sentiment_display,
        "updated_at": d["updated_at"].isoformat() if d["updated_at"] else None,
    }
