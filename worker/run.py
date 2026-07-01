import os
import sys
import uuid
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse
from dotenv import load_dotenv
from google import genai

# ── Add project root to path so we can import credit_risk_pipeline ──────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from unstructured import compute_sentiment_score
from credit_risk_pipeline import get_credit_scores

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────
DATABASE_URL  = os.getenv("DATABASE_URL")
NEWS_API_KEY  = os.getenv("NEWS_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PARQUET_GLOB  = str(PROJECT_ROOT / "panel_*.parquet")

print(f"DEBUG: DATABASE_URL loaded: {'Yes' if DATABASE_URL else 'No'}")
print(f"DEBUG: NEWS_API_KEY loaded: {'Yes' if NEWS_API_KEY else 'No'}")
print(f"DEBUG: GEMINI_API_KEY loaded: {'Yes' if GEMINI_API_KEY else 'No'}")
print(f"DEBUG: PARQUET_GLOB = {PARQUET_GLOB}")

# ── Configure Gemini ─────────────────────────────────────────────────────────
if GEMINI_API_KEY:
    _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    _gemini_client = None
    print("WARNING: GEMINI_API_KEY not set — analyst notes will use default text.")


# ── Grade helper ─────────────────────────────────────────────────────────────
def get_grade(nexscore: float) -> str:
    if nexscore >= 90:
        return "AAA"
    elif nexscore >= 80:
        return "AA"
    elif nexscore >= 70:
        return "A"
    elif nexscore >= 60:
        return "BBB"
    elif nexscore >= 50:
        return "BB"
    elif nexscore >= 40:
        return "B"
    else:
        return "CCC"


# ── Gemini analyst note ───────────────────────────────────────────────────────
def generate_analyst_note(company_name: str, ticker: str, nexscore: float,
                          grade: str, creditworthiness: float,
                          sentiment_display: float) -> str:
    """Call Gemini to produce a 2-sentence analyst note. Fails gracefully."""
    if _gemini_client is None:
        return (f"{company_name} ({ticker}) has a NexScore of {nexscore}/100 "
                f"(grade: {grade}), reflecting balanced credit and sentiment signals.")

    prompt = (
        f"You are a credit analyst. Write exactly 2 sentences (max 60 words total) "
        f"summarizing the credit outlook for {company_name} ({ticker}). "
        f"Their NexScore is {nexscore}/100 (grade: {grade}), "
        f"creditworthiness {creditworthiness}/100, and market sentiment {sentiment_display}/100. "
        f"Be specific and professional."
    )
    try:
        response = _gemini_client.models.generate_content(
            model='gemini-1.5-flash', contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"    Gemini call failed for {ticker}: {e}")
        return (f"{company_name} ({ticker}) carries a NexScore of {nexscore}/100 "
                f"(grade: {grade}), driven by creditworthiness of {creditworthiness}/100 "
                f"and market sentiment of {sentiment_display}/100.")


# ── Ticker helpers ────────────────────────────────────────────────────────────
def load_tickers_from_file(file_path="tickers.txt"):
    try:
        with open(file_path, "r") as f:
            tickers = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(tickers)} tickers from {file_path}")
        return tickers
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Using defaults.")
        return ["AAPL", "GOOGL", "AMZN", "BAC", "CVX"]


COMPANY_NAMES = {
    "AAPL": "Apple Inc",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc",
    "AMZN": "Amazon.com Inc",
    "META": "Meta Platforms Inc",
    "TSLA": "Tesla Inc",
    "JPM": "JPMorgan Chase & Co",
    "BAC": "Bank of America Corp",
    "GS": "Goldman Sachs Group Inc",
    "V": "Visa Inc",
    "MA": "Mastercard Inc",
    "XOM": "Exxon Mobil Corporation",
    "CVX": "Chevron Corporation",
    "PG": "Procter & Gamble Co",
    "KO": "Coca-Cola Company",
    "PEP": "PepsiCo Inc",
    "WMT": "Walmart Inc",
    "UNH": "UnitedHealth Group Inc",
    "JNJ": "Johnson & Johnson",
    "NVDA": "NVIDIA Corporation",
}

def create_company_mapping(tickers):
    return {t: COMPANY_NAMES.get(t, t) for t in tickers}


# ── DB helpers ────────────────────────────────────────────────────────────────
def get_db_connection():
    try:
        if not DATABASE_URL:
            print("Error: DATABASE_URL not set")
            return None
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        return conn
    except Exception as e:
        print(f"Error connecting to DB: {e}")
        return None


def apply_schema_migrations(conn):
    """
    Idempotent ALTER TABLE migrations — safe to run on existing data.
    Adds new columns only if they don't already exist.
    """
    migrations = [
        # companies table
        ("ALTER TABLE companies ADD COLUMN IF NOT EXISTS nexscore FLOAT", "companies.nexscore"),
        ("ALTER TABLE companies ADD COLUMN IF NOT EXISTS grade TEXT", "companies.grade"),
        ("ALTER TABLE companies ADD COLUMN IF NOT EXISTS analyst_note TEXT", "companies.analyst_note"),
        ("ALTER TABLE companies ADD COLUMN IF NOT EXISTS sector TEXT", "companies.sector"),
        # scores table
        ("ALTER TABLE scores ADD COLUMN IF NOT EXISTS nexscore FLOAT", "scores.nexscore"),
        ("ALTER TABLE scores ADD COLUMN IF NOT EXISTS grade TEXT", "scores.grade"),
        # news_headlines table
        ("""
            CREATE TABLE IF NOT EXISTS news_headlines (
                id SERIAL PRIMARY KEY,
                company_id INT REFERENCES companies(id),
                run_id TEXT,
                title TEXT,
                source TEXT,
                url TEXT,
                sentiment_label TEXT,
                sentiment_score FLOAT,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """, "news_headlines table"),
    ]
    with conn.cursor() as cur:
        for sql, label in migrations:
            try:
                cur.execute(sql)
                print(f"  Migration OK: {label}")
            except Exception as e:
                conn.rollback()
                print(f"  Migration WARNING ({label}): {e}")
    conn.commit()
    print("Schema migrations complete.")


def recreate_database():
    """Drop and recreate tables (init mode)."""
    print("Recreating database tables...")
    conn = get_db_connection()
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS news_headlines CASCADE")
            cur.execute("DROP TABLE IF EXISTS scores CASCADE")
            cur.execute("DROP TABLE IF EXISTS companies CASCADE")
            cur.execute("DROP TABLE IF EXISTS market_features CASCADE")
            cur.execute("DROP TABLE IF EXISTS pipeline_queue CASCADE")
            cur.execute("""
                CREATE TABLE companies (
                    id SERIAL PRIMARY KEY,
                    ticker TEXT UNIQUE NOT NULL,
                    name TEXT,
                    credit_score FLOAT,
                    sentiment_score FLOAT,
                    sentiment_summary TEXT,
                    nexscore FLOAT,
                    grade TEXT,
                    analyst_note TEXT,
                    sector TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE scores (
                    id SERIAL PRIMARY KEY,
                    company_id INT REFERENCES companies(id),
                    credit_score FLOAT,
                    sentiment_score FLOAT,
                    nexscore FLOAT,
                    grade TEXT,
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE TABLE news_headlines (
                    id SERIAL PRIMARY KEY,
                    company_id INT REFERENCES companies(id),
                    run_id TEXT,
                    title TEXT,
                    source TEXT,
                    url TEXT,
                    sentiment_label TEXT,
                    sentiment_score FLOAT,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE TABLE market_features (
                    id SERIAL PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    date DATE NOT NULL,
                    features JSONB,
                    UNIQUE(ticker, date)
                )
            """)
            cur.execute("""
                CREATE TABLE pipeline_queue (
                    id SERIAL PRIMARY KEY,
                    ticker TEXT UNIQUE NOT NULL,
                    status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            conn.commit()
            
            # Seed the queue
            import pandas as pd
            import requests
            import io
            try:
                url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
                html = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text
                df = pd.read_html(io.StringIO(html))[0]
                tickers = df['Symbol'].tolist()[:300]
                records = [(t,) for t in tickers]
                from psycopg2.extras import execute_batch
                execute_batch(cur, "INSERT INTO pipeline_queue (ticker) VALUES (%s) ON CONFLICT DO NOTHING", records)
                conn.commit()
                print(f"Seeded pipeline_queue with {len(tickers)} companies.")
            except Exception as e:
                print(f"Failed to seed pipeline_queue: {e}")
                
        print("Database recreated successfully.")
        return True
    except Exception as e:
        print(f"Error recreating DB: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


def insert_or_update_company(conn, ticker, name, credit_score, sentiment_score,
                              sentiment_summary, nexscore, grade, analyst_note,
                              is_update=False):
    """
    Insert or update a company row.

    credit_score   : ML default probability (0–1). 0=safe, 1=risky.
    sentiment_score: Raw news sentiment (−1 to 1).
    nexscore       : Composite NexScore (0–100).
    grade          : Letter grade (AAA … CCC).
    analyst_note   : 2-sentence Gemini analyst note.
    """
    try:
        with conn.cursor() as cur:
            if is_update:
                cur.execute(
                    "SELECT id, credit_score, sentiment_score FROM companies WHERE ticker = %s",
                    (ticker,)
                )
                result = cur.fetchone()

                if result:
                    company_id, existing_credit, existing_sentiment = result
                    # Average existing with new
                    new_credit    = (existing_credit + credit_score) / 2 if existing_credit is not None else credit_score
                    new_sentiment = (existing_sentiment + sentiment_score) / 2 if existing_sentiment is not None else sentiment_score
                    print(f"  Update {ticker}: credit {existing_credit:.3f}→{new_credit:.3f}, "
                          f"sentiment {existing_sentiment:.3f}→{new_sentiment:.3f}")
                    cur.execute("""
                        UPDATE companies
                        SET name=%s, credit_score=%s, sentiment_score=%s, sentiment_summary=%s,
                            nexscore=%s, grade=%s, analyst_note=%s
                        WHERE id=%s
                    """, (name, new_credit, new_sentiment, sentiment_summary,
                          nexscore, grade, analyst_note, company_id))
                else:
                    cur.execute("""
                        INSERT INTO companies
                          (ticker, name, credit_score, sentiment_score, sentiment_summary,
                           nexscore, grade, analyst_note)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
                    """, (ticker, name, credit_score, sentiment_score, sentiment_summary,
                          nexscore, grade, analyst_note))
                    company_id = cur.fetchone()[0]
            else:
                # Init mode — fresh insert (tables were just recreated)
                cur.execute("""
                    INSERT INTO companies
                      (ticker, name, credit_score, sentiment_score, sentiment_summary,
                       nexscore, grade, analyst_note)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
                """, (ticker, name, credit_score, sentiment_score, sentiment_summary,
                      nexscore, grade, analyst_note))
                company_id = cur.fetchone()[0]

            conn.commit()
            return company_id

    except Exception as e:
        print(f"Error inserting/updating {ticker}: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        return None


def insert_score_record(conn, company_id, credit_score, sentiment_score,
                        nexscore, grade):
    """Append a timestamped score record for historical tracking."""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO scores (company_id, credit_score, sentiment_score,
                                    nexscore, grade, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (company_id, credit_score, sentiment_score,
                  nexscore, grade, datetime.now()))
            conn.commit()
    except Exception as e:
        print(f"Error inserting score record for company_id {company_id}: {e}")
        try:
            conn.rollback()
        except Exception:
            pass


def insert_news_headlines(conn, company_id, run_id, headlines):
    """Bulk-insert top_headlines list into news_headlines table."""
    if not headlines:
        return
    try:
        with conn.cursor() as cur:
            for h in headlines:
                cur.execute("""
                    INSERT INTO news_headlines
                      (company_id, run_id, title, source, url, sentiment_label, sentiment_score)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    company_id,
                    run_id,
                    h.get('title', ''),
                    h.get('source', ''),
                    h.get('url', ''),
                    h.get('sentiment_label', 'neutral'),
                    h.get('sentiment_score', 0.0),
                ))
        conn.commit()
    except Exception as e:
        print(f"Error inserting headlines for company_id {company_id}: {e}")
        try:
            conn.rollback()
        except Exception:
            pass


# ── Main pipeline ─────────────────────────────────────────────────────────────

def load_tickers_from_queue(conn, batch_size=5):
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, ticker FROM pipeline_queue 
                WHERE status = 'pending' 
                ORDER BY created_at ASC 
                LIMIT %s FOR UPDATE SKIP LOCKED
            """, (batch_size,))
            rows = cur.fetchall()
            
            if not rows:
                return [], []
            
            ids = [r[0] for r in rows]
            tickers = [r[1] for r in rows]
            
            cur.execute("""
                UPDATE pipeline_queue SET status = 'processing', updated_at = NOW() 
                WHERE id = ANY(%s)
            """, (ids,))
            conn.commit()
            return ids, tickers
    except Exception as e:
        print(f"Error fetching from queue: {e}")
        return [], []

def mark_queue_done(conn, ids):
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE pipeline_queue SET status = 'done', updated_at = NOW() 
                WHERE id = ANY(%s)
            """, (ids,))
            conn.commit()
    except Exception as e:
        print(f"Error marking queue done: {e}")

def process_data(start_date=None, end_date=None, is_update=False):
    """
    Full pipeline:
      1. Compute ML credit scores from parquet files
      2. Compute news sentiment scores + top headlines
      3. Compute NexScore, grade, analyst note (Gemini)
      4. Write everything to PostgreSQL
    """
    print("=" * 60)
    print("CredTech — Full Pipeline")
    print("=" * 60)
    print(f"Mode: {'Update (averaging)' if is_update else 'Init (fresh)'}")

    # Unique run identifier so headlines can be grouped by pipeline run
    run_id = str(uuid.uuid4())
    print(f"Run ID: {run_id}")

    # ── Init: recreate tables ─────────────────────────────────────────
    if not is_update:
        if not recreate_database():
            print("Failed to recreate DB. Exiting.")
            return
    else:
        # Update mode: apply migrations to ensure new columns exist
        conn_m = get_db_connection()
        if conn_m:
            apply_schema_migrations(conn_m)
            conn_m.close()

    # ── Step 1: Credit scores (ML) ────────────────────────────────────
    print("\n[Step 1] Running ML credit risk pipeline...")
    credit_scores = {}
    try:
        credit_scores = get_credit_scores(PARQUET_GLOB)
        if credit_scores:
            print(f"  Got credit scores for: {list(credit_scores.keys())}")
        else:
            print("  WARNING: No credit scores returned — parquet files may be missing.")
            print(f"  Run: python worker/build_structured_features.py --tickers-file worker/tickers.txt --years 10 --outdir .")
    except Exception as e:
        print(f"  WARNING: Credit pipeline failed: {e}. Defaulting to 0.5 for all tickers.")

    conn = get_db_connection()
    if not conn:
        print("  Failed to connect to database. Exiting.")
        return

    import time
    while True:
        ids, tickers = load_tickers_from_queue(conn, batch_size=5)
        if not tickers:
            print("No more pending items in pipeline_queue. Exiting loop.")
            break
            
        print(f"\n[Step 2] Running sentiment analysis for batch: {tickers}...")
        tag_to_comp = create_company_mapping(tickers)

        try:
            sentiment_results = compute_sentiment_score(
                tickers, tag_to_comp, NEWS_API_KEY,
                start_date=start_date, end_date=end_date
            )
            print(f"  Processed sentiment for {len(sentiment_results)} companies")
        except Exception as e:
            print(f"  ERROR computing sentiment: {e}")
            # we might want to un-mark them? For now just break
            break

        # ── Step 3: Merge, compute NexScore, get analyst note, write to DB ─
        print("\n[Step 3] Computing NexScore + Gemini analyst notes + writing to DB...")
        try:
            for _, row in sentiment_results.iterrows():
                ticker           = row["ticker"]
                company_name     = row["company_name"]
                sentiment_score  = float(row["sentiment_score"])
                sentiment_summary = row.get("sentiment_summary", "")
                top_headlines    = row.get("top_headlines", [])
                if not isinstance(top_headlines, list):
                    top_headlines = []

                # ML credit score (default probability). Fall back to 0.5 (neutral) if unavailable.
                raw_credit_score = credit_scores.get(ticker, 0.5)

                # Display-scale values (0–100)
                creditworthiness   = round((1 - raw_credit_score) * 100, 1)
                # Sentiment raw is in [-1, 1]; shift to [0, 100] display scale
                sentiment_display  = round(((sentiment_score + 1) / 2) * 100, 1)

                # ── NexScore ──────────────────────────────────────────────
                nexscore = round(0.6 * creditworthiness + 0.4 * sentiment_display, 1)
                grade    = get_grade(nexscore)

                print(f"\n  {ticker} ({company_name})")
                print(f"    credit_score (default prob) = {raw_credit_score:.4f}  "
                      f"→ creditworthiness = {creditworthiness}/100")
                print(f"    sentiment_score (raw)        = {sentiment_score:.4f}  "
                      f"→ display = {sentiment_display}/100")
                print(f"    NexScore = {nexscore}/100  Grade = {grade}")

                # ── Gemini analyst note ───────────────────────────────────
                print(f"    Generating Gemini analyst note...")
                analyst_note = generate_analyst_note(
                    company_name, ticker, nexscore, grade,
                    creditworthiness, sentiment_display
                )
                print(f"    Analyst note: {analyst_note[:80]}...")

                # ── Write company + score ─────────────────────────────────
                company_id = insert_or_update_company(
                    conn, ticker, company_name,
                    raw_credit_score, sentiment_score, sentiment_summary,
                    nexscore, grade, analyst_note,
                    is_update
                )

                if company_id:
                    insert_score_record(conn, company_id, raw_credit_score,
                                        sentiment_score, nexscore, grade)
                    insert_news_headlines(conn, company_id, run_id, top_headlines)
                    print(f"    ✓ Saved to DB (company_id={company_id}, "
                          f"{len(top_headlines)} headlines)")
                else:
                    print(f"    ✗ Failed to save {ticker}")

            # mark done
            mark_queue_done(conn, ids)
            print("\n✅ Batch complete.")
            
            # rate limit handling
            print("Sleeping for 60 seconds to respect Gemini 15 RPM limits...")
            time.sleep(60)

        except Exception as e:
            print(f"Error in DB write phase: {e}")
            import traceback; traceback.print_exc()
            try:
                conn.rollback()
            except Exception:
                pass
                
    try:
        conn.close()
    except Exception:
        pass


def get_latest_scores():
    """Retrieve latest scores from DB for display."""
    conn = get_db_connection()
    if not conn:
        return None
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT c.ticker, c.name,
                       c.credit_score, c.sentiment_score,
                       c.nexscore, c.grade,
                       s.updated_at
                FROM companies c
                LEFT JOIN scores s ON c.id = s.company_id
                WHERE s.id = (SELECT MAX(id) FROM scores WHERE company_id = c.id)
                ORDER BY c.nexscore DESC NULLS LAST
            """)
            results = cur.fetchall()
            return pd.DataFrame(results)
    except Exception as e:
        print(f"Error retrieving scores: {e}")
        return None
    finally:
        conn.close()


def parse_date(date_str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CredTech full pipeline runner")
    parser.add_argument("--start-date", type=parse_date,
                        help="Start date for news collection (YYYY-MM-DD)")
    parser.add_argument("--end-date",   type=parse_date,
                        help="End date for news collection (YYYY-MM-DD)")
    parser.add_argument("--mode", choices=["init", "update"], default="init",
                        help="init: wipe DB and reload. update: average with existing scores.")
    args = parser.parse_args()

    if args.start_date and args.end_date:
        if args.start_date >= args.end_date:
            print("Error: start-date must be before end-date")
            exit(1)

    process_data(args.start_date, args.end_date, args.mode == "update")

    print("\n── Latest scores in DB ──")
    latest = get_latest_scores()
    if latest is not None and not latest.empty:
        latest["creditworthiness"] = ((1 - latest["credit_score"]) * 100).round(1)
        latest["sentiment_display"] = (latest["sentiment_score"] * 100).round(1)
        print(latest[["ticker", "nexscore", "grade", "creditworthiness",
                       "sentiment_display", "updated_at"]].to_string(index=False))
    else:
        print("No scores found in DB.")
