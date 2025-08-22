from fastapi import FastAPI
import os, psycopg2
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")

app = FastAPI()

@app.get("/scores")
def get_scores():
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute("""
        SELECT c.ticker, s.score, s.updated_at
        FROM scores s
        JOIN companies c ON s.company_id = c.id
        ORDER BY s.updated_at DESC
        LIMIT 20;
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [{"ticker": r[0], "score": r[1], "updated_at": str(r[2])} for r in rows]

@app.get("/companies")
def get_companies():
    """Get all companies data except sentiment_summary"""
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute("""
        SELECT id, ticker, name, credit_score, sentiment_score
        FROM companies
        ORDER BY ticker;
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [{"id": r[0], "ticker": r[1], "name": r[2], "credit_score": r[3], "sentiment_score": r[4]} for r in rows]

@app.get("/company/{ticker}/history")
def get_company_history(ticker: str):
    """Get complete score history, sentiment score history, and sentiment summary for a specific company"""
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    
    # Get company info and sentiment summary
    cur.execute("""
        SELECT id, ticker, name, credit_score, sentiment_score, sentiment_summary
        FROM companies
        WHERE ticker = %s;
    """, (ticker.upper(),))
    
    company_result = cur.fetchone()
    if not company_result:
        cur.close()
        conn.close()
        return {"error": f"Company with ticker {ticker} not found"}
    
    company_id, company_ticker, company_name, current_credit_score, current_sentiment_score, sentiment_summary = company_result
    
    # Get complete score history
    cur.execute("""
        SELECT credit_score, sentiment_score, updated_at
        FROM scores
        WHERE company_id = %s
        ORDER BY updated_at DESC;
    """, (company_id,))
    
    score_history = cur.fetchall()
    cur.close()
    conn.close()
    
    # Format the response
    score_history_formatted = [
        {
            "credit_score": row[0],
            "sentiment_score": row[1],
            "updated_at": str(row[2])
        }
        for row in score_history
    ]
    
    return {
        "company_info": {
            "id": company_id,
            "ticker": company_ticker,
            "name": company_name,
            "current_credit_score": current_credit_score,
            "current_sentiment_score": current_sentiment_score,
            "sentiment_summary": sentiment_summary
        },
        "score_history": score_history_formatted,
        "total_records": len(score_history_formatted)
    }
