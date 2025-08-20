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
