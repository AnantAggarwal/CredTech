CREATE TABLE IF NOT EXISTS companies (
    id SERIAL PRIMARY KEY,
    ticker TEXT UNIQUE NOT NULL,
    name TEXT,
    credit_score FLOAT,
    sentiment_score FLOAT,
    sentiment_summary TEXT
);

CREATE TABLE IF NOT EXISTS scores (
    id SERIAL PRIMARY KEY,
    company_id INT REFERENCES companies(id),
    credit_score FLOAT,
    sentiment_score FLOAT,
    updated_at TIMESTAMP DEFAULT NOW()
);