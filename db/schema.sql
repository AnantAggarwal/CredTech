CREATE TABLE IF NOT EXISTS companies (
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
);

CREATE TABLE IF NOT EXISTS scores (
    id SERIAL PRIMARY KEY,
    company_id INT REFERENCES companies(id),
    credit_score FLOAT,
    sentiment_score FLOAT,
    nexscore FLOAT,
    grade TEXT,
    updated_at TIMESTAMP DEFAULT NOW()
);

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
);
CREATE TABLE IF NOT EXISTS market_features (
    id SERIAL PRIMARY KEY,
    ticker TEXT NOT NULL,
    date DATE NOT NULL,
    features JSONB,
    UNIQUE(ticker, date)
);

CREATE TABLE IF NOT EXISTS pipeline_queue (
    id SERIAL PRIMARY KEY,
    ticker TEXT UNIQUE NOT NULL,
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
