CREATE TABLE IF NOT EXISTS companies (
    id SERIAL PRIMARY KEY,
    ticker TEXT UNIQUE NOT NULL,
    name TEXT
);

CREATE TABLE IF NOT EXISTS scores (
    id SERIAL PRIMARY KEY,
    company_id INT REFERENCES companies(id),
    score FLOAT,
    updated_at TIMESTAMP DEFAULT NOW()
);