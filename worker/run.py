import os
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
from unstructured import compute_sentiment_score
from datetime import datetime

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'database': os.getenv('DB_NAME', 'credtech'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', ''),
    'port': os.getenv('DB_PORT', '5432')
}

# API Keys
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")

# Sample company data - replace with your actual data
TICKERS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
TAG_TO_COMP = {
    'AAPL': 'Apple Inc',
    'GOOGL': 'Alphabet Inc',
    'MSFT': 'Microsoft Corporation',
    'AMZN': 'Amazon.com Inc',
    'TSLA': 'Tesla Inc'
}

def get_db_connection():
    """Create and return database connection"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def insert_or_update_company(conn, ticker, name, sentiment_score, sentiment_summary):
    """Insert or update company record in companies table"""
    try:
        with conn.cursor() as cursor:
            # Check if company exists
            cursor.execute("SELECT id FROM companies WHERE ticker = %s", (ticker,))
            result = cursor.fetchone()
            
            if result:
                # Update existing company
                company_id = result[0]
                cursor.execute("""
                    UPDATE companies 
                    SET name = %s, credit_score = %s, sentiment_score = %s, sentiment_summary = %s
                    WHERE id = %s
                """, (name, sentiment_score, sentiment_score, sentiment_summary, company_id))
            else:
                # Insert new company
                cursor.execute("""
                    INSERT INTO companies (ticker, name, credit_score, sentiment_score, sentiment_summary)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                """, (ticker, name, sentiment_score, sentiment_score, sentiment_summary))
                company_id = cursor.fetchone()[0]
            
            conn.commit()
            return company_id
            
    except Exception as e:
        print(f"Error inserting/updating company {ticker}: {e}")
        conn.rollback()
        return None

def insert_score_record(conn, company_id, sentiment_score):
    """Insert new score record in scores table"""
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO scores (company_id, credit_score, sentiment_score, updated_at)
                VALUES (%s, %s, %s, %s)
            """, (company_id, sentiment_score, sentiment_score, datetime.now()))
            
            conn.commit()
            
    except Exception as e:
        print(f"Error inserting score record for company_id {company_id}: {e}")
        conn.rollback()

def process_sentiment_data():
    """Main function to process sentiment data and update database"""
    print("Starting sentiment analysis...")
    
    # Get database connection
    conn = get_db_connection()
    if not conn:
        print("Failed to connect to database. Exiting.")
        return
    
    try:
        # Compute sentiment scores for all tickers
        print("Computing sentiment scores...")
        sentiment_results = compute_sentiment_score(TICKERS, TAG_TO_COMP, NEWS_API_KEY, TWITTER_BEARER_TOKEN)
        
        print(f"Processed {len(sentiment_results)} companies")
        
        # Process each company's results
        for _, row in sentiment_results.iterrows():
            ticker = row['ticker']
            company_name = row['company_name']
            sentiment_score = row['sentiment_score']
            sentiment_summary = row['sentiment_summary']
            
            print(f"Processing {ticker} ({company_name}) - Sentiment Score: {sentiment_score:.3f}")
            
            # Insert/update company record
            company_id = insert_or_update_company(
                conn, ticker, company_name, sentiment_score, sentiment_summary
            )
            
            if company_id:
                # Insert score record
                insert_score_record(conn, company_id, sentiment_score)
                print(f"Successfully updated database for {ticker}")
            else:
                print(f"Failed to update database for {ticker}")
        
        print("Sentiment analysis and database update completed successfully!")
        
    except Exception as e:
        print(f"Error in sentiment processing: {e}")
        conn.rollback()
    
    finally:
        conn.close()

def get_latest_scores():
    """Retrieve latest sentiment scores from database"""
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT c.ticker, c.name, c.sentiment_score, c.sentiment_summary, s.updated_at
                FROM companies c
                LEFT JOIN scores s ON c.id = s.company_id
                WHERE s.id = (
                    SELECT MAX(id) FROM scores WHERE company_id = c.id
                )
                ORDER BY c.ticker
            """)
            
            results = cursor.fetchall()
            return pd.DataFrame(results)
            
    except Exception as e:
        print(f"Error retrieving scores: {e}")
        return None
    finally:
        conn.close()

if __name__ == "__main__":
    # Process sentiment data and update database
    process_sentiment_data()
    
    # Display latest scores
    print("\nLatest sentiment scores:")
    latest_scores = get_latest_scores()
    if latest_scores is not None:
        print(latest_scores.to_string(index=False))
    else:
        print("Failed to retrieve latest scores")
