import os
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
from unstructured import compute_sentiment_score
from datetime import datetime, timedelta
import argparse
from dotenv import load_dotenv

load_dotenv()

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL")

# API Keys
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

def load_tickers_from_file(file_path='tickers.txt'):
    """Load tickers from tickers.txt file"""
    try:
        with open(file_path, 'r') as file:
            tickers = [line.strip() for line in file if line.strip()]
        print(f"Loaded {len(tickers)} tickers from {file_path}")
        return tickers
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Using default tickers.")
        return ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']

def create_company_mapping(tickers):
    """Create a mapping from ticker to company name"""
    # Common company name mappings
    company_names = {
        'AAPL': 'Apple Inc',
        'MSFT': 'Microsoft Corporation',
        'GOOGL': 'Alphabet Inc',
        'AMZN': 'Amazon.com Inc',
        'META': 'Meta Platforms Inc',
        'TSLA': 'Tesla Inc',
        'JPM': 'JPMorgan Chase & Co',
        'BAC': 'Bank of America Corp',
        'GS': 'Goldman Sachs Group Inc',
        'V': 'Visa Inc',
        'MA': 'Mastercard Inc',
        'XOM': 'Exxon Mobil Corporation',
        'CVX': 'Chevron Corporation',
        'PG': 'Procter & Gamble Co',
        'KO': 'Coca-Cola Company',
        'PEP': 'PepsiCo Inc',
        'WMT': 'Walmart Inc',
        'UNH': 'UnitedHealth Group Inc',
        'JNJ': 'Johnson & Johnson',
        'NVDA': 'NVIDIA Corporation'
    }
    
    # Create mapping for loaded tickers
    tag_to_comp = {}
    for ticker in tickers:
        if ticker in company_names:
            tag_to_comp[ticker] = company_names[ticker]
        else:
            # Use ticker as company name if not found in mapping
            tag_to_comp[ticker] = ticker
            print(f"Warning: No company name mapping found for {ticker}")
    
    return tag_to_comp

def get_db_connection():
    """Create and return database connection"""
    try:
        if not DATABASE_URL:
            print("Error: DATABASE_URL environment variable not set")
            return None
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def insert_or_update_company(conn, ticker, name, sentiment_score, sentiment_summary, is_update=False):
    """Insert or update company record in companies table"""
    try:
        with conn.cursor() as cursor:
            # Check if company exists
            cursor.execute("SELECT id, sentiment_score FROM companies WHERE ticker = %s", (ticker,))
            result = cursor.fetchone()
            
            if result:
                company_id, existing_score = result
                if is_update and existing_score is not None:
                    # Calculate average of existing and new score
                    new_average_score = (existing_score + sentiment_score) / 2
                    print(f"Updating {ticker}: existing score {existing_score:.3f}, new score {sentiment_score:.3f}, average {new_average_score:.3f}")
                    sentiment_score = new_average_score
                
                # Update existing company
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

def process_sentiment_data(start_date=None, end_date=None, is_update=False):
    """Main function to process sentiment data and update database"""
    print("Starting sentiment analysis...")
    
    if start_date and end_date:
        print(f"Processing data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    else:
        print("Processing recent data (no date range specified)")
    
    print(f"Database mode: {'Update (averaging scores)' if is_update else 'Initialize (replacing scores)'}")
    
    # Load tickers and create company mapping
    tickers = load_tickers_from_file()
    tag_to_comp = create_company_mapping(tickers)
    
    # Get database connection
    conn = get_db_connection()
    if not conn:
        print("Failed to connect to database. Exiting.")
        return
    
    try:
        # Compute sentiment scores for all tickers
        print("Computing sentiment scores...")
        sentiment_results = compute_sentiment_score(
            tickers, tag_to_comp, NEWS_API_KEY, 
            start_date=start_date, end_date=end_date
        )
        
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
                conn, ticker, company_name, sentiment_score, sentiment_summary, is_update
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

def parse_date(date_str):
    """Parse date string in YYYY-MM-DD format"""
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Process sentiment data for companies')
    parser.add_argument('--start-date', type=parse_date, 
                       help='Start date for data collection (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=parse_date, 
                       help='End date for data collection (YYYY-MM-DD)')
    parser.add_argument('--mode', choices=['init', 'update'], default='init',
                       help='Database mode: init (replace scores) or update (average scores)')
    
    args = parser.parse_args()
    
    # Validate date range
    if args.start_date and args.end_date:
        if args.start_date >= args.end_date:
            print("Error: Start date must be before end date")
            exit(1)
    
    # Convert mode to boolean
    is_update = args.mode == 'update'
    
    # Process sentiment data and update database
    process_sentiment_data(args.start_date, args.end_date, is_update)
    
    # Display latest scores
    print("\nLatest sentiment scores:")
    latest_scores = get_latest_scores()
    if latest_scores is not None:
        print(latest_scores.to_string(index=False))
    else:
        print("Failed to retrieve latest scores")
