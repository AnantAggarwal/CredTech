import os
import requests
import pandas as pd
from transformers import pipeline
from google import genai
from datetime import datetime, timedelta

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

def get_company_news(ticker, tag_to_comp, NEWS_API_KEY, start_date=None, end_date=None):
    query = f'"{tag_to_comp[ticker]}" OR {ticker}'
    print(f"Searching for news with query: {query}")
    
    news_data = []
    try:
        # Add date range to query if provided
        if start_date and end_date:
            from_str = start_date.strftime('%Y-%m-%d')
            to_str = end_date.strftime('%Y-%m-%d')
            url = (f'https://newsapi.org/v2/everything?'
                   f'q={query}&'
                   f'language=en&'
                   f'sortBy=publishedAt&'
                   f'from={from_str}&'
                   f'to={to_str}&'
                   f'apiKey={NEWS_API_KEY}')
            print(f"Using date range: {from_str} to {to_str}")
        else:
            url = (f'https://newsapi.org/v2/everything?'
                   f'q={query}&'
                   f'language=en&'
                   f'sortBy=publishedAt&'
                   f'apiKey={NEWS_API_KEY}')
            print("No date range specified, using recent articles")

        print(f"Making request to News API...")
        response = requests.get(url)
        
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', [])
            
            for i, article in enumerate(articles):
                if not article.get('title') or not article.get('description'):
                    continue
                news_data.append({
                    'source': article['source']['name'] if article.get('source') else 'Unknown',
                    'title': article['title'],
                    'description': article['description'],
                    'url': article.get('url', ''),
                    'published_at': article.get('publishedAt', ''),
                    'text': article['title'] + '\n' + article['description']
                })
        else:
            print(f"NewsAPI error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"NewsAPI exception: {e}")
        
    if not news_data:
        print(f"Falling back to yfinance for {ticker} news...")
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            y_news = t.news
            if y_news:
                for article in y_news:
                    title = article.get('title', '')
                    link = article.get('link', '')
                    publisher = article.get('publisher', 'Unknown')
                    summary = article.get('summary', title)
                    news_data.append({
                        'source': publisher,
                        'title': title,
                        'description': summary,
                        'url': link,
                        'published_at': str(article.get('providerPublishTime', '')),
                        'text': title + '\n' + summary
                    })
        except Exception as e:
            print(f"yfinance fallback failed: {e}")

    print(f"Processed {len(news_data)} valid articles")
    return pd.DataFrame(news_data)

def label_to_numeric(row):
    """
    Convert model label + confidence score to a numeric value in [-1, 1].
    The model (rahilv/news-sentiment-analysis-roberta) outputs:
      'bullish'  → positive sentiment
      'bearish'  → negative sentiment
      'neutral'  → no signal
    Handles any casing variation defensively.
    """
    label = row['sentiment_label'].lower().strip()
    score = row['sentiment_score']   # model confidence [0, 1]
    if label in ('bullish', 'positive'):
        return score
    elif label in ('bearish', 'negative'):
        return -score
    else:   # neutral or unknown
        return 0.0

def analyze_sentiment_dataframe(df, text_column, pipeline):
    """Analyze sentiment for a dataframe and return processed results"""
    if df.empty:
        return df
    
    # Get sentiment analysis
    sentiments = pipeline(df[text_column].to_list())
    
    # Add sentiment columns
    df['sentiment'] = sentiments
    df['sentiment_label'] = df['sentiment'].apply(lambda x: x['label'])
    df['sentiment_score'] = df['sentiment'].apply(lambda x: x['score'])
    df['numeric_sentiment'] = df.apply(label_to_numeric, axis=1)
    
    return df

def get_extreme_content(df, column, n=5):
    """Get top and bottom n items from dataframe based on numeric_sentiment"""
    if df.empty:
        return [], []
    
    top_items = df.nlargest(n, 'numeric_sentiment')[column].tolist()
    bottom_items = df.nsmallest(n, 'numeric_sentiment')[column].tolist()
    
    return top_items, bottom_items

def get_top_headlines(df, n=5):
    """
    Return top N most impactful headlines sorted by abs(numeric_sentiment) descending.
    Each headline is a dict: {title, source, url, sentiment_label, sentiment_score}
    """
    if df.empty:
        return []

    # Work on a copy so we don't mutate the caller's df
    work = df.copy()
    work['abs_sentiment'] = work['numeric_sentiment'].abs()
    top = work.nlargest(n, 'abs_sentiment')

    headlines = []
    for _, r in top.iterrows():
        headlines.append({
            'title': r.get('title', ''),
            'source': r.get('source', 'Unknown'),
            'url': r.get('url', ''),
            'sentiment_label': r.get('sentiment_label', 'neutral'),
            'sentiment_score': float(r.get('numeric_sentiment', 0.0)),
        })
    return headlines

def generate_sentiment_summary(company_name, news_score, top_headlines, bottom_headlines):
    """Generate sentiment summary using Gemini"""
    client = genai.Client(api_key=GOOGLE_API_KEY)
    
    prompt = f"""
    Analyze the sentiment data for {company_name} and provide a comprehensive summary.
    
    Sentiment Scores:
    - News Sentiment: {news_score:.3f}
    
    Top 10 Positive Headlines:
    {chr(10).join(top_headlines) if top_headlines else 'No positive headlines found'}
    
    Top 10 Negative Headlines:
    {chr(10).join(bottom_headlines) if bottom_headlines else 'No negative headlines found'}
    
    Please provide a 2-3 paragraph summary covering:
    1. Overall sentiment assessment
    2. Key themes from news coverage
    3. Potential market implications
    """
    
    try:
        response = client.models.generate_content(model='gemma-3-27b-it', contents=prompt)
        return response.text
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def compute_sentiment_score(tickers, tag_to_comp, NEWS_API_KEY, bearer_token=None, start_date=None, end_date=None):
    """
    Compute sentiment scores for multiple tickers and return summary dataframe.

    Returns a DataFrame where each row has:
      ticker, company_name, sentiment_score, sentiment_summary, top_headlines

    top_headlines is a list[dict] with keys:
      {title, source, url, sentiment_label, sentiment_score}
    sorted by abs(sentiment_score) descending (top 5 most impactful articles).
    """
    
    # Check if NEWS_API_KEY is provided
    if not NEWS_API_KEY:
        print("ERROR: NEWS_API_KEY is not provided!")
        return pd.DataFrame()
    
    print(f"Using NEWS_API_KEY: {NEWS_API_KEY[:10]}..." if len(NEWS_API_KEY) > 10 else f"Using NEWS_API_KEY: {NEWS_API_KEY}")
    
    # Initialize sentiment analysis pipelines
    sentiment_pipeline = pipeline("sentiment-analysis", model="rahilv/news-sentiment-analysis-roberta")
    
    results = []
    
    for ticker in tickers:
        try:
            print(f"\n=== Processing {ticker} ===")
            # Get news data with date range if provided
            news_df = get_company_news(ticker, tag_to_comp, NEWS_API_KEY, start_date, end_date)
            print(f"News DataFrame shape: {news_df.shape}")
            print(f"News DataFrame empty: {news_df.empty}")
            if not news_df.empty:
                print(f"First few articles: {news_df.head(2)}")
            
            # Analyze sentiment
            news_df = analyze_sentiment_dataframe(news_df, 'text', sentiment_pipeline)
            
            # Calculate average sentiments
            news_sentiment = news_df['numeric_sentiment'].mean() if not news_df.empty else 0.0
            
            # Use news sentiment as the main sentiment score
            sentiment_score = news_sentiment

            # ── NEW: build top-5 impactful headlines ─────────────────────
            top_headlines_list = get_top_headlines(news_df, n=5)
            
            # Get extreme content (for the Gemini summary)
            top_title_headlines, bottom_title_headlines = get_extreme_content(news_df, 'title')
            
            # Generate summary
            summary = generate_sentiment_summary(
                tag_to_comp[ticker], news_sentiment, top_title_headlines, bottom_title_headlines
            )
            
            # Store results
            results.append({
                'ticker': ticker,
                'company_name': tag_to_comp[ticker],
                'sentiment_score': sentiment_score,
                'sentiment_summary': summary,
                'top_headlines': top_headlines_list,   # NEW
            })
            
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
            results.append({
                'ticker': ticker,
                'company_name': tag_to_comp[ticker],
                'news_sentiment': 0.0,
                'sentiment_score': 0.0,
                'sentiment_summary': f"Error processing data: {str(e)}",
                'top_headlines': [],   # NEW
            })
    
    return pd.DataFrame(results)