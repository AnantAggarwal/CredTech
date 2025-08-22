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
    if response.status_code != 200:
        print(f"Error: News API returned status code {response.status_code}")
        print(f"Response: {response.text}")
        return pd.DataFrame()
    
    data = response.json()
    print(f"News API response status: {data.get('status', 'unknown')}")
    print(f"Total results: {data.get('totalResults', 0)}")
    
    articles = data.get('articles', [])
    print(f"Number of articles received: {len(articles)}")

    # Filter for relevant articles, create a DataFrame
    news_data = []
    for i, article in enumerate(articles):
        # Check if required fields exist
        if not article.get('title') or not article.get('description'):
            print(f"Skipping article {i}: missing title or description")
            continue
            
        news_data.append({
            'source': article['source']['name'] if article.get('source') else 'Unknown',
            'title': article['title'],
            'description': article['description'],
            'url': article.get('url', ''),
            'published_at': article.get('publishedAt', ''),
            'text': article['title'] + '\n' + article['description']
        })
    
    print(f"Processed {len(news_data)} valid articles")
    return pd.DataFrame(news_data)

def label_to_numeric(row):
    """Convert sentiment label and score to numeric value"""
    score = row['sentiment_score']
    print(score)
    if row['sentiment_label'] == 'negative':
        return -score
    elif row['sentiment_label'] == 'positive':
        return score
    else: # Neutral
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
        response = client.models.generate_content(model='gemma-3-27b-it', content=prompt)
        return response.text
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def compute_sentiment_score(tickers, tag_to_comp, NEWS_API_KEY, bearer_token=None, start_date=None, end_date=None):
    """Compute sentiment scores for multiple tickers and return summary dataframe"""
    
    # Check if NEWS_API_KEY is provided
    if not NEWS_API_KEY:
        print("ERROR: NEWS_API_KEY is not provided!")
        return pd.DataFrame()
    
    print(f"Using NEWS_API_KEY: {NEWS_API_KEY[:10]}..." if len(NEWS_API_KEY) > 10 else f"Using NEWS_API_KEY: {NEWS_API_KEY}")
    
    # Initialize sentiment analysis pipelines
    sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    
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
            
            # Get extreme content
            top_headlines, bottom_headlines = get_extreme_content(news_df, 'title')
            
            # Generate summary
            summary = generate_sentiment_summary(
                tag_to_comp[ticker], news_sentiment, top_headlines, bottom_headlines
            )
            
            # Store results
            results.append({
                'ticker': ticker,
                'company_name': tag_to_comp[ticker],
                'sentiment_score': sentiment_score,
                'sentiment_summary': summary
            })
            
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
            results.append({
                'ticker': ticker,
                'company_name': tag_to_comp[ticker],
                'news_sentiment': 0.0,
                'sentiment_score': 0.0,
                'sentiment_summary': f"Error processing data: {str(e)}"
            })
    
    return pd.DataFrame(results)