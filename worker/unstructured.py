import tweepy
import os
import requests
import pandas as pd
from transformers import pipeline
import google.generativeai as genai
from datetime import datetime

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def get_company_tweets(ticker, tag_to_comp, bearer_token):
    client = tweepy.Client(bearer_token)
    query = ticker + " or " + tag_to_comp[ticker]
    tweet_data = []
    response = client.search_recent_tweets(query, max_results=100, tweet_fields=["created_at"])
    tweets = response.data
    for tweet in tweets:
        tweet_data.append({"text": tweet.text, "created_at": tweet.created_at, "company": ticker})

    df = pd.DataFrame(tweet_data)
    return df

def get_company_news(ticker, tag_to_comp, NEWS_API_KEY):
    query = f'"{tag_to_comp[ticker]}" OR {ticker}'
    url = (f'https://newsapi.org/v2/everything?'
           f'q={query}&'
           f'language=en&'
           f'sortBy=publishedAt&'
           f'apiKey={NEWS_API_KEY}')

    response = requests.get(url)
    articles = response.json().get('articles', [])

    # Filter for relevant articles, create a DataFrame
    news_data = []
    for article in articles:
        news_data.append({
            'source': article['source']['name'],
            'title': article['title'],
            'description': article['description'],
            'url': article['url'],
            'published_at': article['publishedAt'],
            'text':article['title']+'\n'+article['description']
        })
    return pd.DataFrame(news_data)

def label_to_numeric(row):
    """Convert sentiment label and score to numeric value"""
    score = row['sentiment_score']
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

def generate_sentiment_summary(company_name, tweet_score, news_score, blended_score, 
                             top_tweets, bottom_tweets, top_headlines, bottom_headlines):
    """Generate sentiment summary using Gemini"""
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    
    prompt = f"""
    Analyze the sentiment data for {company_name} and provide a comprehensive summary.
    
    Sentiment Scores:
    - Twitter Sentiment: {tweet_score:.3f}
    - News Sentiment: {news_score:.3f}
    - Blended Score: {blended_score:.3f}
    
    Top 10 Positive Tweets:
    {chr(10).join(top_tweets) if top_tweets else 'No positive tweets found'}
    
    Top 10 Negative Tweets:
    {chr(10).join(bottom_tweets) if bottom_tweets else 'No negative tweets found'}
    
    Top 10 Positive Headlines:
    {chr(10).join(top_headlines) if top_headlines else 'No positive headlines found'}
    
    Top 10 Negative Headlines:
    {chr(10).join(bottom_headlines) if bottom_headlines else 'No negative headlines found'}
    
    Please provide a 2-3 paragraph summary covering:
    1. Overall sentiment assessment
    2. Key themes from social media
    3. Key themes from news coverage
    4. Potential market implications
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def compute_sentiment_score(tickers, tag_to_comp, NEWS_API_KEY, bearer_token):
    """Compute sentiment scores for multiple tickers and return summary dataframe"""
    
    # Initialize sentiment analysis pipelines
    sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    
    results = []
    
    for ticker in tickers:
        try:
            # Get data
            tweets_df = get_company_tweets(ticker, tag_to_comp, bearer_token)
            news_df = get_company_news(ticker, tag_to_comp, NEWS_API_KEY)
            
            # Analyze sentiment
            tweets_df = analyze_sentiment_dataframe(tweets_df, 'text', sentiment_pipeline)
            news_df = analyze_sentiment_dataframe(news_df, 'text', sentiment_pipeline)
            
            # Calculate average sentiments
            tweet_sentiment = tweets_df['numeric_sentiment'].mean() if not tweets_df.empty else 0.0
            news_sentiment = news_df['numeric_sentiment'].mean() if not news_df.empty else 0.0
            
            # Calculate weighted sentiment
            tweet_weight = 0.4
            news_weight = 0.6
            sentiment_score = tweet_sentiment * tweet_weight + news_sentiment * news_weight
            
            # Get extreme content
            top_tweets, bottom_tweets = get_extreme_content(tweets_df, 'text')
            top_headlines, bottom_headlines = get_extreme_content(news_df, 'title')
            
            # Generate summary
            summary = generate_sentiment_summary(
                tag_to_comp[ticker], tweet_sentiment, news_sentiment, sentiment_score,
                top_tweets, bottom_tweets, top_headlines, bottom_headlines
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
                'tweet_sentiment': 0.0,
                'news_sentiment': 0.0,
                'blended_sentiment': 0.0,
                'tweet_count': 0,
                'news_count': 0,
                'sentiment_summary': f"Error processing data: {str(e)}"
            })
    
    return pd.DataFrame(results)