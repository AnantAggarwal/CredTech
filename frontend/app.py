import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np

# Configuration
st.set_page_config(
    page_title="CredTech - Stock Credit Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API URL
API_URL = "https://lcf-web.onrender.com/scores"

# Sample data for demonstration (replace with actual API calls)
def get_sample_data():
    """Generate sample data for demonstration"""
    data = requests.get(API_URL).json()
    
    return data

def get_sentiment_summary(sentiment_score):
    """Generate sentiment summary based on score"""
    if sentiment_score >= 60:
        return {
            'overall': 'Very Positive',
            'color': 'green',
            'summary': 'Market sentiment is very positive for this stock. Investors are optimistic about the company\'s future prospects, driven by strong fundamentals and positive news flow.',
            'key_points': [
                'Strong investor confidence',
                'Positive news sentiment',
                'Favorable analyst ratings',
                'Growing market share'
            ]
        }
    elif sentiment_score >= 20:
        return {
            'overall': 'Positive',
            'color': 'lightgreen',
            'summary': 'Market sentiment is generally positive. While there may be some concerns, the overall outlook remains favorable.',
            'key_points': [
                'Moderate investor confidence',
                'Generally positive news',
                'Stable analyst outlook',
                'Steady performance'
            ]
        }
    elif sentiment_score >= -20:
        return {
            'overall': 'Neutral',
            'color': 'gray',
            'summary': 'Market sentiment is neutral. Investors are taking a wait-and-see approach, with mixed signals from news and analyst reports.',
            'key_points': [
                'Mixed investor sentiment',
                'Balanced news coverage',
                'Diverse analyst opinions',
                'Stable but uncertain outlook'
            ]
        }
    elif sentiment_score >= -60:
        return {
            'overall': 'Negative',
            'color': 'orange',
            'summary': 'Market sentiment is negative. There are concerns about the company\'s performance and future prospects.',
            'key_points': [
                'Declining investor confidence',
                'Negative news sentiment',
                'Analyst downgrades',
                'Performance concerns'
            ]
        }
    else:
        return {
            'overall': 'Very Negative',
            'color': 'red',
            'summary': 'Market sentiment is very negative. Significant concerns exist about the company\'s viability and future prospects.',
            'key_points': [
                'Very low investor confidence',
                'Highly negative news coverage',
                'Multiple analyst downgrades',
                'Serious performance issues'
            ]
        }

def get_credit_rating(credit_score):
    """Get credit rating based on score"""
    if credit_score >= 90:
        return 'AAA', 'Excellent', 'green'
    elif credit_score >= 80:
        return 'AA', 'Very Good', 'lightgreen'
    elif credit_score >= 70:
        return 'A', 'Good', 'blue'
    elif credit_score >= 60:
        return 'BBB', 'Fair', 'orange'
    elif credit_score >= 50:
        return 'BB', 'Below Average', 'red'
    else:
        return 'B', 'Poor', 'darkred'

def create_credit_score_chart(dates, scores, ticker):
    """Create credit score trend chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=scores,
        mode='lines+markers',
        name='Credit Score',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=6),
        fill='tonexty',
        fillcolor='rgba(31, 119, 180, 0.1)'
    ))
    
    fig.update_layout(
        title=f'{ticker} Credit Score Trend',
        xaxis_title='Date',
        yaxis_title='Credit Score',
        yaxis=dict(range=[0, 100]),
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig

def create_sentiment_chart(dates, scores, ticker):
    """Create sentiment score trend chart"""
    fig = go.Figure()
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=scores,
        mode='lines+markers',
        name='Sentiment Score',
        line=dict(color='#ff7f0e', width=3),
        marker=dict(size=6),
        fill='tonexty',
        fillcolor='rgba(255, 127, 14, 0.1)'
    ))
    
    fig.update_layout(
        title=f'{ticker} Sentiment Score Trend',
        xaxis_title='Date',
        yaxis_title='Sentiment Score',
        yaxis=dict(range=[-100, 100]),
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig

def main():
    # Sidebar navigation
    st.sidebar.title("📈 CredTech")
    page = st.sidebar.selectbox(
        "Navigation",
        ["🏆 Leaderboard", "🔍 Stock Analysis", "📊 About"]
    )
    
    # Get sample data
    sample_data = get_sample_data()
    
    if page == "🏆 Leaderboard":
        show_leaderboard(sample_data)
    elif page == "🔍 Stock Analysis":
        show_stock_analysis(sample_data)
    elif page == "📊 About":
        show_about_page()

def show_leaderboard(data):
    st.title("🏆 Stock Credit Score Leaderboard")
    st.markdown("Ranking of stocks based on their credit scores and financial health")
    
    # Sort data by credit score
    sorted_data = sorted(data, key=lambda x: x['credit_score'], reverse=True)
    
    # Create leaderboard
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader("Rank")
    with col2:
        st.subheader("Credit Score")
    with col3:
        st.subheader("Rating")
    
    st.markdown("---")
    
    for i, stock in enumerate(sorted_data, 1):
        rating, rating_text, color = get_credit_rating(stock['credit_score'])
        
        col1, col2, col3, col4 = st.columns([0.5, 1.5, 1, 1])
        
        with col1:
            if i == 1:
                st.markdown("🥇")
            elif i == 2:
                st.markdown("🥈")
            elif i == 3:
                st.markdown("🥉")
            else:
                st.markdown(f"**{i}**")
        
        with col2:
            st.markdown(f"**{stock['ticker']}** - {stock['name']}")
        
        with col3:
            st.markdown(f"**{stock['credit_score']}**")
        
        with col4:
            st.markdown(f":{color}[**{rating}**]")
    
    # Summary statistics
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_score = sum(s['credit_score'] for s in data) / len(data)
        st.metric("Average Credit Score", f"{avg_score:.1f}")
    
    with col2:
        max_score = max(s['credit_score'] for s in data)
        st.metric("Highest Score", f"{max_score}")
    
    with col3:
        min_score = min(s['credit_score'] for s in data)
        st.metric("Lowest Score", f"{min_score}")
    
    with col4:
        st.metric("Total Stocks", len(data))

def show_stock_analysis(data):
    st.title("🔍 Stock Analysis")
    
    # Stock selection
    stock_options = {f"{s['ticker']} - {s['company']}": s for s in data}
    selected_stock_name = st.selectbox(
        "Select a stock to analyze:",
        options=list(stock_options.keys())
    )
    
    if selected_stock_name:
        selected_stock = stock_options[selected_stock_name]
        
        # Header with stock info
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Ticker", selected_stock['ticker'])
        
        with col2:
            credit_rating, rating_text, color = get_credit_rating(selected_stock['credit_score'])
            st.metric("Credit Score", f"{selected_stock['credit_score']} ({credit_rating})")
        
        with col3:
            sentiment_info = get_sentiment_summary(selected_stock['sentiment_score'])
            st.metric("Sentiment Score", f"{selected_stock['sentiment_score']}")
        
        with col4:
            st.metric("Last Updated", selected_stock['last_updated'].strftime('%Y-%m-%d'))
        
        # Main content tabs
        tab1, tab2, tab3 = st.tabs(["📊 Credit Analysis", "😊 Sentiment Analysis", "📈 Historical Trends"])
        
        with tab1:
            show_credit_analysis(selected_stock)
        
        with tab2:
            show_sentiment_analysis(selected_stock)
        
        with tab3:
            show_historical_trends(selected_stock)

def show_credit_analysis(stock):
    st.subheader("Credit Score Analysis")
    
    credit_rating, rating_text, color = get_credit_rating(stock['credit_score'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Credit score gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=stock['credit_score'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Credit Score"},
            delta={'reference': 70},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 70], 'color': "yellow"},
                    {'range': [70, 90], 'color': "lightgreen"},
                    {'range': [90, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown(f"**Credit Rating:** {credit_rating}")
        st.markdown(f"**Rating Description:** {rating_text}")
        st.markdown(f"**Score:** {stock['credit_score']}/100")
        
        # Credit score interpretation
        if stock['credit_score'] >= 80:
            st.success("This stock has excellent creditworthiness and low default risk.")
        elif stock['credit_score'] >= 60:
            st.info("This stock has good creditworthiness with moderate risk.")
        else:
            st.warning("This stock has below-average creditworthiness and higher risk.")

def show_sentiment_analysis(stock):
    st.subheader("Sentiment Analysis")
    
    sentiment_info = get_sentiment_summary(stock['sentiment_score'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=stock['sentiment_score'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Sentiment Score"},
            delta={'reference': 0},
            gauge={
                'axis': {'range': [-100, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [-100, -50], 'color': "red"},
                    {'range': [-50, 0], 'color': "orange"},
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown(f"**Overall Sentiment:** {sentiment_info['overall']}")
        st.markdown(f"**Score:** {stock['sentiment_score']}/100")
        
        # Sentiment summary
        st.markdown("**Summary:**")
        st.markdown(sentiment_info['summary'])
        
        st.markdown("**Key Points:**")
        for point in sentiment_info['key_points']:
            st.markdown(f"• {point}")

def show_historical_trends(stock):
    st.subheader("Historical Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Credit score trend
        credit_fig = create_credit_score_chart(
            stock['historical_dates'], 
            stock['historical_credit'], 
            stock['ticker']
        )
        st.plotly_chart(credit_fig, use_container_width=True)
    
    with col2:
        # Sentiment trend
        sentiment_fig = create_sentiment_chart(
            stock['historical_dates'], 
            stock['historical_sentiment'], 
            stock['ticker']
        )
        st.plotly_chart(sentiment_fig, use_container_width=True)
    
    # Combined analysis
    st.subheader("Combined Analysis")
    
    # Create combined chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=stock['historical_dates'],
        y=stock['historical_credit'],
        mode='lines+markers',
        name='Credit Score',
        line=dict(color='#1f77b4', width=3),
        yaxis='y'
    ))
    
    fig.add_trace(go.Scatter(
        x=stock['historical_dates'],
        y=stock['historical_sentiment'],
        mode='lines+markers',
        name='Sentiment Score',
        line=dict(color='#ff7f0e', width=3),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title=f'{stock["ticker"]} - Credit Score vs Sentiment Score',
        xaxis_title='Date',
        yaxis=dict(title='Credit Score', range=[0, 100]),
        yaxis2=dict(title='Sentiment Score', range=[-100, 100], overlaying='y', side='right'),
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_about_page():
    st.title("📊 About CredTech")
    
    st.markdown("""
    ## What is CredTech?
    
    CredTech is an advanced financial analysis platform that combines traditional credit scoring 
    methodologies with modern sentiment analysis to provide comprehensive stock evaluation.
    
    ## Features
    
    ### 🏆 Credit Score Analysis
    - **Financial Health Assessment**: Analyzes key financial ratios and metrics
    - **Credit Rating**: Provides standardized credit ratings (AAA to B)
    - **Risk Assessment**: Evaluates default risk and financial stability
    
    ### 😊 Sentiment Analysis
    - **Market Sentiment**: Analyzes news, social media, and analyst opinions
    - **Sentiment Scoring**: Provides numerical sentiment scores (-100 to +100)
    - **Trend Analysis**: Tracks sentiment changes over time
    
    ### 📈 Historical Trends
    - **Credit Score Trends**: Visualizes credit score changes over time
    - **Sentiment Trends**: Shows sentiment evolution
    - **Combined Analysis**: Correlates credit and sentiment metrics
    
    ## Methodology
    
    Our platform uses:
    - **Machine Learning Models**: For sentiment analysis and credit scoring
    - **Real-time Data**: From financial APIs and news sources
    - **Statistical Analysis**: For trend identification and risk assessment
    
    ## Disclaimer
    
    This platform is for educational and research purposes. Investment decisions should be based 
    on comprehensive analysis and consultation with financial advisors.
    """)

if __name__ == "__main__":
    main()
