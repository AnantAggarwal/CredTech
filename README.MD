# CredTech - Stock Credit Analysis Platform

A comprehensive financial analysis platform that combines traditional credit scoring with modern sentiment analysis to provide detailed stock evaluation.

## Features

### 🏆 Leaderboard
- **Credit Score Rankings**: View stocks ranked by their credit scores
- **Financial Health Overview**: Quick comparison of stock creditworthiness
- **Summary Statistics**: Average, highest, and lowest scores across all stocks

### 🔍 Stock Analysis
- **Credit Score Analysis**: Detailed credit rating and risk assessment
- **Sentiment Analysis**: Market sentiment scoring and interpretation
- **Historical Trends**: Visual charts showing credit and sentiment score evolution
- **Combined Analysis**: Correlation between credit and sentiment metrics

### 📊 Interactive Visualizations
- **Gauge Charts**: Real-time credit and sentiment score displays
- **Trend Charts**: Historical performance tracking
- **Combined Charts**: Side-by-side analysis of multiple metrics

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd CredTech
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   cd frontend
   streamlit run app.py
   ```

## Usage

### Navigation
The application has three main pages accessible via the sidebar:

1. **🏆 Leaderboard**: View all stocks ranked by credit score
2. **🔍 Stock Analysis**: Select and analyze individual stocks
3. **📊 About**: Learn about the platform and methodology

### Stock Analysis Features
- **Credit Score**: Numerical score (0-100) with letter rating (AAA to B)
- **Sentiment Score**: Market sentiment (-100 to +100) with detailed summary
- **Historical Trends**: Monthly data showing score evolution over time
- **Risk Assessment**: Color-coded risk levels and recommendations

## Data Sources

Currently using sample data for demonstration. In production, the platform integrates with:
- Financial APIs for real-time stock data
- News APIs for sentiment analysis
- Credit rating agencies for official ratings
- Social media platforms for market sentiment

## Technology Stack

- **Frontend**: Streamlit (Python web framework)
- **Visualization**: Plotly (Interactive charts)
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, Transformers
- **APIs**: FastAPI, Requests

## Disclaimer

This platform is for educational and research purposes. Investment decisions should be based on comprehensive analysis and consultation with financial advisors.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.


