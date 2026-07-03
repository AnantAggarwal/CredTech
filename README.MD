# CredTech: Corporate Credit Risk Platform

![CredTech](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL-blue)
![Next.js](https://img.shields.io/badge/Frontend-Next.js-black)

**CredTech** is a real-time corporate credit risk analysis platform that evaluates the financial health and default probability of publicly traded S&P 500 companies. It bridges the critical gap between lagging quantitative fundamental financial data and highly dynamic qualitative market sentiment.

## ✨ Key Features
- **Data Ingestion Pipeline**: Programmatically extracts and standardizes 10 years of financial filings (balance sheets and income statements) from Yahoo Finance to calculate core leverage ratios.
- **Machine Learning Engine**: Utilizes a **LightGBM** gradient-boosting classifier trained on historical financial ratios to predict the mathematical probability of corporate default.
- **Generative AI Sentiment**: Integrates the **Gemini 1.5 Flash API** to scrape and analyze live breaking financial news, extracting structured market sentiment scores and generating concise analyst notes.
- **The 'NexScore' Synthesis**: Mathematically blends the quantitative ML default probability (60% weight) and qualitative AI market sentiment (40% weight) into a unified 0-100 credit score.
- **Interactive Dashboard**: A responsive Next.js web application utilizing server-side pagination to visualize the synthesized NexScores, letter grades (AAA to CCC), and analyst notes.

## 🏗️ Architecture & Tech Stack
1. **Database**: PostgreSQL (hosted on Neon Serverless)
2. **Data Engineering**: Python, pandas, yfinance
3. **Machine Learning**: scikit-learn, LightGBM
4. **AI/LLM**: Google Gemini 1.5 Pro API
5. **Backend API**: FastAPI, Uvicorn, psycopg2
6. **Frontend UI**: Next.js, React, Tailwind CSS

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js (v18+) and npm
- A PostgreSQL database (e.g., [Neon](https://neon.tech))
- A Gemini API Key from Google AI Studio
- A News API Key (or alternative for scraping headlines)

### 1. Environment Setup
Clone the repository and set up your `.env` file in the root directory:
```env
DATABASE_URL=postgresql://user:password@endpoint.neon.tech/neondb
GEMINI_API_KEY=your_gemini_key_here
NEWS_KEY=your_news_api_key_here
```

Set up your Python virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Data Pipeline
Populate your PostgreSQL database with historical financial data for the S&P 500:
```bash
cd worker
python build_structured_features.py
```

### 3. Start the Background Worker (The "Brain")
This daemon runs continuously. It pulls companies from the queue, runs the LightGBM model, analyzes breaking news with Gemini, and saves the final NexScore to the database.
```bash
cd worker
python run.py --mode update
```

### 4. Start the Backend API
In a new terminal tab, start the FastAPI server:
```bash
source venv/bin/activate
cd backend
uvicorn main:app --host 127.0.0.1 --port 8000
```

### 5. Start the Web Dashboard
In a third terminal tab, start the Next.js frontend:
```bash
cd frontend-next
npm install
npm run dev
```

Navigate to **http://localhost:3000** in your web browser to view the live dashboard!

## 📝 License
This project is open-source and available under the MIT License.
