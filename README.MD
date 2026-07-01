# CredTech - Credit Risk & Market Sentiment Pipeline

A comprehensive AI-driven financial analysis platform that combines machine learning credit scoring with Gemini-powered sentiment analysis to evaluate top public companies.

## Architecture

This project is built using a modern decoupled architecture:
- **Database**: PostgreSQL (Neon) - Stores ML market features, news headlines, queue status, and final scores.
- **Background Worker**: Python (FastAPI/Pandas) - A daemon that continuously pulls pending companies, evaluates financial health, fetches latest news, and prompts Gemini for sentiment analysis.
- **Backend API**: FastAPI - Exposes the final computed scores to the frontend.
- **Frontend**: Next.js (React) - A modern, responsive dashboard to view company deep dives and leaderboards.

## Prerequisites

1. **Python 3.10+** (Recommended to use a virtual environment)
2. **Node.js 18+** & `npm`
3. **PostgreSQL** database instance (e.g., Neon or local)

You need a `.env` file in the root of the project with the following keys:
```env
# Database
DATABASE_URL=postgresql://user:password@host/dbname

# APIs
NEWS_API_KEY=your_newsapi_key
GEMINI_API_KEY=your_gemini_api_key
```

## Initializing the Data Pipeline

Before you can view anything in the frontend, you must build the machine learning database and seed the job queue.

1. **Activate the Virtual Environment**:
   ```bash
   source venv/bin/activate
   ```

2. **Initialize the Database Schema & Seed the Queue**:
   This drops existing tables and seeds the `pipeline_queue` with the S&P 500 top 300 companies.
   *(Press `Ctrl+C` once you see "Database recreated successfully.")*
   ```bash
   cd worker
   python run.py --mode init
   ```

3. **Download Market Features (Historical Data)**:
   This fetches 10 years of market data for the 300 companies from Yahoo Finance and saves it to the database so the ML model can generate credit scores.
   ```bash
   cd worker
   python build_structured_features.py
   ```

## Running the Application

To run the full stack locally, you need three separate terminal windows:

### Terminal 1: Background Worker
This daemon pulls companies from the queue, runs the ML model, fetches news, generates Gemini analyst notes, and saves the final `NexScore` to the database.
```bash
source venv/bin/activate
cd worker
python run.py --mode update
```

### Terminal 2: FastAPI Backend
This serves the API endpoints for the frontend to consume.
```bash
source venv/bin/activate
cd backend
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### Terminal 3: Next.js Frontend
This starts the web dashboard on `localhost:3000`.
```bash
cd frontend-next
npm run dev
```

## Usage
Once everything is running, open your browser and navigate to `http://localhost:3000`. 
Click **LOAD MORE** at the bottom of the page to refresh the list as the background worker processes more companies!


