# main.py

import pandas as pd
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize FastAPI app
app = FastAPI(
    title="Credence: Explainable Credit Intelligence Platform MVP",
    description="A real-time, explainable credit intelligence platform built for the CredTech Hackathon.",
    version="1.0.0"
)

# --- 1. In-Memory "Database" and Data Loading ---
# For the MVP, we use dictionaries to simulate our database tables.
# In a real-world scenario, this data would be loaded from a database (e.g., PostgreSQL).

# Load structured data (simulating our credit_scores table)
# Replace with your actual CSV file
structured_data_path = 'credit_data.csv' 
try:
    credit_scores_df = pd.read_csv(structured_data_path)
    # Convert DataFrame to a list of dictionaries for easy lookup
    credit_scores_db = {row['id']: row.to_dict() for _, row in credit_scores_df.iterrows()}
except FileNotFoundError:
    # A dummy in-memory database if the file is not found
    credit_scores_db = {
        101: {
            "id": 101,
            "income": 75000,
            "loan_amount": 25000,
            "age": 35,
            "employment_status": "Employed",
            "loan_purpose": "Home Renovation",
            "credit_history": 5,
            "score": 720,
            "risk_category": "Low Risk",
            "score_history": json.dumps([
                {"timestamp": "2025-07-01T12:00:00Z", "score": 710, "version": "v1.0.0"},
                {"timestamp": "2025-07-15T12:00:00Z", "score": 720, "version": "v1.0.0"}
            ])
        },
        102: {
            "id": 102,
            "income": 45000,
            "loan_amount": 10000,
            "age": 28,
            "employment_status": "Self-Employed",
            "loan_purpose": "Debt Consolidation",
            "credit_history": 2,
            "score": 580,
            "risk_category": "High Risk",
            "score_history": json.dumps([
                {"timestamp": "2025-07-02T12:00:00Z", "score": 600, "version": "v1.0.0"},
                {"timestamp": "2025-07-16T12:00:00Z", "score": 580, "version": "v1.0.0"}
            ])
        }
    }

# --- 2. Unstructured Data Simulation ---
# Simulating our unstructured_data table
unstructured_data_db = [
    {"id": 1, "associated_entity_id": 101, "headline_text": "Company A reports record profits this quarter.", "publish_date": "2025-07-10"},
    {"id": 2, "associated_entity_id": 102, "headline_text": "Firm B faces legal investigation over accounting irregularities.", "publish_date": "2025-07-14"},
    {"id": 3, "associated_entity_id": 101, "headline_text": "CEO of Company A wins 'Visionary Leader' award.", "publish_date": "2025-07-16"}
]
# Pre-process unstructured data with VADER sentiment analysis
sid = SentimentIntensityAnalyzer()
for record in unstructured_data_db:
    record['sentiment_score'] = sid.polarity_scores(record['headline_text'])['compound']
    risk_keywords = ["investigation", "lawsuit", "bankruptcy", "default", "fraud"]
    record['risk_keywords_found'] = [kw for kw in risk_keywords if kw in record['headline_text'].lower()]

# --- 3. Pydantic Models for API Responses ---
# Defines the structure of the data that the API will return,
# ensuring type safety and automatic documentation.

class ScoreHistory(BaseModel):
    timestamp: datetime
    score: int
    version: str

class Explanation(BaseModel):
    feature_name: str
    feature_value: Any
    contribution_value: Optional[float] = None
    rationale: Optional[str] = None

class UnstructuredDataItem(BaseModel):
    headline_text: str
    sentiment_score: float
    risk_keywords_found: List[str]

class CreditScoreResponse(BaseModel):
    id: int
    score: int
    risk_category: str
    explainability: List[Explanation]
    unstructured_data: List[UnstructuredDataItem]
    score_history: List[ScoreHistory]

# --- 4. API Endpoints ---

@app.get("/")
def read_root():
    return {"message": "Welcome to the Credence API. Use /docs for API documentation."}

@app.get("/credit_score/{entity_id}", response_model=CreditScoreResponse)
def get_credit_score(entity_id: int):
    """
    Retrieves the credit score, explanations, and related unstructured data for a given entity.
    """
    # 1. Lookup the entity in our "database"
    entity_data = credit_scores_db.get(entity_id)
    if not entity_data:
        raise HTTPException(status_code=404, detail="Entity not found")

    # 2. Generate simplified explanations (simulating Logistic Regression coefficients)
    # In a real model, this would come from a model's explainability library like SHAP.
    # For the MVP, we'll hardcode some logical "contributions".
    explainability = [
        Explanation(feature_name="income", feature_value=entity_data['income'], contribution_value=0.4, rationale="Higher income positively impacts the score."),
        Explanation(feature_name="loan_amount", feature_value=entity_data['loan_amount'], contribution_value=-0.2, rationale="A higher loan amount increases risk."),
        Explanation(feature_name="credit_history", feature_value=entity_data['credit_history'], contribution_value=0.3, rationale="Longer credit history reduces risk."),
        # Add a simple explanation for the unstructured data's sentiment score
        Explanation(
            feature_name="sentiment_score", 
            feature_value=next((item['sentiment_score'] for item in unstructured_data_db if item['associated_entity_id'] == entity_id), 0.0), 
            rationale="News sentiment contributes to the overall risk assessment."
        )
    ]

    # 3. Retrieve related unstructured data
    unstructured_data_list = [
        UnstructuredDataItem(**item) 
        for item in unstructured_data_db 
        if item['associated_entity_id'] == entity_id
    ]

    # 4. Parse the JSON score history
    score_history_data = json.loads(entity_data['score_history'])
    score_history_list = [ScoreHistory(**item) for item in score_history_data]

    # 5. Construct and return the full response
    return {
        "id": entity_data['id'],
        "score": entity_data['score'],
        "risk_category": entity_data['risk_category'],
        "explainability": explainability,
        "unstructured_data": unstructured_data_list,
        "score_history": score_history_list,
    }

# --- 5. Running the Application ---
# To run this from the command line, save the code as main.py and execute:
# uvicorn main:app --reload
# Access the API at http://127.0.0.1:8000/
# The interactive documentation is at http://127.0.0.1:8000/docs