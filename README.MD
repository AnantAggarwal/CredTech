# 🏆 CredTech Design Document

## 1. Problem Statement

Current credit scoring is opaque and often biased. Startups, banks, and even individuals lack a transparent, auditable system for assessing financial health beyond traditional bureaus.
**Goal:** Build an **open-source Credit Risk Engine** that fetches financial data, computes features, scores risk, and displays it transparently in a dashboard.


## 2. Success Criteria (Judges’ POV)

* End-to-end demo: data ingestion → scoring → API → dashboard.
* Live deployment with working URLs.
* Clear modularity (team could extend easily).
* Explainability (why score X was given).
* Stable enough to refresh with real data during demo.
* User friendly interface and innovative
* Well explained design choices and tradeoffs
* We need to establish the models used are accurate

## 3. System Overview

### The Product Architecture

1. **Background Worker to update our website in realtime**  
    * Fetch realtime data through a robust pipeline
    * Finetune the values on the newly fetched data
    * Update storage with updated values
2. **Storage API: Store all generated stuff for a stock in db**
    * An authorised end point to update data in the DB
    * Endpoint to request data
3. **Front End component to retrieve info from DB and present it in a Effective way**
    * Landing Page: Shows general stuff
    * Company Wise: All company specific stuff

### Data to Result
Credit Score  
Sentiment Score
SHAP Graph
Credit History
Sentiment History
Overall Summary
Sentiment Summary
News Headlines and Corresponding Impact
Candle Stick
Leaderboard
News Highlights


