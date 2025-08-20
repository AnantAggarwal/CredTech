import streamlit as st
import requests
import pandas as pd

API_URL = "http://localhost:8000/scores"  # later replace with your Render URL

st.title("📈 CredTech - Phase 1 Prototype")

try:
    r = requests.get(API_URL)
    data = r.json()
    df = pd.DataFrame(data)
    st.dataframe(df)

    if not df.empty:
        st.line_chart(df.set_index("updated_at")["score"])
except Exception as e:
    st.error(f"Error fetching data: {e}")
