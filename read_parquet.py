import pandas as pd
import glob

# Path to your parquet files
files = glob.glob("/Users/vaibhavkalyan/Desktop/ParquetReader/*.parquet")

# Load & merge into one DataFrame
dfs = [pd.read_parquet(f) for f in files]
data = pd.concat(dfs, ignore_index=True)

print("Shape:", data.shape)

# --- Clean column names (remove tuple formatting like ('mkt__adj close', 'mkt__bac')) ---
data.columns = [str(c).replace("(", "").replace(")", "").replace("'", "").replace(",", "_").replace(" ", "") 
                for c in data.columns]

print("Columns after cleaning:")
print(data.columns.tolist()[:30])  # show first 30 for preview

# --- Show sample ---
print("\nHead of Data:")
print(data.head())

# --- Date range and tickers ---
print("\nDate Range:", data['date'].min(), "to", data['date'].max())
print("Unique Tickers:", data['ticker'].unique())
