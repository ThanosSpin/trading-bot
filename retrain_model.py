# retrain_model.py
from data_loader_multi import fetch_historical_data
from model_xgb import train_model
from config import SYMBOL
import pandas as pd

LOOKBACK_YEARS = 2   # how many years of historical data to use
INTERVAL = "1d" 

# Fetch data
df = fetch_historical_data(symbol=SYMBOL, years=LOOKBACK_YEARS, interval=INTERVAL)
df['Return'] = df['Close'].pct_change()

# Add technical indicators
df['MA_5'] = df['Close'].rolling(window=5).mean()
df['MA_20'] = df['Close'].rolling(window=20).mean()
df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
df['Volatility_10'] = df['Return'].rolling(window=10).std()
df['Volume_Change'] = df['Volume'].pct_change()

# Drop rows with NaNs created by indicators
df = df.dropna()

# Train
train_model(df)