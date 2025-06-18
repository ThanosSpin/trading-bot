# retrain_model.py
from data_loader import fetch_historical_data
from model_xgb import train_model
import pandas as pd

# Fetch data
df = fetch_historical_data(period="6mo", interval="1d")
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