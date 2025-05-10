# data_loader.py
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

from config import SYMBOL

def fetch_historical_data(symbol=SYMBOL, period="60d", interval="1d"):
    """Fetch historical stock data from Yahoo Finance."""
    data = yf.download(symbol, period=period, interval=interval, progress=False)
    data = data.dropna()
    return data

def fetch_latest_price(symbol=SYMBOL):
    """Fetch the latest close price."""
    data = yf.download(symbol, period="1d", interval="1m", progress=False)
    if not data.empty:
        return data.iloc[-1]['Close']
    return None

if __name__ == "__main__":
    df = fetch_historical_data()
    print(df.tail())
