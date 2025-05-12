# data_loader.py
import yfinance as yf
import pandas as pd
from config import SYMBOL

def fetch_historical_data(period="60d", interval="1d"):
    data = yf.download(SYMBOL, period=period, interval=interval)
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    return data

def fetch_latest_price():
    data = yf.download(SYMBOL, period="1d", interval="1m")
    if not data.empty:
        price = data['Close'].iloc[-1]
        return float(price.iloc[0])
    return None
