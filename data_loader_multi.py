# data_loader.py
import yfinance as yf
import pandas as pd
from config_multi import SYMBOL  # Default symbol fallback

def fetch_historical_data(symbol=SYMBOL, period="90d", interval="1d"):
    """
    Download historical data for a given stock symbol.
    """
    data = yf.download(symbol, period=period, interval=interval)
    if not data.empty:
        return data[['Open', 'High', 'Low', 'Close', 'Volume']]
    return data

def fetch_latest_price(symbol=SYMBOL):
    """
    Fetch the most recent price for a given stock symbol.
    """
    data = yf.download(symbol, period="1d", interval="1m")
    if not data.empty:
        return float(data['Close'].iloc[-1].item())
    return None