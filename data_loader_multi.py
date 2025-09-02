# data_loader.py
import yfinance as yf
import pandas as pd
from config_multi import SYMBOL, LOOKBACK_YEARS, INTERVAL


def fetch_historical_data(symbol, years=LOOKBACK_YEARS, interval=INTERVAL):
    """
    Download historical data for a given stock symbol.
    Default: LOOKBACK_YEARS (config) years, INTERVAL (config) interval.
    """
    period = f"{years}y" if years > 0 else "max"

    data = yf.download(symbol, period=period, interval=interval)
    if not data.empty:
        return data[['Open', 'High', 'Low', 'Close', 'Volume']]
    return data


def fetch_latest_price(symbol):
    """
    Fetch the most recent price for a given stock symbol.
    Uses 1m interval for live-like updates.
    """
    data = yf.download(symbol, period="1d", interval="1m")
    if not data.empty:
        return float(data['Close'].iloc[-1].item())
    return None