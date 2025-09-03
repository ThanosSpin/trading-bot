# data_loader.py
import yfinance as yf
import pandas as pd
from config_multi import SYMBOL, LOOKBACK_YEARS, INTERVAL


def fetch_historical_data(symbol, years=None, period=None, interval="1d"):
    """
    Fetch historical price data for a given symbol.

    Parameters:
        symbol (str): Stock symbol.
        years (int, optional): Number of years of historical data (used for training).
        period (str, optional): Yahoo Finance period string (e.g., '6mo', '1y') for prediction.
        interval (str): Data interval ('1d', '1h', etc.)

    Returns:
        pd.DataFrame: Historical OHLCV data or None if failed.
    """

    try:
        if period:
            # Use period string if provided (for predictions)
            df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
        elif years:
            # Convert years to period for Yahoo Finance (e.g., 2 years = '2y')
            df = yf.download(symbol, period=f"{years}y", interval=interval, progress=False, auto_adjust=True)
        else:
            raise ValueError("Either 'years' or 'period' must be provided.")

        if df.empty:
            print(f"[WARN] DataFrame is empty or missing required columns for {symbol}.")
            return None

        return df

    except Exception as e:
        print(f"[ERROR] Failed to fetch historical data for {symbol}: {e}")
        return None


def fetch_latest_price(symbol):
    """
    Fetch the most recent price for a given stock symbol.
    Uses 1m interval for live-like updates.
    """
    data = yf.download(symbol, period="1d", interval="1m")
    if not data.empty:
        return float(data['Close'].iloc[-1].item())
    return None
