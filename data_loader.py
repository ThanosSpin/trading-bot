# data_loader.py
import yfinance as yf
import pandas as pd
from typing import Optional
from broker import api_market
import pytz
from datetime import datetime, timedelta


def fetch_historical_data(
    symbol: str,
    years: Optional[int] = None,
    period: Optional[str] = None,
    interval: str = "1d",
) -> Optional[pd.DataFrame]:
    """
    Fetch historical price data for a given symbol.

    Parameters:
        symbol (str): Stock symbol.
        years (int, optional): Number of years of historical data (used for training).
        period (str, optional): Yahoo Finance period string (e.g., '6mo', '1y', '60d')
                                used for prediction or intraday training.
        interval (str): Data interval ('1d', '1h', '15m', '1m', etc.)

    Returns:
        pd.DataFrame or None
    """
    try:
        if period is not None:
            # e.g. period="6mo" for predictions, "60d" for intraday training
            df = yf.download(
                symbol,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True,
            )
        elif years is not None:
            # e.g. years=2 → period="2y" for daily training
            df = yf.download(
                symbol,
                period=f"{years}y",
                interval=interval,
                progress=False,
                auto_adjust=True,
            )
        else:
            raise ValueError("Either 'years' or 'period' must be provided.")

        if df is None or df.empty:
            print(f"[WARN] DataFrame is empty or missing required columns for {symbol}.")
            return None

        return df

    except Exception as e:
        print(f"[ERROR] Failed to fetch historical data for {symbol}: {e}")
        return None


def fetch_latest_price(symbol: str) -> Optional[float]:
    """
    Fetch the most recent price for a given stock symbol.
    Uses 1m interval for live-like updates.
    """
    try:
        data = yf.download(
            symbol,
            period="1d",
            interval="1m",
            progress=False,
            auto_adjust=True,
        )
        if not data.empty:
            return float(data["Close"].iloc[-1].item())
        return None
    except Exception as e:
        print(f"[ERROR] fetch_latest_price {symbol}: {e}")
        return None


def fetch_intraday_history(symbol, lookback_minutes=120, interval="1m"):
    """Wrapper using Alpaca intraday bars."""
    return fetch_intraday_history_alpaca(symbol, lookback_minutes)

# ----------------------------------------------------------
# Alpaca Intraday Bars (1m) — replaces yfinance intraday
# ----------------------------------------------------------
def fetch_intraday_history_alpaca(symbol: str, lookback_minutes: int = 120):

    ny = pytz.timezone("America/New_York")

    now = datetime.now(ny)
    start = now - timedelta(minutes=lookback_minutes + 30)   # extra padding
    end = now

    try:
        bars = api_market.get_bars(
            symbol,
            timeframe="1Min",
            start=start.isoformat(),
            end=end.isoformat(),
            adjustment="raw"
        )

        if not bars:
            print(f"[WARN] Alpaca returned no intraday bars for {symbol}.")
            return None

        # Convert to DataFrame
        df = pd.DataFrame([{
            "timestamp": b.t,
            "Open": float(b.o),
            "High": float(b.h),
            "Low": float(b.l),
            "Close": float(b.c),
            "Volume": int(b.v),
        } for b in bars])

        if df.empty:
            print(f"[WARN] Intraday DF empty for {symbol}.")
            return None

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()

        # Trim exactly to lookback window
        df = df.last(f"{lookback_minutes}T")

        return df

    except Exception as e:
        print(f"[ERROR] Alpaca intraday fetch failed for {symbol}: {e}")
        return None