# data_loader.py
import yfinance as yf
import pandas as pd
from typing import Optional


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


def fetch_intraday_history(
    symbol: str,
    lookback_minutes: int = 60,
    interval: str = "1m",
) -> Optional[pd.DataFrame]:
    """
    Fetch intraday bars for the last `lookback_minutes` (approximate).

    Implementation detail:
    - yfinance requires at least period='1d' for intraday.
    - We download a small recent period (e.g. '2d') and then trim by timestamp.
    """
    try:
        period = "2d"
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
        )
        if df is None or df.empty:
            print(f"[WARN] Intraday DataFrame is empty for {symbol}.")
            return None

        df = df.sort_index()

        # approximate trim by row count instead of timezone math
        if lookback_minutes > 0 and len(df) > lookback_minutes:
            df = df.tail(lookback_minutes)

        return df

    except Exception as e:
        print(f"[ERROR] fetch_intraday_history {symbol}: {e}")
        return None