# data_loader.py
import pandas as pd
import pytz
from datetime import datetime, timedelta
from typing import Optional
import yfinance as yf

# Alpaca API
from broker import api_market


# ============================================================
# DAILY (Yahoo) — still OK for training and daily signals
# ============================================================
def fetch_historical_data(symbol: str, years: Optional[int] = None,
                          period: Optional[str] = None,
                          interval: str = "1d") -> Optional[pd.DataFrame]:

    try:
        if period:
            df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
        elif years:
            df = yf.download(symbol, period=f"{years}y", interval=interval, progress=False, auto_adjust=True)
        else:
            raise ValueError("Either 'years' or 'period' must be provided.")

        if df is None or df.empty:
            print(f"[WARN] Empty daily data for {symbol}")
            return None

        return df

    except Exception as e:
        print(f"[ERROR] Daily data fetch failed for {symbol}: {e}")
        return None


# ============================================================
# LATEST PRICE (Yahoo fallback)
# ============================================================
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
            last_close = data["Close"].iloc[-1]

            # Fix FutureWarning: if pandas returns a Series, extract first element
            if isinstance(last_close, pd.Series):
                last_close = last_close.iloc[0]

            return float(last_close)

        return None

    except Exception as e:
        print(f"[ERROR] fetch_latest_price {symbol}: {e}")
        return None


# ============================================================
# **REPLACED** INTRADAY — ALPACA MARKET DATA v2
# ============================================================
def fetch_intraday_history_alpaca(symbol: str, lookback_minutes: int = 120) -> Optional[pd.DataFrame]:
    try:
        ny = pytz.timezone("America/New_York")
        now = datetime.now(ny)

        start = now - timedelta(minutes=lookback_minutes + 30)
        end = now

        bars = api_market.get_bars(
            symbol,
            timeframe="1Min",
            start=start.isoformat(),
            end=end.isoformat(),
            adjustment="raw"
        )

        if not bars:
            print(f"[WARN] Alpaca returned NO bars for {symbol}")
            return None

        df = pd.DataFrame([{
            "timestamp": b.t,
            "Open": float(b.o),
            "High": float(b.h),
            "Low": float(b.l),
            "Close": float(b.c),
            "Volume": int(b.v),
        } for b in bars])

        if df.empty:
            print(f"[WARN] Empty Alpaca intraday DF for {symbol}")
            return None

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()

        # Trim exactly to lookback window
        df = df.last(f"{lookback_minutes}T")

        return df

    except Exception as e:
        print(f"[ERROR] Alpaca intraday fetch failed for {symbol}: {e}")
        return None


# ============================================================
# Wrapper (compute_signals calls this)
# ============================================================
def fetch_intraday_history(symbol: str, lookback_minutes: int = 120, interval: str = "1m"):
    return fetch_intraday_history_alpaca(symbol, lookback_minutes)