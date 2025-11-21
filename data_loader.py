# data_loader.py
import pandas as pd
import pytz
from datetime import datetime, timedelta
from typing import Optional
import yfinance as yf

# Alpaca API (market data)
from broker import api_market


# ============================================================
# DAILY DATA (Yahoo Finance)
# ============================================================
def fetch_historical_data(symbol: str,
                          years: Optional[int] = None,
                          period: Optional[str] = None,
                          interval: str = "1d") -> Optional[pd.DataFrame]:
    """
    Fetches daily or multi-interval historical data from Yahoo Finance.
    Works well for training + daily model inference.
    """
    try:
        if period:
            df = yf.download(symbol, period=period, interval=interval,
                             progress=False, auto_adjust=True)
        elif years:
            df = yf.download(symbol, period=f"{years}y", interval=interval,
                             progress=False, auto_adjust=True)
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
# LATEST PRICE (Yahoo Finance fallback)
# ============================================================
def fetch_latest_price(symbol: str) -> Optional[float]:
    """
    Fetch the most recent price for a symbol using 1m Yahoo Finance candles.
    """
    try:
        data = yf.download(
            symbol, period="1d", interval="1m",
            progress=False, auto_adjust=True
        )

        if data.empty:
            return None

        last_close = data["Close"].iloc[-1]

        # Fix FutureWarning: Series → extract scalar
        if isinstance(last_close, pd.Series):
            last_close = last_close.iloc[0]

        return float(last_close)

    except Exception as e:
        print(f"[ERROR] fetch_latest_price {symbol}: {e}")
        return None


# ============================================================
# INTRADAY — Alpaca (IEX feed) → FREE TIER COMPATIBLE
# ============================================================
def fetch_intraday_history_alpaca(symbol: str,
                                  lookback_minutes: int = 120) -> Optional[pd.DataFrame]:
    """
    Fetch intraday (1-minute) bars using Alpaca's IEX feed.
    This works on free tier — unlike SIP data.
    """
    try:
        ny = pytz.timezone("America/New_York")
        now = datetime.now(ny)

        # Extra buffer (Alpaca may return slightly fewer bars)
        start = now - timedelta(minutes=lookback_minutes + 30)
        end = now

        bars = api_market.get_bars(
            symbol,
            timeframe="1Min",
            start=start.isoformat(),
            end=end.isoformat(),
            feed="iex",          # 🚀 FREE tier OK
            adjustment="raw",
            limit=lookback_minutes + 30
        )

        if not bars:
            print(f"[WARN] Alpaca returned NO bars for {symbol}")
            return None

        df = pd.DataFrame([
            {
                "timestamp": b.t,
                "Open": float(b.o),
                "High": float(b.h),
                "Low": float(b.l),
                "Close": float(b.c),
                "Volume": int(b.v),
            }
            for b in bars
        ])

        if df.empty:
            print(f"[WARN] Empty Alpaca intraday DF for {symbol}")
            return None

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()

        # =====================================================
        # FIX: Remove deprecated `.last()` and "T"
        # =====================================================
        cutoff = df.index.max() - pd.Timedelta(minutes=lookback_minutes)
        df = df.loc[df.index >= cutoff]

        return df

    except Exception as e:
        print(f"[ERROR] Alpaca intraday fetch failed for {symbol}: {e}")
        return None


# ============================================================
# WRAPPER — compute_signals() calls this
# ============================================================
def fetch_intraday_history(symbol: str,
                           lookback_minutes: int = 120,
                           interval: str = "1m"):
    """
    Wrapper so compute_signals() does not need changes.
    """
    return fetch_intraday_history_alpaca(symbol, lookback_minutes)