# data_loader.py
import pandas as pd
import pytz
from datetime import datetime, timedelta
from typing import Optional
import yfinance as yf
from broker import api_market


# ============================================================
# DAILY DATA (Yahoo Finance)
# ============================================================
def fetch_historical_data(
    symbol: str,
    years: Optional[int] = None,
    period: Optional[str] = None,
    interval: str = "1d",
) -> Optional[pd.DataFrame]:
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
    try:
        data = yf.download(symbol, period="1d", interval="1m",
                           progress=False, auto_adjust=True)
        if data.empty:
            return None

        last_close = data["Close"].iloc[-1]
        if isinstance(last_close, pd.Series):
            last_close = last_close.iloc[0]
        return float(last_close)

    except Exception as e:
        print(f"[ERROR] fetch_latest_price {symbol}: {e}")
        return None


# ============================================================
# INTRADAY — Alpaca (IEX feed)
# ============================================================
def _alpaca_timeframe_from_str(interval: str) -> str:
    s = str(interval).lower().strip()
    if s in ("15m", "15min", "15mins"):
        return "15Min"
    if s in ("5m", "5min", "5mins"):
        return "5Min"
    if s in ("1m", "1min", "1mins"):
        return "1Min"
    return "15Min"


def _estimate_limit(lookback_minutes: int, timeframe: str) -> int:
    # limit is number of bars, not minutes
    if timeframe == "15Min":
        return int((lookback_minutes / 15) + 200)
    if timeframe == "5Min":
        return int((lookback_minutes / 5) + 400)
    return int(lookback_minutes + 800)


def fetch_intraday_history_alpaca(
    symbol: str,
    lookback_minutes: int = 900,
    interval: str = "15min",
) -> Optional[pd.DataFrame]:
    """
    Fetch intraday bars from Alpaca IEX feed at the requested interval.
    Uses a calendar-time window wide enough to span multiple sessions.
    """
    try:
        timeframe = _alpaca_timeframe_from_str(interval)

        # Use UTC consistently for Alpaca requests
        now_utc = datetime.now(pytz.UTC)

        # Important: minutes-based lookback doesn't map cleanly to trading sessions.
        # Add a calendar buffer so we actually cross multiple days.
        # Example: 2400 minutes (~1.6 days) may still return only one session if you
        # start mid-day; buffer ensures multiple sessions are included.
        buffer_days = max(2, int((lookback_minutes / 390) + 2))  # 390 mins = 1 session
        start_utc = now_utc - timedelta(days=buffer_days)

        limit = _estimate_limit(lookback_minutes, timeframe)

        bars = api_market.get_bars(
            symbol,
            timeframe=timeframe,
            start=start_utc.isoformat(),
            end=now_utc.isoformat(),
            feed="iex",
            adjustment="raw",
            limit=limit,
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
            "Volume": float(b.v),
        } for b in bars])

        if df.empty:
            return None

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()

        # Trim by bars instead of minutes (trading-time aware)
        if timeframe == "15Min":
            bars_needed = int(lookback_minutes / 15) + 10
        elif timeframe == "5Min":
            bars_needed = int(lookback_minutes / 5) + 20
        else:
            bars_needed = int(lookback_minutes) + 50

        df = df.tail(bars_needed)

        return df

    except Exception as e:
        print(f"[ERROR] Alpaca intraday fetch failed for {symbol}: {e}")
        return None


# ============================================================
# WRAPPER — compute_signals() calls this
# ============================================================
def fetch_intraday_history(
    symbol: str,
    lookback_minutes: int = 900,
    interval: str = "15min",
) -> Optional[pd.DataFrame]:
    """
    Single wrapper used by compute_signals().
    Tries Alpaca at requested interval; if empty, falls back to yfinance.
    """
    df = fetch_intraday_history_alpaca(symbol, lookback_minutes=lookback_minutes, interval=interval)
    if df is not None and not df.empty:
        return df

    # fallback: yfinance
    try:
        yf_interval = "15m" if str(interval).lower() in ("15m", "15min", "15mins") else "1m"
        yf_period = "60d" if yf_interval == "15m" else "7d"

        df_y = yf.download(symbol, period=yf_period, interval=yf_interval,
                           progress=False, auto_adjust=True)
        if df_y is None or df_y.empty:
            return None

        df_y = df_y.copy()
        df_y.index = pd.to_datetime(df_y.index, utc=True, errors="coerce")
        df_y = df_y.sort_index()

        cutoff = df_y.index.max() - pd.Timedelta(minutes=lookback_minutes)
        df_y = df_y.loc[df_y.index >= cutoff]

        return df_y

    except Exception:
        return None