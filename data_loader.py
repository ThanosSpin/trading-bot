# data_loader.py
import os
import time
import logging
import pandas as pd
import pytz
from datetime import datetime, timedelta
from typing import Optional
import yfinance as yf
from broker import api_market
from alpaca_client import api as alpaca_api

# ============================================================
# LOGGING CONFIGURATION
# ============================================================
logger = logging.getLogger(__name__)

# ============================================================
# DAILY DATA (Yahoo Finance)
# ============================================================
_CACHE_DIR = os.path.join(os.path.dirname(__file__), "data_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)


def _cache_path(symbol: str, period: Optional[str], years: Optional[int], interval: str) -> str:
    sym = str(symbol).upper().strip()
    p = period if period else f"{years}y"
    safe = f"{sym}_{p}_{interval}".replace("/", "_").replace(" ", "")
    return os.path.join(_CACHE_DIR, f"{safe}.parquet")


def _read_cache(path: str, max_age_sec: int) -> Optional[pd.DataFrame]:
    try:
        if not os.path.exists(path):
            return None
        age = time.time() - os.path.getmtime(path)
        if age > max_age_sec:
            logger.debug(f"Cache expired (age: {age:.0f}s > {max_age_sec}s): {path}")
            return None
        df = pd.read_parquet(path)
        logger.debug(f"Cache hit: {path}")
        return df if (df is not None and not df.empty) else None
    except Exception as e:
        logger.warning(f"Cache read failed: {path} - {e}")
        return None


def _write_cache(path: str, df: pd.DataFrame) -> None:
    try:
        if df is None or df.empty:
            return
        df.to_parquet(path)
        logger.debug(f"Cache written: {path}")
    except Exception as e:
        logger.warning(f"Cache write failed: {path} - {e}")


def fetch_historical_data(
    symbol: str,
    years: Optional[int] = None,
    period: Optional[str] = None,
    interval: str = "1d",
) -> Optional[pd.DataFrame]:
    """
    Yahoo daily/intraday fetch with disk cache (parquet) and retry with exponential backoff.
    """
    symbol = str(symbol).upper().strip()

    if not period and not years:
        raise ValueError("Either 'years' or 'period' must be provided.")

    # Cache TTL
    ttl_sec = 6 * 60 * 60 if interval == "1d" else 5 * 60
    cpath = _cache_path(symbol, period, years, interval)

    # Try cache first
    cached = _read_cache(cpath, ttl_sec)
    if cached is not None:
        if isinstance(cached.columns, pd.MultiIndex):
            cached.columns = cached.columns.get_level_values(0)
        return cached

    # Fetch with retries/backoff
    last_err = None
    for attempt in range(5):
        try:
            period_str = period or f"{years}y"
            df = yf.download(
                symbol,
                period=period_str,
                interval=interval,
                progress=False,
                auto_adjust=True,
                threads=False,
            )

            if df is None or df.empty:
                logger.warning(f"Empty data for {symbol} (period={period_str}, interval={interval})")
                return None

            # Fix MultiIndex if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            _write_cache(cpath, df)
            logger.info(f"Fetched historical data for {symbol}: {len(df)} rows")
            return df

        except Exception as e:
            last_err = e
            sleep_s = 2 ** (attempt + 1)
            logger.warning(
                f"Historical data fetch failed for {symbol} (attempt {attempt+1}/5): {e} - retrying in {sleep_s}s"
            )
            time.sleep(sleep_s)

    logger.error(f"Historical data fetch failed for {symbol} after 5 retries: {last_err}")
    return None


# ============================================================
# LATEST PRICE
# ============================================================
def fetch_latest_price(symbol: str, prefer_yfinance=False) -> Optional[float]:
    """
    Fetch the latest real-time price for a symbol.
    
    Args:
        symbol: Stock symbol
        prefer_yfinance: If True, skip Alpaca and use yfinance (better for pre-market)
    
    Returns:
        Latest price or None
    """
    sym = str(symbol).upper().strip()
    logger.debug(f"Fetching latest price for {sym} (prefer_yfinance={prefer_yfinance})")
    
    # Try Alpaca first unless explicitly preferring yfinance
    if not prefer_yfinance:
        try:
            logger.debug(f"Attempting Alpaca API for {sym}")
            bar = alpaca_api.get_latest_bar(sym)
            
            if bar:
                bar_time = getattr(bar, 't', None)
                if bar_time:
                    now_utc = datetime.now(pytz.UTC)
                    age_hours = (now_utc - bar_time).total_seconds() / 3600
                    
                    logger.debug(f"Alpaca bar for {sym}: timestamp={bar_time}, age={age_hours:.1f}h")
                    
                    if age_hours > 12:
                        logger.warning(f"Alpaca bar for {sym} is stale ({age_hours:.1f}h old) - falling back to yfinance")
                    else:
                        # Fresh data from Alpaca
                        price = getattr(bar, "c", None) or getattr(bar, "close", None)
                        if price is not None:
                            price = float(price)
                            logger.info(f"Alpaca price for {sym}: ${price:.2f}")
                            return price
                else:
                    # No timestamp - try anyway
                    price = getattr(bar, "c", None) or getattr(bar, "close", None)
                    if price is not None:
                        price = float(price)
                        logger.info(f"Alpaca price for {sym}: ${price:.2f} (no timestamp)")
                        return price
                
                logger.warning(f"Alpaca bar for {sym} has no valid price field")
            else:
                logger.warning(f"Alpaca returned empty bar for {sym}")
                
        except Exception as e:
            logger.error(f"Alpaca API failed for {sym}: {e}", exc_info=True)
    else:
        logger.debug(f"Skipping Alpaca (prefer_yfinance=True)")

    # Fallback to yfinance
    try:
        logger.debug(f"Attempting yfinance for {sym}")
        data = yf.download(sym, period="1d", interval="1m", progress=False, auto_adjust=True)
        
        if data is None or data.empty:
            logger.warning(f"yfinance returned empty data for {sym}")
            return None
        
        logger.debug(f"yfinance returned {len(data)} bars for {sym}")
        
        last_close = data["Close"].iloc[-1]
        if isinstance(last_close, pd.Series):
            last_close = last_close.iloc[0]
        
        price = float(last_close)
        last_time = data.index[-1]
        
        logger.info(f"yfinance price for {sym}: ${price:.2f} at {last_time}")
        return price
    
    except Exception as e:
        logger.error(f"yfinance failed for {sym}: {e}", exc_info=True)
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
    """
    try:
        timeframe = _alpaca_timeframe_from_str(interval)
        now_utc = datetime.now(pytz.UTC)
        
        buffer_days = max(2, int((lookback_minutes / 390) + 2))
        start_utc = now_utc - timedelta(days=buffer_days)
        limit = _estimate_limit(lookback_minutes, timeframe)
        
        logger.debug(
            f"Fetching Alpaca intraday for {symbol}: "
            f"timeframe={timeframe}, lookback={lookback_minutes}min, limit={limit}"
        )

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
            logger.warning(f"Alpaca returned no intraday bars for {symbol}")
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

        # Trim by bars
        if timeframe == "15Min":
            bars_needed = int(lookback_minutes / 15) + 10
        elif timeframe == "5Min":
            bars_needed = int(lookback_minutes / 5) + 20
        else:
            bars_needed = int(lookback_minutes) + 50

        df = df.tail(bars_needed)
        logger.info(f"Alpaca intraday data for {symbol}: {len(df)} bars")
        return df

    except Exception as e:
        logger.error(f"Alpaca intraday fetch failed for {symbol}: {e}", exc_info=True)
        return None


# ============================================================
# WRAPPER
# ============================================================
def fetch_intraday_history(
    symbol: str,
    lookback_minutes: int = 900,
    interval: str = "15min",
) -> Optional[pd.DataFrame]:
    """
    Wrapper used by compute_signals(). Tries Alpaca first, then yfinance fallback.
    """
    df = fetch_intraday_history_alpaca(symbol, lookback_minutes=lookback_minutes, interval=interval)
    if df is not None and not df.empty:
        return df

    # Fallback to yfinance
    try:
        yf_interval = "15m" if str(interval).lower() in ("15m", "15min", "15mins") else "1m"
        yf_period = "60d" if yf_interval == "15m" else "7d"
        
        logger.debug(f"Falling back to yfinance for {symbol}: period={yf_period}, interval={yf_interval}")

        df_y = yf.download(symbol, period=yf_period, interval=yf_interval,
                           progress=False, auto_adjust=True)
        if df_y is None or df_y.empty:
            logger.warning(f"yfinance intraday returned empty for {symbol}")
            return None

        df_y = df_y.copy()
        df_y.index = pd.to_datetime(df_y.index, utc=True, errors="coerce")
        df_y = df_y.sort_index()

        cutoff = df_y.index.max() - pd.Timedelta(minutes=lookback_minutes)
        df_y = df_y.loc[df_y.index >= cutoff]
        
        logger.info(f"yfinance intraday data for {symbol}: {len(df_y)} bars")
        return df_y

    except Exception as e:
        logger.error(f"yfinance intraday fetch failed for {symbol}: {e}", exc_info=True)
        return None
