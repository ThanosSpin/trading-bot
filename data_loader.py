# data_loader.py
import os
import time
import pandas as pd
import pytz
from datetime import datetime, timedelta
from typing import Optional
import yfinance as yf
from broker import api_market
from alpaca_client import api as alpaca_api


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
            return None
        df = pd.read_parquet(path)
        return df if (df is not None and not df.empty) else None
    except Exception:
        return None

def _write_cache(path: str, df: pd.DataFrame) -> None:
    try:
        if df is None or df.empty:
            return
        df.to_parquet(path)
    except Exception:
        pass

def fetch_historical_data(
    symbol: str,
    years: Optional[int] = None,
    period: Optional[str] = None,
    interval: str = "1d",
) -> Optional[pd.DataFrame]:
    """
    Yahoo daily/intraday fetch with:
      - disk cache (parquet)
      - retry + exponential backoff
    """
    symbol = str(symbol).upper().strip()

    if not period and not years:
        raise ValueError("Either 'years' or 'period' must be provided.")

    # cache TTL
    if interval == "1d":
        ttl_sec = 6 * 60 * 60
    else:
        ttl_sec = 5 * 60

    cpath = _cache_path(symbol, period, years, interval)

    # 1) Try cache first
    cached = _read_cache(cpath, ttl_sec)
    if cached is not None:
        # ✅ ADDED: Ensure cached data is also clean
        if isinstance(cached.columns, pd.MultiIndex):
            cached.columns = cached.columns.get_level_values(0)
        return cached

    # 2) Fetch with retries/backoff
    last_err = None
    for attempt in range(5):
        try:
            if period:
                df = yf.download(
                    symbol,
                    period=period,
                    interval=interval,
                    progress=False,
                    auto_adjust=True,
                    threads=False,
                )
            else:
                df = yf.download(
                    symbol,
                    period=f"{years}y",
                    interval=interval,
                    progress=False,
                    auto_adjust=True,
                    threads=False,
                )

            if df is None or df.empty:
                print(f"[WARN] Empty data for {symbol} (period={period or str(years)+'y'}, interval={interval})")
                return None

            # ✅ ADDED: Fix MultiIndex if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Save cache and return
            _write_cache(cpath, df)
            return df

        except Exception as e:
            last_err = e
            
            sleep_s = 2 ** (attempt + 1)
            print(f"[WARN] Daily data fetch failed for {symbol} (attempt {attempt+1}/5): {e} — sleeping {sleep_s}s")
            time.sleep(sleep_s)

    print(f"[ERROR] Daily data fetch failed for {symbol} after retries: {last_err}")
    return None


# ============================================================
# LATEST PRICE (Yahoo Finance fallback)
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
    
    print(f"[DEBUG fetch_latest_price] Called for {sym} (prefer_yfinance={prefer_yfinance})")
    
    # During pre-market, Alpaca often returns stale data (yesterday's close)
    # Skip to yfinance for real-time pre-market prices
    if not prefer_yfinance:
        # 1) Try Alpaca first (no Yahoo rate limits)
        try:
            print(f"[DEBUG] Trying Alpaca API for {sym}...")
            bar = alpaca_api.get_latest_bar(sym)
            
            if bar:
                # Check bar timestamp to see if it's stale
                bar_time = getattr(bar, 't', None)
                if bar_time:
                    from datetime import datetime, timezone
                    now_utc = datetime.now(timezone.utc)
                    age_hours = (now_utc - bar_time).total_seconds() / 3600
                    print(f"[DEBUG] Alpaca bar timestamp: {bar_time}")
                    print(f"[DEBUG] Bar age: {age_hours:.1f} hours")
                    
                    # If bar is more than 12 hours old, it's from yesterday
                    if age_hours > 12:
                        print(f"[WARN] Alpaca bar is {age_hours:.1f}h old (stale) - skipping to yfinance")
                    else:
                        # Fresh data from Alpaca
                        if getattr(bar, "c", None) is not None:
                            price = float(bar.c)
                            print(f"[DEBUG] ✅ Alpaca price (bar.c): ${price:.2f}")
                            return price
                        
                        if getattr(bar, "close", None) is not None:
                            price = float(bar.close)
                            print(f"[DEBUG] ✅ Alpaca price (bar.close): ${price:.2f}")
                            return price
                else:
                    # No timestamp - try to use it anyway
                    if getattr(bar, "c", None) is not None:
                        price = float(bar.c)
                        print(f"[DEBUG] ⚠️ Alpaca price (no timestamp): ${price:.2f}")
                        return price
                    
                    if getattr(bar, "close", None) is not None:
                        price = float(bar.close)
                        print(f"[DEBUG] ⚠️ Alpaca price (no timestamp): ${price:.2f}")
                        return price
                
                print(f"[WARN] Alpaca bar exists but has no .c or .close")
            else:
                print(f"[WARN] Alpaca returned None/empty bar")
                
        except Exception as e:
            print(f"[WARN] Alpaca latest price failed for {sym}: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"[DEBUG] Skipping Alpaca (prefer_yfinance=True)")

    # 2) Fallback: yfinance (real-time data, works in pre-market)
    try:
        print(f"[DEBUG] Trying yfinance for {sym}...")
        
        # Use 1d period, 1m interval to get most recent tick
        data = yf.download(sym, period="1d", interval="1m", progress=False, auto_adjust=True)
        
        if data is None or data.empty:
            print(f"[WARN] yfinance returned None/empty for {sym}")
            return None
        
        print(f"[DEBUG] yfinance returned {len(data)} bars")
        
        # Get last close price
        last_close = data["Close"].iloc[-1]
        
        # Handle MultiIndex (sometimes yfinance returns this)
        if isinstance(last_close, pd.Series):
            last_close = last_close.iloc[0]
        
        price = float(last_close)
        
        # Get timestamp of last bar
        last_time = data.index[-1]
        print(f"[DEBUG] Last bar timestamp: {last_time}")
        print(f"[DEBUG] ✅ yfinance price: ${price:.2f}")
        
        return price
    
    except Exception as e:
        print(f"[ERROR] yfinance fetch_latest_price for {sym}: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    print(f"[ERROR] All methods failed for {sym}")
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