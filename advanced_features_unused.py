# ================================================================
# advanced_features.py
# Complete High-Signal Feature Engineering for Daily/Intraday ML
# ================================================================

import pandas as pd
import numpy as np
from features import (
    _fix_ohlcv,
    add_advanced_features,
    add_intraday_time_features,
    ema, sma, rsi, macd, stoch, bollinger, atr, obv,
)

# ------------------------------------------------------------
# Helper Indicators
# ------------------------------------------------------------
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def sma(series, length):
    return series.rolling(length).mean()

def rsi(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(length).mean()
    loss = -delta.clip(upper=0).rolling(length).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(close):
    fast = ema(close, 12)
    slow = ema(close, 26)
    macd_line = fast - slow
    signal = ema(macd_line, 9)
    hist = macd_line - signal
    return macd_line, signal, hist

def stoch(df, k=14, d=3):
    low_min = df["Low"].rolling(k).min()
    high_max = df["High"].rolling(k).max()
    denom = (high_max - low_min).replace(0, np.nan)
    percent_k = (df["Close"] - low_min) / denom * 100
    percent_d = percent_k.rolling(d).mean()
    return percent_k, percent_d

def bollinger(close, length=20, num_std=2):
    mid = close.rolling(length).mean()
    std = close.rolling(length).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower

def atr(df, length=14):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(length).mean()

def obv(df):
    """On-balance volume"""
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"]).cumsum()


# ------------------------------------------------------------
# Time-of-day features for intraday TFs
# ------------------------------------------------------------
def add_intraday_time_features(df):
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    df = df.copy()
    df["hour"] = df.index.hour
    df["minute"] = df.index.minute

    # Encode time cyclically: (for 1m/5m data)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)

    return df

def build_intraday_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _fix_ohlcv(df)
    df = add_advanced_features(df, mode="intraday")
    return df

# ------------------------------------------------------------
# Main Feature Builder
# ------------------------------------------------------------
def add_advanced_features(df: pd.DataFrame, mode: str = "daily") -> pd.DataFrame:
    """
    Adds 40+ engineered features.
    Safe for both Daily and Intraday datasets.
    """

    df = df.copy()

    # =====================================================
    # Price Normalizations
    # =====================================================
    df["return_1"] = df["Close"].pct_change()
    df["return_5"] = df["Close"].pct_change(5)
    df["volume_roc"] = df["Volume"].pct_change(fill_method=None)

    df["close_open_ratio"] = df["Close"] / df["Open"]
    df["high_low_ratio"] = df["High"] / df["Low"]

    # Candle shape
    df["candle_body"] = df["Close"] - df["Open"]
    df["candle_range"] = df["High"] - df["Low"]
    df["upper_wick"] = df["High"] - df[["Close", "Open"]].max(axis=1)
    df["lower_wick"] = df[["Close", "Open"]].min(axis=1) - df["Low"]

    cr = df["candle_range"].replace(0, np.nan)
    df["body_ratio"] = df["candle_body"] / cr
    df["upper_wick_ratio"] = df["upper_wick"] / cr
    df["lower_wick_ratio"] = df["lower_wick"] / cr

    # =====================================================
    # Moving Averages (EMAs + SMAs)
    # =====================================================
    # Choose windows based on timeframe (intraday needs shorter windows)
    if mode == "intraday":
        ma_windows = [3, 5, 10, 20]   # ✅ no 50 on intraday
    else:
        ma_windows = [5, 10, 20, 50]

    for win in ma_windows:
        df[f"ema_{win}"] = ema(df["Close"], win)
        df[f"sma_{win}"] = sma(df["Close"], win)

    # =====================================================
    # RSI
    # =====================================================
    if mode == "intraday":
        df["rsi_6"] = rsi(df["Close"], 6)
        df["rsi_10"] = rsi(df["Close"], 10)
    else:
        df["rsi_6"] = rsi(df["Close"], 6)
        df["rsi_14"] = rsi(df["Close"], 14)

    # =====================================================
    # MACD
    # =====================================================
    if mode == "intraday":
        # faster MACD for intraday
        fast = ema(df["Close"], 8)
        slow = ema(df["Close"], 18)
        macd_line = fast - slow
        macd_signal = ema(macd_line, 6)
        macd_hist = macd_line - macd_signal
    else:
        macd_line, macd_signal, macd_hist = macd(df["Close"])

    df["macd"] = macd_line
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist

    # =====================================================
    # Stochastic
    # =====================================================
    stoch_k, stoch_d = stoch(df)
    df["stoch_k"] = stoch_k
    df["stoch_d"] = stoch_d

    # =====================================================
    # Bollinger Bands
    # =====================================================
    mid, upper, lower = bollinger(df["Close"])
    df["bb_mid"] = mid
    df["bb_upper"] = upper
    df["bb_lower"] = lower
    df["bb_width"] = (upper - lower) / mid

    # =====================================================
    # ATR + Volatility
    # =====================================================
    df["atr"] = atr(df)
    if mode == "intraday":
        df["vol_3"] = df["return_1"].rolling(3).std()
        df["vol_6"] = df["return_1"].rolling(6).std()
        df["vol_12"] = df["return_1"].rolling(12).std()
    else:
        df["vol_5"] = df["return_1"].rolling(5).std()
        df["vol_15"] = df["return_1"].rolling(15).std()
        df["vol_30"] = df["return_1"].rolling(30).std()

    # =====================================================
    # OBV
    # =====================================================
    df["obv"] = obv(df)

    # =====================================================
    # Lagged Returns
    # =====================================================
    for lag in [1, 2, 3, 4, 5]:
        df[f"lag_ret_{lag}"] = df["return_1"].shift(lag)

    # =====================================================
    # Intraday encodings (only if intraday index)
    # =====================================================
    df = add_intraday_time_features(df)

    # =====================================================
    # Clean infinities / NaNs (but DON'T nuke everything)
    # =====================================================
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Always require core price columns
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

    # Warmup cut (rolling features need some bars first)
    # Intraday uses smaller windows, daily usesZ uses larger.
    warmup = 30 if mode == "intraday" else 80

    # If not enough rows to survive warmup, return empty safely
    if len(df) <= warmup:
        return df.iloc[0:0]

    # Trim warmup region first, THEN drop remaining NaNs
    df = df.iloc[warmup:].dropna()

    return df