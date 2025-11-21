# indicators.py
import pandas as pd
import numpy as np
import ta

# ============================================================
#  NORMALIZE OHLCV INPUT (force 1D numeric)
# ============================================================
def _fix_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures Open, High, Low, Close, Volume are 1D numeric columns.
    """
    df = df.copy()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            val = df[col]
            if isinstance(val, pd.DataFrame):
                val = val.iloc[:, 0]
            if isinstance(val.values, np.ndarray) and val.values.ndim == 2 and val.values.shape[1] == 1:
                val = pd.Series(val.values.ravel(), index=df.index)
            df[col] = pd.to_numeric(val, errors="coerce")
    return df

# ============================================================
#  ADD INDICATORS
# ============================================================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _fix_ohlcv(df)

    # Ensure 'Close' is 1D numeric
    close_series = pd.Series(df['Close'].values.ravel(), index=df.index).astype(float)

    # Momentum Indicators
    rsi = ta.momentum.RSIIndicator(close=close_series, window=14)
    df['rsi'] = rsi.rsi()

    macd = ta.trend.MACD(close=close_series, window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()

    # Trend Indicators
    ema14 = ta.trend.EMAIndicator(close=close_series, window=14)
    df['ema_14'] = ema14.ema_indicator()

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df