# indicators.py
import pandas as pd
import ta

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure 'Close' is a 1D float Series
    close_series = pd.Series(df['Close'].values.flatten(), index=df.index).astype(float)

    # Momentum Indicators
    rsi = ta.momentum.RSIIndicator(close=close_series)
    df['rsi'] = rsi.rsi()

    macd = ta.trend.MACD(close=close_series)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()

    # Trend Indicators
    ema = ta.trend.EMAIndicator(close=close_series, window=14)
    df['ema_14'] = ema.ema_indicator()

    df = df.dropna()
    return df
