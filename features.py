# features.py
import pandas as pd
import numpy as np
import ta

# ============================================================================
# MULTIINDEX → FLAT OHLCV COLUMN CLEANER
# ============================================================================
def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fully flattens yfinance MultiIndex columns.
    Extracts correct OHLCV names regardless of structure.
    Resulting columns will always be: Open, High, Low, Close, Volume
    """

    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for col in df.columns:
            # col is a tuple like ("Price","Open") or ("Price","Close")
            chosen = None
            for part in col:
                if str(part).lower() in ["open", "high", "low", "close", "volume"]:
                    chosen = part
                    break
            if chosen is None:
                chosen = col[-1]  # fallback
            new_cols.append(str(chosen).capitalize())

        df.columns = new_cols
    else:
        # Normalize to capitalized OHLCV names
        df.columns = [str(c).capitalize() for c in df.columns]

    rename_map = {
        "Adj close": "Close",
        "Adjclose": "Close",
        "Close*": "Close",
    }
    df.rename(columns=rename_map, inplace=True)

    return df


# ============================================================================
# FORCE ALL OHLCV TO 1D FLOAT SERIES
# ============================================================================
def _fix_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _clean_columns(df)

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"[FATAL] Missing required column: {col}")

        s = df[col]

        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]

        vals = np.asarray(s)
        if vals.ndim > 1:
            s = pd.Series(vals.ravel(), index=df.index)

        df[col] = pd.to_numeric(s, errors="coerce")

    return df


# ============================================================================
# BASE FEATURES
# ============================================================================
def _build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _fix_ohlcv(df)

    df["returns"] = df["Close"].pct_change()
    df["log_returns"] = np.log(df["Close"]).diff()

    # MOMENTUM
    df["rsi_14"] = ta.momentum.RSIIndicator(df["Close"], 14).rsi()
    stoch = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"], 14, 3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    macd = ta.trend.MACD(df["Close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # TREND
    df["ema_9"] = ta.trend.EMAIndicator(df["Close"], 9).ema_indicator()
    df["ema_21"] = ta.trend.EMAIndicator(df["Close"], 21).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(df["Close"], 50).ema_indicator()
    adx = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], 14)
    df["adx_14"] = adx.adx()

    # VOLATILITY
    boll = ta.volatility.BollingerBands(df["Close"])
    df["bb_high"] = boll.bollinger_hband()
    df["bb_low"] = boll.bollinger_lband()
    atr = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"])
    df["atr_14"] = atr.average_true_range()

    # VOLUME
    df["vol_roc"] = df["Volume"].pct_change()

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


# ============================================================================
# PUBLIC FUNCTIONS
# ============================================================================
def build_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _clean_columns(df)
    return _build_base_features(df)


def build_intraday_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _clean_columns(df)
    df = _build_base_features(df)
    df["time_sin"] = np.sin(np.linspace(0, 2 * np.pi, len(df)))
    df["time_cos"] = np.cos(np.linspace(0, 2 * np.pi, len(df)))
    return df