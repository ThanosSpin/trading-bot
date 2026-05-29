# patched_features.py
import pandas as pd
import numpy as np

from predictive_model.data_loader import fetch_historical_data
from predictive_model.time_features import add_time_features as canonical_add_time_features


# ============================================================================
# MULTIINDEX -> FLAT OHLCV COLUMN CLEANER
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
            chosen = None
            for part in col:
                if str(part).lower() in ["open", "high", "low", "close", "volume"]:
                    chosen = part
                    break
            if chosen is None:
                chosen = col[-1]
            new_cols.append(str(chosen).capitalize())
        df.columns = new_cols
    else:
        df.columns = [str(c).capitalize() for c in df.columns]

    rename_map = {
        "Adj close": "Close",
        "Adjclose": "Close",
        "Close*": "Close",
    }
    df.rename(columns=rename_map, inplace=True)
    return df


# ============================================================================
# Clean Columns
# ============================================================================
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean DataFrame column names by removing special characters.
    Ensures compatibility with XGBoost and other ML libraries.
    Handles both regular Index and MultiIndex columns.
    """
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(str(i) for i in col).strip("_") for col in df.columns]

    df.columns = df.columns.astype(str)
    for ch in ["[", "]", "<", ">", " ", ",", "(", ")"]:
        df.columns = df.columns.str.replace(ch, "_", regex=False)
    df.columns = df.columns.str.replace("__+", "_", regex=True)
    df.columns = df.columns.str.strip("_")
    return df


# ============================================================================
# FORCE ALL OHLCV TO 1D FLOAT SERIES
# ============================================================================
def _fix_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure we have numeric 1D Open/High/Low/Close/Volume columns.
    """
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
# CLEAN INVALID COLUMNS
# ============================================================================
def _remove_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove any unnamed or invalid column names and duplicates.
    """
    df = df.copy()
    unnamed_cols = []
    for col in df.columns:
        if col == "" or pd.isna(col) or str(col).strip() == "":
            unnamed_cols.append(col)

    if unnamed_cols:
        print(f"[WARN] Dropping {len(unnamed_cols)} unnamed columns from features")
        df = df.drop(columns=unnamed_cols)

    if df.columns.duplicated().any():
        duplicated = df.columns[df.columns.duplicated()].tolist()
        print(f"[WARN] Found duplicate columns: {duplicated}")
        df = df.loc[:, ~df.columns.duplicated()]

    return df


# ============================================================================
# ADVANCED INDICATOR HELPERS
# ============================================================================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(length).mean()
    loss = -delta.clip(upper=0).rolling(length).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series):
    fast = ema(close, 12)
    slow = ema(close, 26)
    macd_line = fast - slow
    signal = ema(macd_line, 9)
    hist = macd_line - signal
    return macd_line, signal, hist


def stoch(df: pd.DataFrame, k: int = 14, d: int = 3):
    low_min = df["Low"].rolling(k).min()
    high_max = df["High"].rolling(k).max()
    denom = (high_max - low_min).replace(0, np.nan)
    percent_k = (df["Close"] - low_min) / denom * 100
    percent_d = percent_k.rolling(d).mean()
    return percent_k, percent_d


def bollinger(close: pd.Series, length: int = 20, num_std: int = 2):
    mid = close.rolling(length).mean()
    std = close.rolling(length).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower


def atr(df: pd.DataFrame, length: int = 14):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(length).mean()


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"]).cumsum()


# ============================================================================
# FEATURE REGISTRY
# ============================================================================
def get_feature_groups(mode: str = "daily") -> dict:
    if mode == "intraday":
        ma_windows = [3, 5, 10, 20]
        vol_windows = [3, 6, 12]
        momentum_windows = [3, 5, 10, 20]
    else:
        ma_windows = [5, 10, 20, 50]
        vol_windows = [5, 15, 30]
        momentum_windows = [3, 5, 10, 20]

    return {
        "ma_windows": ma_windows,
        "vol_windows": vol_windows,
        "momentum_windows": momentum_windows,
        "lag_windows": [1, 2, 3, 4, 5],
        "range_windows": [10, 20],
    }


# ============================================================================
# CANONICAL FEATURE BUILDER
# ============================================================================
def add_advanced_features(df: pd.DataFrame, mode: str = "daily") -> pd.DataFrame:
    """
    Deterministic engineered features.
    Uses a single canonical time feature path and avoids duplicated feature definitions.
    """
    df = df.copy()
    groups = get_feature_groups(mode)

    print(f"[FEATURES] Building deterministic feature set for mode={mode}...")

    df["return_1"] = df["Close"].pct_change(fill_method=None)
    df["return_5"] = df["Close"].pct_change(5, fill_method=None)

    v = df["Volume"].replace(0, np.nan)
    df["volume_roc"] = v.pct_change(fill_method=None)

    df["close_open_ratio"] = df["Close"] / df["Open"]
    df["high_low_ratio"] = df["High"] / df["Low"]

    df["candle_body"] = df["Close"] - df["Open"]
    df["candle_range"] = df["High"] - df["Low"]

    close_open_max = np.maximum(df["Close"].values, df["Open"].values)
    close_open_min = np.minimum(df["Close"].values, df["Open"].values)
    df["upper_wick"] = df["High"] - close_open_max
    df["lower_wick"] = close_open_min - df["Low"]

    cr = df["candle_range"].replace(0, np.nan)
    df["body_ratio"] = df["candle_body"] / cr
    df["upper_wick_ratio"] = df["upper_wick"] / cr
    df["lower_wick_ratio"] = df["lower_wick"] / cr

    for win in groups["ma_windows"]:
        df[f"ema_{win}"] = ema(df["Close"], win)
        df[f"sma_{win}"] = sma(df["Close"], win)

    df["rsi_6"] = rsi(df["Close"], 6)
    df["rsi_14"] = rsi(df["Close"], 14)

    macd_line, macd_signal, macd_hist = macd(df["Close"])
    df["macd"] = macd_line
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist

    stoch_k, stoch_d = stoch(df)
    df["stoch_k"] = stoch_k
    df["stoch_d"] = stoch_d

    mid, upper, lower = bollinger(df["Close"])
    df["bb_mid"] = mid
    df["bb_upper"] = upper
    df["bb_lower"] = lower
    df["bb_width"] = (upper - lower) / mid.replace(0, np.nan)

    df["atr"] = atr(df, length=14)
    for w in groups["vol_windows"]:
        df[f"vol_{w}"] = df["return_1"].rolling(w).std()

    df["obv"] = obv(df)

    for lag in groups["lag_windows"]:
        df[f"lag_ret_{lag}"] = df["return_1"].shift(lag)

    for period in groups["momentum_windows"]:
        df[f"momentum_{period}"] = df["Close"].pct_change(period, fill_method=None)

    df["momentum_accel_5"] = df["momentum_5"].diff()
    df["momentum_accel_10"] = df["momentum_10"].diff()

    for window in groups["range_windows"]:
        low_min = df["Low"].rolling(window).min()
        high_max = df["High"].rolling(window).max()
        range_denom = (high_max - low_min).replace(0, np.nan)
        df[f"price_position_{window}"] = (df["Close"] - low_min) / range_denom * 100

    vol_norm = df["Volume"] / df["Volume"].rolling(20).mean()
    df["vol_momentum_5"] = df["momentum_5"] * vol_norm
    df["vol_momentum_10"] = df["momentum_10"] * vol_norm

    high_10_max = df["High"].rolling(10).max().shift(1)
    high_20_max = df["High"].rolling(20).max().shift(1)
    df["breakout_10"] = (df["Close"] > high_10_max).astype(int)
    df["breakout_20"] = (df["Close"] > high_20_max).astype(int)

    df["trend_consistency_10"] = (df["Close"] > df["Close"].shift(1)).rolling(10).mean()

    if mode == "intraday" and isinstance(df.index, pd.DatetimeIndex):
        df = canonical_add_time_features(df)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

    warmup = 22 if mode == "intraday" else 60
    if len(df) <= warmup:
        return df.iloc[0:0]

    df = df.iloc[warmup:].dropna()
    df = _remove_unnamed_columns(df)
    return df


# ============================================================================
# SPY REGIME FEATURES (MACRO CONTEXT)
# ============================================================================
def add_spy_regime_features(df: pd.DataFrame, spy_symbol: str = "SPY") -> pd.DataFrame:
    """
    Add SPY market regime features.
    """
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    df = df.copy()

    try:
        if df.attrs.get("interval", "1d") == "1d":
            spy_df = fetch_historical_data(spy_symbol, period="3mo", interval="1d")
        else:
            spy_df = fetch_historical_data(spy_symbol, period="1mo", interval="1d")

        if spy_df is None or spy_df.empty:
            print("[SPY-FEATURES] No SPY data available - skipping regime features")
            return df

        if isinstance(spy_df.index, pd.DatetimeIndex) and spy_df.index.tz is not None:
            spy_df.index = spy_df.index.tz_localize(None)
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        spy_raw = spy_df["Close"]
        if isinstance(spy_raw, pd.DataFrame):
            spy_close = spy_raw.iloc[:, 0]
        else:
            spy_close = spy_raw
        spy_close = pd.to_numeric(spy_close, errors="coerce").dropna()

        if hasattr(spy_close.index, "tz") and spy_close.index.tz is not None:
            spy_close.index = spy_close.index.tz_localize(None)

        if len(spy_close) < 20:
            print("[SPY-FEATURES] Insufficient SPY data - skipping")
            return df

        df["spy_ret10d"] = spy_close.pct_change(10).reindex(df.index, method="ffill").fillna(0)
        spy_ma20 = spy_close.rolling(20).mean()
        df["spy_above_ma20"] = (spy_close > spy_ma20).reindex(df.index, method="ffill").fillna(False).astype(int)
        spy_peak10 = spy_close.rolling(10, min_periods=1).max()
        df["spy_dd10d"] = (spy_close / spy_peak10 - 1).reindex(df.index, method="ffill").fillna(0)
        spy_vol10 = spy_close.pct_change().rolling(10).std()
        spy_vol_regime = spy_vol10 / spy_vol10.mean()
        df["spy_vol_regime"] = spy_vol_regime.reindex(df.index, method="ffill").fillna(1.0)
        df["spy_mom_accel"] = df["spy_ret10d"].diff().fillna(0)

        print(f"[SPY-FEATURES] Added 5 SPY regime features | spy_ret10d={df['spy_ret10d'].iloc[-1]:.2%}")
    except Exception as e:
        print(f"[SPY-FEATURES] Error adding regime features: {e}")

    return df


# ============================================================================
# FEATURE METADATA
# ============================================================================
def get_feature_schema(df: pd.DataFrame) -> dict:
    columns = list(df.columns)
    groups = {
        "price_action": [c for c in columns if c in [
            "return_1", "return_5", "close_open_ratio", "high_low_ratio",
            "candle_body", "candle_range", "upper_wick", "lower_wick",
            "body_ratio", "upper_wick_ratio", "lower_wick_ratio"
        ]],
        "trend": [c for c in columns if c.startswith(("ema_", "sma_", "momentum_", "breakout_", "trend_consistency_"))],
        "oscillators": [c for c in columns if c.startswith(("rsi_", "macd", "stoch_", "bb_"))],
        "volatility": [c for c in columns if c.startswith(("atr", "vol_"))],
        "volume": [c for c in columns if c.startswith(("volume_roc", "obv", "vol_momentum_"))],
        "lags": [c for c in columns if c.startswith("lag_ret_")],
        "time": [c for c in columns if c.startswith(("time_", "minutes_", "is_", "clock_"))],
        "market_regime": [c for c in columns if c.startswith("spy_")],
    }
    return {
        "feature_count": len(columns),
        "features": columns,
        "groups": groups,
    }


# ============================================================================
# BASE FEATURE ENTRY POINT
# ============================================================================
def _build_base_features(df: pd.DataFrame, mode: str = "daily") -> pd.DataFrame:
    df = _fix_ohlcv(df)
    df = add_advanced_features(df, mode=mode)
    df = _remove_unnamed_columns(df)
    return df


# ============================================================================
# PUBLIC FUNCTIONS (used by model_xgb)
# ============================================================================
def build_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    result = _build_base_features(df, mode="daily")
    result = add_spy_regime_features(result)
    result = _remove_unnamed_columns(result)
    result.attrs["feature_schema"] = get_feature_schema(result)
    return result


def build_intraday_features(df: pd.DataFrame) -> pd.DataFrame:
    result = _build_base_features(df, mode="intraday")
    result = add_spy_regime_features(result)
    result = _remove_unnamed_columns(result)
    result.attrs["feature_schema"] = get_feature_schema(result)
    return result
