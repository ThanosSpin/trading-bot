# features.py
import pandas as pd
import numpy as np
from time_features import add_time_features


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

        # If somehow a DataFrame, collapse to first column
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]

        vals = np.asarray(s)
        if vals.ndim > 1:
            s = pd.Series(vals.ravel(), index=df.index)

        df[col] = pd.to_numeric(s, errors="coerce")

    return df


# ============================================================================
# ADVANCED INDICATOR HELPERS (from advanced_features.py)
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
    denom = (high_max - low_min).replace(0, np.nan)   # ✅ avoid /0
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
    """On-Balance Volume."""
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"]).cumsum()


# ============================================================================
# TIME-OF-DAY FEATURES (INTRADAY)
# ============================================================================
def add_intraday_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclic time-of-day features for intraday data.
    Safe to call on daily; will no-op if index is not DatetimeIndex.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return df

    df = df.copy()
    df["hour"] = df.index.hour
    df["minute"] = df.index.minute

    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)

    return df


# ============================================================================
# ✅ NEW: CLEAN UNNAMED COLUMNS
# ============================================================================
def _remove_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove any unnamed or invalid column names.
    This prevents XGBoost errors with empty column names.
    """
    df = df.copy()
    
    # Find unnamed/empty columns
    unnamed_cols = []
    for col in df.columns:
        if col == '' or pd.isna(col) or str(col).strip() == '':
            unnamed_cols.append(col)
    
    if unnamed_cols:
        print(f"[WARN] Dropping {len(unnamed_cols)} unnamed columns from features")
        df = df.drop(columns=unnamed_cols)
    
    # Remove duplicate column names
    if df.columns.duplicated().any():
        duplicated = df.columns[df.columns.duplicated()].tolist()
        print(f"[WARN] Found duplicate columns: {duplicated}")
        df = df.loc[:, ~df.columns.duplicated()]
    
    return df


# ============================================================================
# ADVANCED FEATURE BUILDER
# ============================================================================
def add_advanced_features(df: pd.DataFrame, mode: str = "daily") -> pd.DataFrame:
    """
    Adds engineered features.
    Intraday uses shorter windows so we don't require huge history to produce a row.
    """
    df = df.copy()
    
    # ============================================================
    # ✅ NEW: USE ENHANCED INDICATORS FROM indicators.py
    # ============================================================
    from indicators import add_indicators
    
    print(f"[FEATURES] Adding enhanced indicators...")
    df = add_indicators(df)
    print(f"[FEATURES] ✅ Enhanced indicators added")
    
    # -----------------------------
    # Core returns / ratios
    # -----------------------------
    df["return_1"] = df["Close"].pct_change(fill_method=None)
    df["return_5"] = df["Close"].pct_change(5, fill_method=None)
    
    v = df["Volume"].replace(0, np.nan)
    df["volume_roc"] = v.pct_change(fill_method=None)
    
    df["close_open_ratio"] = df["Close"] / df["Open"]
    df["high_low_ratio"] = df["High"] / df["Low"]
    
    # Candle shape
    df["candle_body"] = df["Close"] - df["Open"]
    df["candle_range"] = df["High"] - df["Low"]
    
    # ✅ FIX: Use .values to prevent unnamed columns from max/min operations
    close_open_max = np.maximum(df["Close"].values, df["Open"].values)
    close_open_min = np.minimum(df["Close"].values, df["Open"].values)
    
    df["upper_wick"] = df["High"] - close_open_max
    df["lower_wick"] = close_open_min - df["Low"]
    
    cr = df["candle_range"].replace(0, np.nan)
    df["body_ratio"] = df["candle_body"] / cr
    df["upper_wick_ratio"] = df["upper_wick"] / cr
    df["lower_wick_ratio"] = df["lower_wick"] / cr
    
    # -----------------------------
    # Window choices by mode
    # -----------------------------
    if mode == "intraday":
        ma_windows = [3, 5, 10, 20]
        
        vol_windows = [3, 6, 12]
        warmup = 20
    else:
        ma_windows = [5, 10, 20, 50]
        
        vol_windows = [5, 15, 30]
        warmup = 80
    
    # -----------------------------
    # Additional moving averages (complement indicators.py)
    # -----------------------------
    for win in ma_windows:
        # Only add if not already present from indicators.py
        if f"ema_{win}" not in df.columns:
            df[f"ema_{win}"] = ema(df["Close"], win)
        if f"sma_{win}" not in df.columns:
            df[f"sma_{win}"] = sma(df["Close"], win)
    
    # ✅ REMOVED: RSI, MACD, Stochastic, Bollinger - now from indicators.py
    
    # -----------------------------
    # ATR + volatility (if not already present)
    # -----------------------------
    if "atr" not in df.columns:
        df["atr"] = atr(df, length=14)
    for w in vol_windows:
        df[f"vol_{w}"] = df["return_1"].rolling(w).std()
    
    # -----------------------------
    # OBV + lagged returns
    # -----------------------------
    if "obv" not in df.columns:
        df["obv"] = obv(df)
    
    for lag in [1, 2, 3, 4, 5]:
        df[f"lag_ret_{lag}"] = df["return_1"].shift(lag)
    
    # -----------------------------
    # Time-of-day (only meaningful intraday, safe otherwise)
    # -----------------------------
    df = add_intraday_time_features(df)
    
    # ========================================
    # ✅ ENHANCED: LEADING MOMENTUM FEATURES
    # ========================================
    
    # 1. Multi-period momentum (captures acceleration)
    for period in [3, 5, 10, 20]:
        df[f"momentum_{period}"] = df["Close"].pct_change(period, fill_method=None)
    
    # 2. Momentum acceleration (ROC of momentum = trend strength)
    df["momentum_accel_5"] = df["momentum_5"].diff()
    df["momentum_accel_10"] = df["momentum_10"].diff()
    
    # 3. Price position relative to recent range (0-100 scale)
    for window in [10, 20]:
        low_min = df["Low"].rolling(window).min()
        high_max = df["High"].rolling(window).max()
        range_denom = (high_max - low_min).replace(0, np.nan)
        df[f"price_position_{window}"] = ((df["Close"] - low_min) / range_denom * 100)
    
    # 4. Volume-weighted momentum (strong moves need volume)
    vol_norm = df["Volume"] / df["Volume"].rolling(20).mean()
    df["vol_momentum_5"] = df["momentum_5"] * vol_norm
    df["vol_momentum_10"] = df["momentum_10"] * vol_norm
    
    # 5. Breakout detection (price breaking above recent highs)
    
    high_10_max = df["High"].rolling(10).max().shift(1)
    high_20_max = df["High"].rolling(20).max().shift(1)
    df["breakout_10"] = (df["Close"] > high_10_max).astype(int)
    df["breakout_20"] = (df["Close"] > high_20_max).astype(int)
    
    # 6. Trend consistency (% of recent bars that are up)
    df["trend_consistency_10"] = (df["Close"] > df["Close"].shift(1)).rolling(10).mean()
    
    # =====================================================
    # Clean infinities / NaNs (but DON'T nuke everything)
    # =====================================================
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Always require core price columns
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    

    # --- Dynamic warmup ---

    if mode == "intraday":

        required = 20
        extra_buffer = 2
    else:
        required = 50
        extra_buffer = 10
    
    warmup = required + extra_buffer
    

    if len(df) <= warmup:
        return df.iloc[0:0]
    
    # Normal case: trim warmup then drop remaining NaNs
    df = df.iloc[warmup:].dropna()
    
    # ✅ CRITICAL: Remove any unnamed columns before returning
    df = _remove_unnamed_columns(df)
    
    return df



# ============================================================================
# BASE FEATURE ENTRY POINT
# ============================================================================
def _build_base_features(df: pd.DataFrame, mode: str = "daily") -> pd.DataFrame:
    df = _fix_ohlcv(df)
    df = add_advanced_features(df, mode=mode)
    
    # ✅ SAFETY: Final check for unnamed columns
    df = _remove_unnamed_columns(df)
    
    return df


# ============================================================================
# PUBLIC FUNCTIONS (used by model_xgb)
# ============================================================================
def build_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build full advanced feature set for daily data.
    """
    result = _build_base_features(df, mode="daily")
    
    # ✅ FINAL SAFETY CHECK
    result = _remove_unnamed_columns(result)
    
    return result


def build_intraday_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build full advanced feature set for intraday data.
    Uses same core features as daily, plus extra time-of-day encodings.
    """
    df = _build_base_features(df, mode="intraday")
    
    # ✅ ADD TIME FEATURES (only for intraday data with datetime index)
    if isinstance(df.index, pd.DatetimeIndex):
        df = add_time_features(df)
    
    # ✅ FINAL SAFETY CHECK
    df = _remove_unnamed_columns(df)
    
    return df


# ============================================================
# TIME-OF-DAY FEATURES (Enhanced version)
# ============================================================

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-of-day features that capture intraday behavioral patterns.

    Market microstructure insights:
    - Opening 30min: High volatility, news-driven, mean reversion
    - Midday (10:30-15:00): Lower volume, trend continuation
    - Closing hour: Volume surge, institutional flows, reversals

    Args:
        df: DataFrame with DatetimeIndex

    Returns:
        df with new time features added
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    
    df = df.copy()

    # Extract time components
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute

    # Time as decimal (9:30 = 9.5, 16:00 = 16.0)
    df['time_of_day'] = df['hour'] + df['minute'] / 60.0

    # Distance from market open (9:30 AM = 0 minutes)
    df['minutes_since_open'] = (df['time_of_day'] - 9.5) * 60
    df['minutes_since_open'] = df['minutes_since_open'].clip(lower=0)  # Pre-market = 0

    # Distance to market close (16:00)
    df['minutes_to_close'] = (16.0 - df['time_of_day']) * 60
    df['minutes_to_close'] = df['minutes_to_close'].clip(lower=0)  # After-hours = 0

    # Session phase (categorical → one-hot encoded)

    def get_session_phase(minutes_since_open):
        if minutes_since_open <= 30:
            return 'opening'
        elif minutes_since_open >= 330:  # 5.5 hours from open
            return 'closing'
        else:
            return 'midday'

    df['session_phase'] = df['minutes_since_open'].apply(get_session_phase)

    # One-hot encode session phase
    df['is_opening'] = (df['session_phase'] == 'opening').astype(int)
    df['is_midday'] = (df['session_phase'] == 'midday').astype(int)
    df['is_closing'] = (df['session_phase'] == 'closing').astype(int)

    # Non-linear time effects (capture U-shaped volatility pattern)

    df['time_squared'] = df['minutes_since_open'] ** 2
    df['time_cubed'] = df['minutes_since_open'] ** 3

    # Sine/cosine encoding (captures cyclical nature of trading day)

    df['time_sin'] = np.sin(2 * np.pi * df['minutes_since_open'] / 390)
    df['time_cos'] = np.cos(2 * np.pi * df['minutes_since_open'] / 390)

    # Lunch hour flag (12:00-13:00, low liquidity)
    df['is_lunch_hour'] = ((df['time_of_day'] >= 12.0) & (df['time_of_day'] <= 13.0)).astype(int)

    # First/last hour flags (extreme behavior periods)
    df['is_first_hour'] = (df['minutes_since_open'] <= 60).astype(int)
    df['is_last_hour'] = (df['minutes_to_close'] <= 60).astype(int)

    # Drop intermediate columns
    df = df.drop(columns=['hour', 'minute', 'session_phase'], errors='ignore')
    
    # ✅ SAFETY: Clean any unnamed columns that may have been created
    df = _remove_unnamed_columns(df)

    return df
