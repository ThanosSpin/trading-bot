# indicators.py
import pandas as pd
import numpy as np
import ta

# ============================================================
# NORMALIZE OHLCV INPUT (force 1D numeric)
# ============================================================
def _fix_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures Open, High, Low, Close, Volume are 1D numeric columns.
    """
    df = df.copy()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            val = df[col]
            # If it's a DataFrame, extract first column
            if isinstance(val, pd.DataFrame):
                val = val.iloc[:, 0]
            # If values are 2D array, flatten
            if hasattr(val, 'values') and isinstance(val.values, np.ndarray):
                if val.values.ndim == 2 and val.values.shape[1] == 1:
                    val = pd.Series(val.values.ravel(), index=df.index)
            # Convert to numeric
            df[col] = pd.to_numeric(val, errors="coerce")
    return df


def _ensure_series(data, index=None) -> pd.Series:
    """
    Convert data to 1D pandas Series.
    
    Args:
        data: Can be Series, DataFrame column, or array
        index: Optional index to use
        
    Returns:
        pd.Series (guaranteed 1D)
    """
    # Already a Series
    if isinstance(data, pd.Series):
        return data
    
    # DataFrame - extract first column
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]
    
    # NumPy array - flatten if needed
    if isinstance(data, np.ndarray):
        if data.ndim == 2:
            data = data.ravel()
    
    # Convert to Series
    if index is not None:
        return pd.Series(data, index=index)
    else:
        return pd.Series(data)


# ============================================================
# ADD INDICATORS (ENHANCED & FIXED)
# ============================================================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add comprehensive technical indicators including:
    - Momentum: RSI, MACD, Stochastic
    - Trend: EMA14, ADX, Aroon
    - Volatility: Bollinger Bands, ATR, Keltner Channels
    - Volume: OBV, CMF
    """
    df = df.copy()
    df = _fix_ohlcv(df)
    
    # Ensure 1D numeric series using helper function
    close_series = _ensure_series(df['Close'], index=df.index).astype(float)
    high_series = _ensure_series(df['High'], index=df.index).astype(float)
    low_series = _ensure_series(df['Low'], index=df.index).astype(float)
    volume_series = _ensure_series(df['Volume'], index=df.index).astype(float)
    
    # ============================================================
    # MOMENTUM INDICATORS
    # ============================================================
    
    try:
        # RSI (14-period)
        rsi = ta.momentum.RSIIndicator(close=close_series, window=14)
        df['rsi'] = rsi.rsi()
    except Exception as e:
        print(f"[WARN] RSI failed: {e}")
        df['rsi'] = np.nan
    
    try:
        # MACD (12, 26, 9)
        macd = ta.trend.MACD(close=close_series, window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()
    except Exception as e:
        print(f"[WARN] MACD failed: {e}")
        df['macd'] = np.nan
        df['macd_signal'] = np.nan
        df['macd_hist'] = np.nan
    
    try:
        # Stochastic Oscillator (14-period)
        stoch = ta.momentum.StochasticOscillator(
            high=high_series, 
            low=low_series, 
            close=close_series, 
            window=14, 
            smooth_window=3
        )
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
    except Exception as e:
        print(f"[WARN] Stochastic failed: {e}")
        df['stoch_k'] = np.nan
        df['stoch_d'] = np.nan
    
    try:
        # Rate of Change (10-period)
        roc = ta.momentum.ROCIndicator(close=close_series, window=10)
        df['roc'] = roc.roc()
    except Exception as e:
        print(f"[WARN] ROC failed: {e}")
        df['roc'] = np.nan
    
    # ============================================================
    # TREND INDICATORS
    # ============================================================
    
    try:
        # EMA (multiple periods)
        ema14 = ta.trend.EMAIndicator(close=close_series, window=14)
        df['ema_14'] = ema14.ema_indicator()
        
        ema50 = ta.trend.EMAIndicator(close=close_series, window=50)
        df['ema_50'] = ema50.ema_indicator()
    except Exception as e:
        print(f"[WARN] EMA failed: {e}")
        df['ema_14'] = np.nan
        df['ema_50'] = np.nan
    
    try:
        # SMA 20 for Bollinger Bands
        sma20 = ta.trend.SMAIndicator(close=close_series, window=20)
        df['sma_20'] = sma20.sma_indicator()
    except Exception as e:
        print(f"[WARN] SMA failed: {e}")
        df['sma_20'] = np.nan
    
    try:
        # ADX (Average Directional Index) - trend strength
        adx = ta.trend.ADXIndicator(high=high_series, low=low_series, close=close_series, window=14)
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()  # +DI
        df['adx_neg'] = adx.adx_neg()  # -DI
    except Exception as e:
        print(f"[WARN] ADX failed: {e}")
        df['adx'] = np.nan
        df['adx_pos'] = np.nan
        df['adx_neg'] = np.nan
    
    try:
        # Aroon Indicator (trend direction)
        aroon = ta.trend.AroonIndicator(high=high_series, low=low_series, window=25)
        df['aroon_up'] = aroon.aroon_up()
        df['aroon_down'] = aroon.aroon_down()
        df['aroon_indicator'] = aroon.aroon_indicator()  # up - down
    except Exception as e:
        print(f"[WARN] Aroon failed: {e}")
        df['aroon_up'] = np.nan
        df['aroon_down'] = np.nan
        df['aroon_indicator'] = np.nan
    
    # ============================================================
    # VOLATILITY INDICATORS
    # ============================================================
    
    try:
        # Bollinger Bands (20, 2)
        bollinger = ta.volatility.BollingerBands(close=close_series, window=20, window_dev=2)
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_mid'] = bollinger.bollinger_mavg()
        df['bb_lower'] = bollinger.bollinger_lband()
        df['bb_width'] = bollinger.bollinger_wband()  # bandwidth
        df['bb_pct'] = bollinger.bollinger_pband()    # %B (position within bands)
    except Exception as e:
        print(f"[WARN] Bollinger Bands failed: {e}")
        df['bb_upper'] = np.nan
        df['bb_mid'] = np.nan
        df['bb_lower'] = np.nan
        df['bb_width'] = np.nan
        df['bb_pct'] = np.nan
    
    try:
        # ATR (Average True Range) - volatility measure
        atr = ta.volatility.AverageTrueRange(high=high_series, low=low_series, close=close_series, window=14)
        df['atr'] = atr.average_true_range()
        df['atr_pct'] = (df['atr'] / close_series) * 100  # ATR as % of price
    except Exception as e:
        print(f"[WARN] ATR failed: {e}")
        df['atr'] = np.nan
        df['atr_pct'] = np.nan
    
    try:
        # Keltner Channels
        keltner = ta.volatility.KeltnerChannel(high=high_series, low=low_series, close=close_series, window=20)
        df['keltner_upper'] = keltner.keltner_channel_hband()
        df['keltner_mid'] = keltner.keltner_channel_mband()
        df['keltner_lower'] = keltner.keltner_channel_lband()
    except Exception as e:
        print(f"[WARN] Keltner Channels failed: {e}")
        df['keltner_upper'] = np.nan
        df['keltner_mid'] = np.nan
        df['keltner_lower'] = np.nan
    
    # ============================================================
    # VOLUME INDICATORS
    # ============================================================
    
    try:
        # OBV (On-Balance Volume)
        obv = ta.volume.OnBalanceVolumeIndicator(close=close_series, volume=volume_series)
        df['obv'] = obv.on_balance_volume()
    except Exception as e:
        print(f"[WARN] OBV failed: {e}")
        df['obv'] = np.nan
    
    try:
        # CMF (Chaikin Money Flow)
        cmf = ta.volume.ChaikinMoneyFlowIndicator(
            high=high_series, 
            low=low_series, 
            close=close_series, 
            volume=volume_series, 
            window=20
        )
        df['cmf'] = cmf.chaikin_money_flow()
    except Exception as e:
        print(f"[WARN] CMF failed: {e}")
        df['cmf'] = np.nan
    
    try:
        # Volume SMA (for volume ratio)
        df['volume_sma_20'] = volume_series.rolling(window=20).mean()
        df['volume_ratio'] = volume_series / df['volume_sma_20']
    except Exception as e:
        print(f"[WARN] Volume ratio failed: {e}")
        df['volume_sma_20'] = np.nan
        df['volume_ratio'] = np.nan
    
    # ============================================================
    # DERIVED FEATURES
    # ============================================================
    
    try:
        # Price position relative to Bollinger Bands
        df['price_above_bb_upper'] = (close_series > df['bb_upper']).astype(int)
        df['price_below_bb_lower'] = (close_series < df['bb_lower']).astype(int)
    except Exception as e:
        print(f"[WARN] BB derived features failed: {e}")
        df['price_above_bb_upper'] = 0
        df['price_below_bb_lower'] = 0
    
    try:
        # EMA crossovers
        df['ema14_above_ema50'] = (df['ema_14'] > df['ema_50']).astype(int)
    except Exception as e:
        print(f"[WARN] EMA crossover failed: {e}")
        df['ema14_above_ema50'] = 0
    
    try:
        # ADX trend strength classification
        df['adx_strong_trend'] = (df['adx'] > 25).astype(int)  # ADX > 25 = strong trend
        df['adx_weak_trend'] = (df['adx'] < 20).astype(int)    # ADX < 20 = weak/ranging
    except Exception as e:
        print(f"[WARN] ADX classification failed: {e}")
        df['adx_strong_trend'] = 0
        df['adx_weak_trend'] = 0
    
    try:
        # RSI overbought/oversold
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
    except Exception as e:
        print(f"[WARN] RSI classification failed: {e}")
        df['rsi_overbought'] = 0
        df['rsi_oversold'] = 0
    
    try:
        # Stochastic signals
        df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)
        df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)
    except Exception as e:
        print(f"[WARN] Stochastic classification failed: {e}")
        df['stoch_oversold'] = 0
        df['stoch_overbought'] = 0
    
    # Clean infinities and NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Return without dropna() to preserve all rows (handle NaN in model training)
    return df


def add_indicators_minimal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Legacy minimal version (for backward compatibility).
    Use add_indicators() for full feature set.
    """
    df = df.copy()
    df = _fix_ohlcv(df)
    
    close_series = _ensure_series(df['Close'], index=df.index).astype(float)
    
    try:
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
    except Exception as e:
        print(f"[ERROR] Minimal indicators failed: {e}")
    
    return df


if __name__ == "__main__":
    # Test the enhanced indicators
    from data_loader import fetch_historical_data
    
    print("\n" + "="*60)
    print("TESTING ENHANCED INDICATORS")
    print("="*60)
    
    df = fetch_historical_data("NVDA", period="5d", interval="15m")
    
    if df is not None and not df.empty:
        print(f"\nOriginal columns: {df.columns.tolist()}")
        print(f"Shape: {df.shape}")
        print(f"Data types: {df.dtypes}")
        
        df_enhanced = add_indicators(df)
        
        print(f"\nEnhanced columns: {df_enhanced.columns.tolist()}")
        print(f"Shape: {df_enhanced.shape}")
        print(f"\nSample data (last 5 rows):")
        
        # Show key indicators
        key_cols = ['Close', 'rsi', 'macd_hist', 'adx', 'bb_pct', 'atr_pct', 'volume_ratio']
        available_cols = [c for c in key_cols if c in df_enhanced.columns]
        print(df_enhanced[available_cols].tail())
        
        print(f"\n✅ Added {len(df_enhanced.columns) - len(df.columns)} new indicator features")
    else:
        print("❌ Failed to fetch test data")
