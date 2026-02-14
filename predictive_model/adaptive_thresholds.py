# adaptive_thresholds.py
"""
Adaptive regime thresholds that adjust to each symbol's volatility characteristics.
Instead of fixed thresholds (0.003), calculates symbol-specific values based on recent behavior.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
from config.config import INTRADAY_REGIME_OVERRIDES
from predictive_model.data_loader import fetch_historical_data

# Cache thresholds to avoid repeated API calls
_THRESHOLD_CACHE = {}
_CACHE_EXPIRY_HOURS = 24  # Recalculate daily


def get_adaptive_regime_thresholds(
    sym: str,
    lookback_days: int = 30,
    percentile: float = 0.70,
    interval: str = "15m",
    force_refresh: bool = False
) -> Dict[str, float]:
    """
    Calculate symbol-specific regime thresholds based on recent history.

    Args:
        sym: Stock symbol (e.g., "NVDA")
        lookback_days: Days of history to analyze
        percentile: Threshold percentile (0.70 = 70th percentile of movements)
        interval: Data interval ("15m" for intraday)
        force_refresh: Force recalculation even if cached

    Returns:
        {
            'mom_trig': float,  # Momentum threshold for this symbol
            'vol_trig': float,  # Volatility threshold for this symbol
            'calculated_at': str,
            'sample_size': int
        }

    Example:
        >>> thresholds = get_adaptive_regime_thresholds("NVDA")
        >>> print(thresholds)
        {'mom_trig': 0.0045, 'vol_trig': 0.0038, 'calculated_at': '2026-02-03T11:46:00', 'sample_size': 2340}
    """

    
    cache_key = f"{sym}_{lookback_days}_{percentile}_{interval}"

    # ✅ NEW: Check for config override BEFORE doing anything else
    try:
        from config.config import INTRADAY_REGIME_OVERRIDES
        ovr = INTRADAY_REGIME_OVERRIDES.get(sym, {})
        
        if ovr.get("disable_adaptive", False):
            print(f"[ADAPTIVE] {sym}: Adaptive disabled via config override")
            result = {
                'mom_trig': float(ovr.get("mom_trig", 0.0050)),
                'vol_trig': float(ovr.get("vol_trig", 0.0035)),
                'calculated_at': datetime.now().isoformat(),
                'sample_size': 0,
                'percentile': percentile,
                'lookback_days': lookback_days,
                'source': 'config_override'
            }
            print(f"[ADAPTIVE] {sym} using config: mom={result['mom_trig']:.4f} vol={result['vol_trig']:.4f}")
            return result
    except ImportError:
        pass  # Config doesn't have INTRADAY_REGIME_OVERRIDES yet
    
    # Check cache
    if not force_refresh and cache_key in _THRESHOLD_CACHE:
        cached = _THRESHOLD_CACHE[cache_key]
        age = datetime.now() - datetime.fromisoformat(cached['calculated_at'])
        if age < timedelta(hours=_CACHE_EXPIRY_HOURS):
            print(f"[CACHE] Using cached thresholds for {sym} (age: {age.seconds//3600}h)")
            return cached

    print(f"[ADAPTIVE] Calculating regime thresholds for {sym}...")

    try:
        # Fetch historical data
        df = fetch_historical_data(
            sym, 
            period=f"{lookback_days}d", 
            interval=interval
        )

        if df is None or df.empty:
            print(f"[ERROR] No data for {sym}, using default thresholds")
            return _get_default_thresholds(sym)

        if len(df) < 100:
            print(f"[WARN] Insufficient data for {sym} ({len(df)} bars), using defaults")
            return _get_default_thresholds(sym)

        # Calculate momentum (1-hour = 4 bars for 15min interval)
        bars_per_hour = 4 if interval == "15m" else 1
        df['mom_1h'] = df['Close'].pct_change(bars_per_hour)

        # Calculate realized volatility (rolling 12-bar = 3 hours)
        rets = df['Close'].pct_change()
        df['vol_12'] = rets.rolling(12).std()

        # Remove NaN values
        df_clean = df[['mom_1h', 'vol_12']].dropna()

        if len(df_clean) < 50:
            print(f"[WARN] Insufficient clean data for {sym}, using defaults")
            return _get_default_thresholds(sym)

        # Calculate thresholds at specified percentile
        mom_threshold = float(df_clean['mom_1h'].abs().quantile(percentile))
        vol_threshold = float(df_clean['vol_12'].quantile(percentile))

        # Apply reasonable bounds (prevent extreme values)
        mom_threshold = max(0.0010, min(mom_threshold, 0.0200))  # 0.1% to 2%
        vol_threshold = max(0.0005, min(vol_threshold, 0.0150))  # 0.05% to 1.5%

        result = {
            'mom_trig': round(mom_threshold, 5),
            'vol_trig': round(vol_threshold, 5),
            'calculated_at': datetime.now().isoformat(),
            'sample_size': len(df_clean),
            'percentile': percentile,
            'lookback_days': lookback_days
        }

        # Cache result
        _THRESHOLD_CACHE[cache_key] = result

        print(f"✅ {sym} adaptive thresholds: mom={result['mom_trig']:.4f} vol={result['vol_trig']:.4f} (n={result['sample_size']})")

        return result

    except Exception as e:
        print(f"[ERROR] Failed to calculate adaptive thresholds for {sym}: {e}")
        return _get_default_thresholds(sym)


def _get_default_thresholds(sym: str) -> Dict[str, float]:
    """Fallback to sensible defaults based on symbol type."""
    # Default thresholds based on typical volatility
    defaults = {
        'SPY': {'mom_trig': 0.0015, 'vol_trig': 0.0015},  # Low vol index
        'QQQ': {'mom_trig': 0.0020, 'vol_trig': 0.0020},  # Medium vol tech
        'NVDA': {'mom_trig': 0.0035, 'vol_trig': 0.0032}, # High vol stock
        'AAPL': {'mom_trig': 0.0030, 'vol_trig': 0.0030}, # Medium vol stock
        'ABBV': {'mom_trig': 0.0025, 'vol_trig': 0.0025}, # Medium-low vol
    }

    if sym in defaults:
        result = defaults[sym]
    else:
        # Generic default for unknown symbols
        result = {'mom_trig': 0.0030, 'vol_trig': 0.0030}

    result.update({
        'calculated_at': datetime.now().isoformat(),
        'sample_size': 0,
        'source': 'default'
    })

    print(f"[DEFAULT] Using default thresholds for {sym}: {result['mom_trig']:.4f}/{result['vol_trig']:.4f}")
    return result


def get_all_symbols_thresholds(symbols: list, **kwargs) -> Dict[str, Dict[str, float]]:
    """
    Calculate adaptive thresholds for multiple symbols at once.

    Args:
        symbols: List of symbols ["NVDA", "AAPL", ...]
        **kwargs: Passed to get_adaptive_regime_thresholds()

    Returns:
        {'NVDA': {'mom_trig': 0.0045, ...}, 'AAPL': {...}, ...}
    """
    results = {}
    for sym in symbols:
        results[sym] = get_adaptive_regime_thresholds(sym, **kwargs)
    return results


def save_thresholds_to_config(symbols: list, output_path: str = "config_adaptive.py"):
    """
    Calculate thresholds and save to a Python config file.

    Usage:
        save_thresholds_to_config(['NVDA', 'AAPL', 'ABBV'])
        # Creates config_adaptive.py with INTRADAY_REGIME_OVERRIDES
    """
    thresholds = get_all_symbols_thresholds(symbols)

    config_content = f"""# config_adaptive.py
# Auto-generated adaptive regime thresholds
# Generated: {datetime.now().isoformat()}

INTRADAY_REGIME_OVERRIDES = {{
"""

    for sym, thresh in thresholds.items():
        config_content += f"    '{sym}': {{'mom_trig': {thresh['mom_trig']:.5f}, 'vol_trig': {thresh['vol_trig']:.5f}}},  # n={thresh.get('sample_size', 0)}\n"

    config_content += """}
"""

    with open(output_path, 'w') as f:
        f.write(config_content)

    print(f"\n✅ Saved adaptive thresholds to {output_path}")
    print(f"   Add to config.py: from config_adaptive import INTRADAY_REGIME_OVERRIDES")


if __name__ == "__main__":
    # Test/demo
    symbols = ['NVDA', 'AAPL', 'ABBV', 'SPY', 'QQQ']

    print("="*60)
    print("ADAPTIVE REGIME THRESHOLDS - DEMO")
    print("="*60)

    for sym in symbols:
        thresh = get_adaptive_regime_thresholds(sym, lookback_days=30, percentile=0.70)
        print(f"\n{sym}:")
        print(f"  Momentum threshold: {thresh['mom_trig']:.4f} ({thresh['mom_trig']*100:.2f}%)")
        print(f"  Volatility threshold: {thresh['vol_trig']:.5f} ({thresh['vol_trig']*100:.3f}%)")
        print(f"  Sample size: {thresh.get('sample_size', 0)} bars")

    print("\n" + "="*60)
    print("SAVE TO CONFIG")
    print("="*60)
    save_thresholds_to_config(symbols)
