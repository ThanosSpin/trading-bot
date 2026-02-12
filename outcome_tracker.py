#!/usr/bin/env python
"""
Outcome Tracker
Retroactively fills in actual outcomes for past predictions to enable
model performance evaluation and calibration analysis.

Usage:
    python outcome_tracker.py                    # Update all symbols
    python outcome_tracker.py --symbols NVDA AAPL  # Specific symbols
    python outcome_tracker.py --hours 24         # Only update last 24h
    
Schedule: Run daily or before diagnostics
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import argparse

# Import your existing modules
try:
    from data_loader import fetch_historical_data
    from config import SYMBOLS, LOGS_DIR
except ImportError:
    print("[WARN] Could not import from data_loader or config")
    print("      Using fallback configuration")
    SYMBOLS = ['NVDA', 'AAPL', 'ABBV', 'PLTR', 'SPY']
    LOGS_DIR = 'logs'

# Ensure logs directory exists
os.makedirs(LOGS_DIR, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_price_at_time(symbol: str, target_time: pd.Timestamp, tolerance_minutes: int = 5) -> float:
    """
    Get the actual price for a symbol at a specific time.
    
    Args:
        symbol: Stock symbol
        target_time: Target datetime (timezone-aware)
        tolerance_minutes: Look within +/- this many minutes
    
    Returns:
        Price at that time, or None if not found
    """
    try:
        # Make sure target_time is timezone-aware
        if target_time.tzinfo is None:
            target_time = target_time.tz_localize('UTC')
        
        # Calculate date range to fetch
        start_date = target_time - pd.Timedelta(days=1)
        end_date = target_time + pd.Timedelta(days=1)
        
        # Fetch intraday data
        df = fetch_historical_data(
            symbol,
            period="5d",  # Get 5 days to ensure coverage
            interval="1m"  # Use 1-minute for precision
        )
        
        if df is None or len(df) == 0:
            return None
        
        # Ensure index is timezone-aware
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize('UTC')
        
        # Find closest timestamp within tolerance
        time_diffs = (df.index - target_time).total_seconds().abs()
        min_diff_idx = time_diffs.idxmin()
        min_diff_seconds = time_diffs.loc[min_diff_idx]
        
        # Check if within tolerance
        if min_diff_seconds <= tolerance_minutes * 60:
            price = float(df.loc[min_diff_idx, 'Close'])
            return price
        else:
            # print(f"[WARN] {symbol}: No price within {tolerance_minutes}min of {target_time}")
            return None
            
    except Exception as e:
        print(f"[ERROR] get_price_at_time({symbol}, {target_time}): {e}")
        return None


def calculate_return(start_price: float, end_price: float) -> float:
    """Calculate percentage return"""
    if start_price is None or end_price is None or start_price == 0:
        return None
    return (end_price - start_price) / start_price


# ============================================================
# MAIN OUTCOME UPDATE LOGIC
# ============================================================

def update_outcomes_for_symbol(symbol: str, model_type: str, lookback_hours: int = None):
    """
    Update actual outcomes for a specific symbol and model type.
    
    Args:
        symbol: Stock symbol
        model_type: 'intraday_mom' or 'intraday_mr'
        lookback_hours: Only update predictions from last N hours (None = all)
    
    Returns:
        Number of outcomes updated
    """
    log_file = os.path.join(LOGS_DIR, f"predictions_{symbol}_{model_type}.csv")
    
    if not os.path.exists(log_file):
        print(f"[SKIP] {symbol} {model_type}: Log file not found")
        return 0
    
    try:
        # Load predictions log
        df = pd.read_csv(log_file)
        
        if len(df) == 0:
            print(f"[SKIP] {symbol} {model_type}: Empty log")
            return 0
        
        # Ensure timestamp column exists and is datetime
        if 'timestamp' not in df.columns:
            print(f"[ERROR] {symbol} {model_type}: No timestamp column")
            return 0
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        # Add actual_outcome column if it doesn't exist
        if 'actual_outcome' not in df.columns:
            df['actual_outcome'] = np.nan
        
        # Add actual_price and return columns if they don't exist
        if 'actual_price' not in df.columns:
            df['actual_price'] = np.nan
        if 'return_pct' not in df.columns:
            df['return_pct'] = np.nan
        
        # Filter to recent predictions if specified
        if lookback_hours:
            cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(hours=lookback_hours)
            df_update = df[df['timestamp'] > cutoff].copy()
        else:
            df_update = df.copy()
        
        # Find predictions with missing outcomes that are old enough to evaluate
        # (must be at least 15 minutes old)
        min_age = pd.Timestamp.now(tz='UTC') - pd.Timedelta(minutes=15)
        df_pending = df_update[
            (df_update['actual_outcome'].isna()) & 
            (df_update['timestamp'] < min_age)
        ]
        
        if len(df_pending) == 0:
            print(f"[SKIP] {symbol} {model_type}: No pending outcomes to update")
            return 0
        
        print(f"\n[UPDATE] {symbol} {model_type}: Processing {len(df_pending)} predictions...")
        
        updated_count = 0
        
        for idx, row in df_pending.iterrows():
            pred_time = row['timestamp']
            pred_price = row.get('price', None)
            
            if pred_price is None or pred_price == 0:
                continue
            
            # Calculate target time (15 minutes after prediction)
            target_time = pred_time + pd.Timedelta(minutes=15)
            
            # Get actual price at target time
            actual_price = get_price_at_time(symbol, target_time, tolerance_minutes=5)
            
            if actual_price is None:
                continue
            
            # Calculate return
            ret = calculate_return(pred_price, actual_price)
            
            if ret is None:
                continue
            
            # Determine actual outcome (1 = up, 0 = down/flat)
            actual_outcome = 1 if ret > 0 else 0
            
            # Update dataframe
            df.loc[idx, 'actual_outcome'] = actual_outcome
            df.loc[idx, 'actual_price'] = actual_price
            df.loc[idx, 'return_pct'] = ret * 100  # Store as percentage
            
            updated_count += 1
        
        if updated_count > 0:
            # Save updated log
            df.to_csv(log_file, index=False)
            print(f"[SUCCESS] {symbol} {model_type}: Updated {updated_count} outcome(s)")
        else:
            print(f"[SKIP] {symbol} {model_type}: No outcomes could be resolved")
        
        return updated_count
        
    except Exception as e:
        print(f"[ERROR] {symbol} {model_type}: {e}")
        import traceback
        traceback.print_exc()
        return 0


def update_all_outcomes(symbols: list = None, lookback_hours: int = None):
    """
    Update outcomes for all symbols and model types.
    
    Args:
        symbols: List of symbols to update (None = all from config)
        lookback_hours: Only update recent predictions (None = all)
    
    Returns:
        Total number of outcomes updated
    """
    if symbols is None:
        symbols = SYMBOLS
    
    model_types = ['intraday_mom', 'intraday_mr']
    
    print(f"\n{'='*70}")
    print(f"📊 OUTCOME TRACKER")
    print(f"{'='*70}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Models: {', '.join(model_types)}")
    if lookback_hours:
        print(f"Lookback: Last {lookback_hours} hours")
    else:
        print(f"Lookback: All time")
    print(f"{'='*70}\n")
    
    total_updated = 0
    
    for symbol in symbols:
        for model_type in model_types:
            count = update_outcomes_for_symbol(symbol, model_type, lookback_hours)
            total_updated += count
    
    print(f"\n{'='*70}")
    print(f"✅ COMPLETE: Updated {total_updated} total outcome(s)")
    print(f"{'='*70}\n")
    
    return total_updated


# ============================================================
# CLI ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Update prediction outcomes for model evaluation")
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=None,
        help='Symbols to update (default: all from config)'
    )
    parser.add_argument(
        '--hours',
        type=int,
        default=None,
        help='Only update predictions from last N hours (default: all)'
    )
    
    args = parser.parse_args()
    
    try:
        total = update_all_outcomes(symbols=args.symbols, lookback_hours=args.hours)
        
        if total > 0:
            print(f"✅ Successfully updated {total} outcome(s)")
            print(f"\n💡 Next steps:")
            print(f"   1. Run diagnostics: python diagnose_intraday_models.py")
            print(f"   2. Check calibration plots in: diagnostics/")
            sys.exit(0)
        else:
            print(f"ℹ️ No outcomes were updated")
            print(f"\n💡 Possible reasons:")
            print(f"   - All predictions already have outcomes")
            print(f"   - Predictions are too recent (<15 min old)")
            print(f"   - Historical price data unavailable")
            sys.exit(0)
            
    except Exception as e:
        print(f"\n❌ Outcome tracker failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
