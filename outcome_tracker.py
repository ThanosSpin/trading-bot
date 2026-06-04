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
    from predictive_model.data_loader import fetch_historical_data
    from config.config import SYMBOL, LOGS_DIR
except ImportError:
    print("[WARN] Could not import from data_loader or config")
    print("      Using fallback configuration")
    SYMBOL = ['NVDA', 'AAPL', 'ABBV', 'PLTR', 'SPY']
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
        


        # Fetch intraday data
        df = fetch_historical_data(
            symbol,
            period="5d",
            interval="1m"
        )
        
        if df is None or len(df) == 0:
            return None
        
        # Ensure index is timezone-aware
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize('UTC')
        
        # Create Series for time differences
        time_diffs = pd.Series(
            np.abs((df.index - target_time).total_seconds()),
            index=df.index
        )
        
        # Get the index label of minimum
        closest_time = time_diffs.idxmin()
        min_diff_seconds = time_diffs[closest_time]
        
        # Check if within tolerance
        if min_diff_seconds <= tolerance_minutes * 60:
            # ✅ Use .at[] which always returns scalar
            price = float(df.at[closest_time, 'Close'])
            return price
        else:

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

def get_next_day_close(symbol: str, pred_time: pd.Timestamp) -> float:
    """
    Get next trading day's close price relative to the prediction date (NY time).
    """
    try:
        # Convert prediction time to America/New_York to get the trading date
        pred_time_ny = pred_time.tz_convert("America/New_York")
        pred_date = pred_time_ny.date()

        # Fetch recent daily data (use a larger window, e.g. 60d)
        df = fetch_historical_data(symbol, period="60d", interval="1d")
        if df is None or len(df) == 0:
            return None

        df = df.copy()
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        df = df.sort_index()

        # Map index to dates in NY time
        idx_ny = df.index.tz_localize("UTC").tz_convert("America/New_York")
        dates = sorted(set(idx_ny.date))

        if pred_date not in dates:
            # Prediction older than our daily window or date mismatch
            return None

        i = dates.index(pred_date)
        if i + 1 >= len(dates):
            # No next trading day yet in the data
            return None

        next_date = dates[i + 1]

        # Select row(s) where NY date == next_date
        mask = (idx_ny.date == next_date)
        row_next = df.loc[mask]

        if isinstance(row_next, pd.DataFrame):
            close_price = float(row_next["Close"].iloc[-1])
        else:
            close_price = float(row_next["Close"])

        return close_price

    except Exception as e:
        print(f"[ERROR] get_next_day_close({symbol}): {e}")
        return None

def update_outcomes_for_symbol(symbol: str, lookback_hours: int = None):
    """
    Update actual outcomes for a specific symbol (all modes) using next-day close.

    Args:
        symbol: Stock symbol
        lookback_hours: Only update predictions from last N hours (None = all)

    Returns:
        Number of outcomes updated
    """
    log_file = os.path.join(LOGS_DIR, f"predictions_{symbol}.csv")
    core_cols = ["timestamp", "symbol", "mode", "predicted_prob", "price"]

    if not os.path.exists(log_file):
        print(f"[SKIP] {symbol} : Log file not found")
        return 0

    try:
        # Load predictions log (full schema, possibly 16+ columns)
        df = pd.read_csv(log_file)

        if len(df) == 0:
            print(f"[SKIP] {symbol} : Empty log")
            return 0

        # Ensure core columns exist
        for col in core_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column '{col}' in {log_file}")

        # Ensure timestamp is datetime with UTC tz
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Add outcome columns if missing
        if "actual_outcome" not in df.columns:
            df["actual_outcome"] = np.nan
        if "actual_price" not in df.columns:
            df["actual_price"] = np.nan
        if "return_pct" not in df.columns:
            df["return_pct"] = np.nan

        # Optional: restrict to recent predictions
        if lookback_hours:
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=lookback_hours)
            df_update = df[df["timestamp"] > cutoff].copy()
        else:
            df_update = df.copy()

        # Require that we are at least one day past the prediction date (approx)
        # (you can keep or simplify this; it's just to avoid trying to use future data)
        now_utc = pd.Timestamp.now(tz="UTC")

        df_pending = df_update[
            df_update["actual_outcome"].isna()
        ].copy()

        if len(df_pending) == 0:
            print(f"[SKIP] {symbol} : No pending outcomes to update")
            return 0

        print(f"\n[UPDATE] {symbol} : Processing {len(df_pending)} predictions...")

        updated_count = 0
        min_move = 0.002  # 0.2% daily min-move band

        for idx, row in df_pending.iterrows():
            pred_time = row["timestamp"]
            pred_price = row.get("price", None)

            if pred_price is None or pred_price == 0:
                print(f"[DEBUG] {symbol} row {idx}: missing pred_price, skipping")
                continue

            start_price = float(pred_price)

            # Get next day's close
            actual_price = get_next_day_close(symbol, pred_time)
            if actual_price is None:
                print(f"[DEBUG] {symbol} row {idx}: no next-day close found, skipping")
                continue

            ret = calculate_return(start_price, actual_price)
            if ret is None:
                print(f"[DEBUG] {symbol} row {idx}: ret is None, skipping")
                continue

            # Apply min-move band
            if ret >= min_move:
                actual_outcome = 1
            elif ret <= -min_move:
                actual_outcome = 0
            else:
                print(f"[DEBUG] {symbol} row {idx}: ret {ret:.4%} inside noise band, skipping")
                continue

            df.loc[idx, "actual_outcome"] = actual_outcome
            df.loc[idx, "actual_price"] = actual_price
            df.loc[idx, "return_pct"] = ret * 100.0  # percentage

            updated_count += 1

        if updated_count > 0:
            df.to_csv(log_file, index=False)
            print(f"[SUCCESS] {symbol} : Updated {updated_count} outcome(s)")
        else:
            print(f"[SKIP] {symbol} : No outcomes could be resolved")

        return updated_count

    except Exception as e:
        print(f"[ERROR] {symbol} : {e}")
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
        symbols = SYMBOL
    
    print(f"\n{'='*70}")
    print(f"📊 OUTCOME TRACKER")
    print(f"{'='*70}")
    print(f"Symbols: {', '.join(symbols)}")
    if lookback_hours:
        print(f"Lookback: Last {lookback_hours} hours")
    else:
        print(f"Lookback: All time")
    print(f"{'='*70}\n")
    
    total_updated = 0
    
    for symbol in symbols:
        count = update_outcomes_for_symbol(symbol, lookback_hours)
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
