#!/usr/bin/env python
"""
Outcome Tracker
Retroactively fills in actual outcomes for past predictions to enable
model performance evaluation and calibration analysis.

Usage:
    python outcome_tracker.py                      # Update all symbols, all time
    python outcome_tracker.py --symbols NVDA AAPL  # Specific symbols
    python outcome_tracker.py --hours 24           # Only update last 24h (optional)

Schedule: Run daily or before diagnostics
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import argparse

print("[DEBUG] Running outcome_tracker from:", __file__)

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

def calculate_return(start_price: float, end_price: float) -> float:
    """Calculate percentage return"""
    if start_price is None or end_price is None or start_price == 0:
        return None
    return (end_price - start_price) / start_price


def get_next_day_close(symbol: str, pred_time: pd.Timestamp) -> float:
    """
    Get next trading day's close price relative to the prediction date (NY time).
    Works with tz-naive pred_time by treating it as UTC.
    """
    try:
        # If pred_time is tz-naive, assume UTC; if already tz-aware, leave as is
        if pred_time.tzinfo is None or pred_time.tzinfo.utcoffset(pred_time) is None:
            pred_time = pred_time.tz_localize("UTC")

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

        # Here df.index is typically tz-naive daily dates; treat as UTC first
        if df.index.tz is None:
            idx_utc = df.index.tz_localize("UTC")
        else:
            idx_utc = df.index

        # Map index to dates in NY time
        idx_ny = idx_utc.tz_convert("America/New_York")
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

# ============================================================
# MAIN OUTCOME UPDATE LOGIC
# ============================================================

def update_outcomes_for_symbol(symbol: str, lookback_hours: int = None) -> int:
    """
    Update actual outcomes for a specific symbol (all modes) using next-day close.
    Uses the *_old.csv logs which have timestamps like '2026-06-03 11:12:48.728575+00:00'.
    """
    log_file = os.path.join(LOGS_DIR, f"predictions_{symbol}.csv")
    core_cols = ["timestamp", "symbol", "mode", "predicted_prob", "price"]

    if not os.path.exists(log_file):
        print(f"[SKIP] {symbol} : Log file not found")
        return 0

    try:
        df = pd.read_csv(log_file)
        print(f"[DEBUG] {symbol} loaded log from {log_file}, columns: {df.columns.tolist()}")

        if len(df) == 0:
            print(f"[SKIP] {symbol} : Empty log")
            return 0

        # Ensure core columns exist
        for col in core_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column '{col}' in {log_file}")

        # Parse timestamp with timezone, then drop tz => datetime64[ns]
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
        )

        # Reset outcome columns for bootstrapping on *_old files
        df["actual_outcome"] = np.nan
        df["actual_price"] = np.nan
        df["return_pct"] = np.nan

        # For *_old bootstrap, ignore lookback and process ALL rows
        df_update = df.copy()

        # Only consider rows with missing outcomes
        df_pending = df_update[df_update["actual_outcome"].isna()].copy()

        print(
            f"[DEBUG] {symbol} total rows: {len(df)}, rows after lookback: {len(df_update)}, "
            f"pending (NaN outcomes): {len(df_pending)}"
        )

        if len(df_pending) == 0:
            df.to_csv(log_file, index=False)
            print(f"[SKIP] {symbol} : No pending outcomes to update (columns ensured).")
            return 0

        print(f"\n[UPDATE] {symbol} : Processing {len(df_pending)} predictions...")

        updated_count = 0
        min_move = 0.0  # label any non-zero move for now

        for idx, row in df_pending.iterrows():
            pred_time = row["timestamp"]
            pred_price = row.get("price", None)

            if pd.isna(pred_time):
                print(f"[DEBUG] {symbol} row {idx}: timestamp NaT, skipping")
                continue

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
            df.loc[idx, "return_pct"] = ret * 100.0

            updated_count += 1

        df.to_csv(log_file, index=False)
        if updated_count > 0:
            print(f"[SUCCESS] {symbol} : Updated {updated_count} outcome(s)")
        else:
            print(f"[SKIP] {symbol} : No outcomes could be resolved (but columns ensured).")

        return updated_count

    except Exception as e:
        print(f"[ERROR] {symbol} : {e}")
        import traceback
        traceback.print_exc()
        return 0


def update_all_outcomes(symbols: list = None, lookback_hours: int = None) -> int:
    if symbols is None:
        symbols = SYMBOL

    print(f"\n{'='*70}")
    print("📊 OUTCOME TRACKER")
    print(f"{'='*70}")
    print(f"Symbols: {', '.join(symbols)}")
    if lookback_hours:
        print(f"Lookback: Last {lookback_hours} hours")
    else:
        print("Lookback: All time")
    print(f"{'='*70}\n")

    total_updated = 0
    for symbol in symbols:
        count = update_outcomes_for_symbol(symbol, lookback_hours)
        total_updated += count

    print(f"\n{'='*70}")
    print(f"✅ COMPLETE: Updated {total_updated} total outcome(s)")
    print(f"{'='*70}\n")

    return total_updated


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
            print("\n💡 Next steps:")
            print("   1. Run diagnostics: python diagnose_intraday_models.py")
            print("   2. Check calibration plots in: diagnostics/")
            sys.exit(0)
        else:
            print("ℹ️ No outcomes were updated")
            print("\n💡 Possible reasons:")
            print("   - Predictions are too recent (no next-day close yet)")
            print("   - Historical price data unavailable")
            sys.exit(0)

    except Exception as e:
        print(f"\n❌ Outcome tracker failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
