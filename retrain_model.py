# retrain_model.py
import os
import glob
import shutil
from datetime import datetime

import joblib

from data_loader import fetch_historical_data
from model_xgb import train_model, MODEL_DIR
from features import _clean_columns
from config import SYMBOL

# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------
MAX_BACKUPS = 6  # keep only last 6 backups
LOOKBACK_YEARS = 2
DAILY_INTERVAL = "1d"

INTRADAY_LOOKBACK_DAYS = 60
INTRADAY_INTERVAL = "15m"


# ---------------------------------------------------------
# Save model artifact + rolling backups
# ---------------------------------------------------------
def save_model_with_backup(artifact, symbol, mode: str = "daily"):
    """
    Save active model artifact and maintain monthly rolling backups.
    Artifact should be the dict returned by train_model().
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Active model path
    active_path = os.path.join(MODEL_DIR, f"{symbol}_{mode}_xgb.pkl")
    joblib.dump(artifact, active_path)
    print(f"✅ Active {mode} model saved: {active_path}")

    # Monthly backup
    month_tag = datetime.now().strftime("%Y-%m")
    backup_path = os.path.join(MODEL_DIR, f"{symbol}_{mode}_xgb_{month_tag}.pkl")

    if not os.path.exists(backup_path):
        shutil.copy2(active_path, backup_path)
        print(f"📦 Monthly backup created: {backup_path}")
    else:
        print(f"ℹ️ Monthly backup for {symbol} ({mode}) in {month_tag} already exists — skipping.")

    # Cleanup old backups (per symbol+mode)
    pattern = os.path.join(MODEL_DIR, f"{symbol}_{mode}_xgb_*.pkl")
    backups = sorted(glob.glob(pattern))

    if len(backups) > MAX_BACKUPS:
        to_delete = backups[:-MAX_BACKUPS]
        for old in to_delete:
            try:
                os.remove(old)
                print(f"🗑️ Removed old backup for {symbol} ({mode}): {old}")
            except Exception as e:
                print(f"[WARN] Could not remove backup {old}: {e}")


# ---------------------------------------------------------
# Main multi-symbol retraining loop
# ---------------------------------------------------------
def main():
    symbols = SYMBOL if isinstance(SYMBOL, list) else [SYMBOL]
    print("\n🔄 Starting model retraining...")
    print(f"📌 Symbols: {symbols}\n")

    for sym in symbols:
        # =====================================================
        # DAILY MODEL
        # =====================================================
        print(f"\n==============================")
        print(f"🔄 Retraining DAILY model for {sym}")
        print(f"==============================")

        df_daily = fetch_historical_data(sym, years=LOOKBACK_YEARS, interval=DAILY_INTERVAL)
        if df_daily is None or df_daily.empty:
            print(f"[ERROR] No daily data for {sym}. Skipping daily retraining.")
        else:
            try:
                df_daily = _clean_columns(df_daily)
                artifact_daily = train_model(df_daily, symbol=sym, mode="daily")
                save_model_with_backup(artifact_daily, symbol=sym, mode="daily")
            except Exception as e:
                print(f"[ERROR] Failed to train daily model for {sym}: {e}")

        # =====================================================
        # INTRADAY MODEL
        # =====================================================
        print(f"\n==============================")
        print(f"🔄 Retraining INTRADAY model for {sym}")
        print(f"==============================")

        df_intraday = fetch_historical_data(
            sym,
            period=f"{INTRADAY_LOOKBACK_DAYS}d",
            interval=INTRADAY_INTERVAL,
        )
        if df_intraday is None or df_intraday.empty:
            print(f"[ERROR] No intraday data for {sym}. Skipping intraday retraining.")
        else:
            try:
                df_intraday = _clean_columns(df_intraday)
                artifact_intra = train_model(df_intraday, symbol=sym, mode="intraday")
                save_model_with_backup(artifact_intra, symbol=sym, mode="intraday")
            except Exception as e:
                print(f"[ERROR] Failed to train intraday model for {sym}: {e}")

    print("\n🎉 All retraining tasks complete.\n")


if __name__ == "__main__":
    main()