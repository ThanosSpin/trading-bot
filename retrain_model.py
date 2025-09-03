# retrain_model.py
import os
import pickle
import glob
from datetime import datetime
from data_loader import fetch_historical_data
from updated_model_xgb import train_model
from config import SYMBOL, MODEL_DIR

MAX_BACKUPS = 6  # keep only last 6 months of backups

# Training / Data configs
LOOKBACK_YEARS = 2   # how many years of historical data to use
INTERVAL = "1d"      # interval for historical data (daily)


def save_model_with_backup(model, symbol):
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save active model (overwrite each retrain)
    model_path = os.path.join(MODEL_DIR, f"model_{symbol}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Active model saved for {symbol}: {model_path}")

    # Monthly backup (e.g., model_NVDA_2025-09.pkl)
    month_tag = datetime.now().strftime("%Y-%m")
    backup_path = os.path.join(MODEL_DIR, f"model_{symbol}_{month_tag}.pkl")

    if not os.path.exists(backup_path):  # ensure only 1 per month
        with open(backup_path, "wb") as f:
            pickle.dump(model, f)
        print(f"📦 Monthly backup created: {backup_path}")
    else:
        print(f"ℹ️ Backup for {symbol} already exists this month ({month_tag})")

    # Cleanup old backups (keep last MAX_BACKUPS)
    backups = sorted(glob.glob(os.path.join(MODEL_DIR, f"model_{symbol}_*.pkl")))
    if len(backups) > MAX_BACKUPS:
        to_delete = backups[:-MAX_BACKUPS]
        for old in to_delete:
            os.remove(old)
            print(f"🗑️ Removed old backup: {old}")


def main():
    print(f"\n🔄 Retraining model for {SYMBOL}...")

    # Fetch historical data (default LOOKBACK_YEARS, INTERVAL from config)
    df = fetch_historical_data(symbol=SYMBOL, years=LOOKBACK_YEARS, interval=INTERVAL)
    if df is None or df.empty:
        print(f"[ERROR] No data found for {SYMBOL}. Skipping.")

    try:
        model = train_model(df, symbol=SYMBOL)
        save_model_with_backup(model, symbol=SYMBOL)
    except Exception as e:
        print(f"[ERROR] Failed to retrain model for {symbol}: {e}")

if __name__ == "__main__":
    main()