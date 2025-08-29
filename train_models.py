# train_models.py

import os
import pickle
from datetime import datetime
import glob
from data_loader_multi import fetch_historical_data
from updated_model_xgb import train_model
from config import SYMBOL

MODEL_DIR = "models"
MAX_BACKUPS = 6  # keep only last 6 months


def save_model_with_backup(model, symbol):
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save latest active model
    model_path = os.path.join(MODEL_DIR, f"model_{symbol}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Active model saved for {symbol}: {model_path}")

    # Save monthly backup
    month_tag = datetime.now().strftime("%Y-%m")
    backup_path = os.path.join(MODEL_DIR, f"model_{symbol}_{month_tag}.pkl")

    if not os.path.exists(backup_path):  # only one backup per month
        with open(backup_path, "wb") as f:
            pickle.dump(model, f)
        print(f"📦 Monthly backup created: {backup_path}")
    else:
        print(f"ℹ️ Monthly backup for {symbol} already exists ({month_tag})")

    # Cleanup old backups (keep only last MAX_BACKUPS months)
    backups = sorted(glob.glob(os.path.join(MODEL_DIR, f"model_{symbol}_*.pkl")))
    if len(backups) > MAX_BACKUPS:
        to_delete = backups[:-MAX_BACKUPS]
        for old in to_delete:
            os.remove(old)
            print(f"🗑️ Removed old backup: {old}")


def main():
    for symbol in SYMBOL:
        print(f"\n📈 Training model for {symbol}...")

        df = fetch_historical_data(symbol)
        if df is None or df.empty:
            print(f"[ERROR] No data found for {symbol}. Skipping.")
            continue

        try:
            model = train_model(df, symbol)  # train returns model
            save_model_with_backup(model, symbol)
        except Exception as e:
            print(f"[ERROR] Failed to train model for {symbol}: {e}")


if __name__ == "__main__":
    main()