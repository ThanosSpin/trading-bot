# retrain_model.py
import os
import pickle
import glob
from datetime import datetime

from data_loader import fetch_historical_data
from updated_model_xgb import train_model
from config import SYMBOL, MODEL_DIR

MAX_BACKUPS = 6  # keep only last 6 backups
LOOKBACK_YEARS = 2
INTERVAL = "1d"


# ---------------------------------------------------------
# Save model + rolling backups
# ---------------------------------------------------------
def save_model_with_backup(model, symbol):
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Active model path
    model_path = os.path.join(MODEL_DIR, f"model_{symbol}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Active model saved: {model_path}")

    # Monthly backup
    month_tag = datetime.now().strftime("%Y-%m")
    backup_path = os.path.join(MODEL_DIR, f"model_{symbol}_{month_tag}.pkl")

    if not os.path.exists(backup_path):
        with open(backup_path, "wb") as f:
            pickle.dump(model, f)
        print(f"📦 Monthly backup created: {backup_path}")
    else:
        print(f"ℹ️ Monthly backup for {symbol} in {month_tag} already exists — skipping.")

    # Cleanup old backups
    backups = sorted(glob.glob(os.path.join(MODEL_DIR, f"model_{symbol}_*.pkl")))
    if len(backups) > MAX_BACKUPS:
        for old in backups[:-MAX_BACKUPS]:
            os.remove(old)
            print(f"🗑️ Removed old backup for {symbol}: {old}")


# ---------------------------------------------------------
# Main multi-symbol retraining loop
# ---------------------------------------------------------
def main():

    symbols = SYMBOL if isinstance(SYMBOL, list) else [SYMBOL]

    print("\n🔄 Starting model retraining...")
    print(f"📌 Symbols: {symbols}\n")

    for sym in symbols:
        print(f"\n==============================")
        print(f"🔄 Retraining model for {sym}")
        print(f"==============================")

        df = fetch_historical_data(symbol=sym, years=LOOKBACK_YEARS, interval=INTERVAL)

        if df is None or df.empty:
            print(f"[ERROR] No data available for {sym}. Skipping.")
            continue

        try:
            model = train_model(df, symbol=sym)
            save_model_with_backup(model, symbol=sym)
        except Exception as e:
            print(f"[ERROR] Failed to train model for {sym}: {e}")

    print("\n🎉 All retraining tasks complete.\n")


if __name__ == "__main__":
    main()