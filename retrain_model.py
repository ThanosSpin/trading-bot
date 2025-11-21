# retrain_model.py
import os
import pickle
import glob
from datetime import datetime

from data_loader import fetch_historical_data
from model_xgb import train_model
from features import _clean_columns
from config import SYMBOL, MODEL_DIR

# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------
MAX_BACKUPS = 6
LOOKBACK_YEARS = 2
DAILY_INTERVAL = "1d"

INTRADAY_LOOKBACK_DAYS = 60
INTRADAY_INTERVAL = "15m"


# ---------------------------------------------------------
# Save model + rolling backups
# ---------------------------------------------------------
def save_model_with_backup(model, symbol, mode="daily"):
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, f"model_{symbol}_{mode}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Active {mode} model saved: {model_path}")

    month_tag = datetime.now().strftime("%Y-%m")
    backup_path = os.path.join(MODEL_DIR, f"model_{symbol}_{mode}_{month_tag}.pkl")

    if not os.path.exists(backup_path):
        with open(backup_path, "wb") as f:
            pickle.dump(model, f)
        print(f"📦 Monthly backup created: {backup_path}")

    backups = sorted(glob.glob(os.path.join(MODEL_DIR, f"model_{symbol}_{mode}_*.pkl")))
    if len(backups) > MAX_BACKUPS:
        for old in backups[:-MAX_BACKUPS]:
            os.remove(old)
            print(f"🗑️ Removed old backup {old}")


# ---------------------------------------------------------
# Main multi-symbol retraining loop
# ---------------------------------------------------------
def main():
    symbols = SYMBOL if isinstance(SYMBOL, list) else [SYMBOL]
    print("\n🔄 Starting model retraining...")
    print(f"📌 Symbols: {symbols}\n")

    for sym in symbols:

        print(f"\n==============================")
        print(f"🔄 Retraining DAILY model for {sym}")
        print(f"==============================")

        df_daily = fetch_historical_data(sym, years=LOOKBACK_YEARS, interval=DAILY_INTERVAL)
        if df_daily is None or df_daily.empty:
            print(f"[ERROR] No daily data for {sym}. Skipping daily model.")
        else:
            try:
                df_daily = _clean_columns(df_daily)
                model_daily = train_model(df_daily, symbol=sym, mode="daily")
                save_model_with_backup(model_daily, symbol=sym, mode="daily")
            except Exception as e:
                print(f"[ERROR] Failed to train daily model for {sym}: {e}")

        print(f"\n==============================")
        print(f"🔄 Retraining INTRADAY model for {sym}")
        print(f"==============================")

        df_intraday = fetch_historical_data(sym, period=f"{INTRADAY_LOOKBACK_DAYS}d", interval=INTRADAY_INTERVAL)
        if df_intraday is None or df_intraday.empty:
            print(f"[ERROR] No intraday data for {sym}. Skipping intraday model.")
        else:
            try:
                df_intraday = _clean_columns(df_intraday)
                model_intraday = train_model(df_intraday, symbol=sym, mode="intraday")
                save_model_with_backup(model_intraday, symbol=sym, mode="intraday")
            except Exception as e:
                print(f"[ERROR] Failed to train intraday model for {sym}: {e}")

    print("\n🎉 All retraining tasks complete.\n")


if __name__ == "__main__":
    main()