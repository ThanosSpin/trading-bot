# retrain_intraday_dual.py
import os
from datetime import datetime
import joblib

from data_loader import fetch_historical_data
from features import _clean_columns
from model_xgb import train_model, MODEL_DIR
from config import TRAIN_SYMBOLS

INTRADAY_LOOKBACK_DAYS = 60
INTRADAY_INTERVAL = "15m"

def _save_artifact(artifact, symbol: str, mode: str):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"{symbol}_{mode}_xgb.pkl")
    joblib.dump(artifact, path)
    print(f"✅ Saved {symbol} {mode}: {path}")

def main():
    symbols = TRAIN_SYMBOLS if isinstance(TRAIN_SYMBOLS, list) else [TRAIN_SYMBOLS]
    print("\n🔄 Weekend intraday dual-model training")
    print("📌 Symbols:", symbols)
    print("🕒", datetime.now().isoformat(), "\n")

    for sym in symbols:
        df = fetch_historical_data(
            sym,
            period=f"{INTRADAY_LOOKBACK_DAYS}d",
            interval=INTRADAY_INTERVAL,
        )
        if df is None or df.empty:
            print(f"[ERROR] No intraday data for {sym}, skipping.")
            continue

        df = _clean_columns(df)

        # Train mean-reversion intraday model
        try:
            art_mr = train_model(df, symbol=sym, mode="intraday_mr")
            _save_artifact(art_mr, sym, "intraday_mr")
        except Exception as e:
            print(f"[ERROR] {sym} intraday_mr training failed: {e}")

        # Train momentum intraday model
        try:
            art_mom = train_model(df, symbol=sym, mode="intraday_mom")
            _save_artifact(art_mom, sym, "intraday_mom")
        except Exception as e:
            print(f"[ERROR] {sym} intraday_mom training failed: {e}")

    print("\n🎉 Weekend intraday training complete.\n")

if __name__ == "__main__":
    main()