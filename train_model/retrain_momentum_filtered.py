#!/usr/bin/env python
"""
Retrain intraday_mom models using the patched training pipeline.

Important:
- Do NOT pre-filter momentum regime bars in this script.
- patched train_model(..., mode="intraday_mom") now performs the correct
  regime filtering internally using the accepted percentile logic.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predictive_model.data_loader import fetch_historical_data
from predictive_model.features_dev import _clean_columns
from predictive_model.model_xgb_dev import train_model


def retrain_momentum_model(symbol, period="60d"):
    """Retrain intraday momentum model using patched train_model()."""
    print(f"\n{'=' * 60}")
    print(f"Retraining {symbol} intraday_mom with patched regime handling")
    print(f"{'=' * 60}")

    df = fetch_historical_data(symbol, period=period, interval="15m")

    if df is None or len(df) < 500:
        print(f"❌ Insufficient data for {symbol} ({len(df) if df is not None else 0} bars)")
        print(" Need at least 500 bars for training")
        return False

    print(f"📊 Loaded {len(df)} bars")

    df = _clean_columns(df)

    print("\n🔄 Training momentum model...")
    try:
        artifact = train_model(
            df,
            symbol=symbol,
            mode="intraday_mom",
            use_multiclass=False,
        )

        metrics = artifact.get("metrics", {}) or {}
        threshold = artifact.get(
            "decision_threshold",
            (artifact.get("threshold_optimization", {}) or {}).get("best_threshold"),
        )

        print(f"✅ {symbol} intraday_mom retrained")
        if threshold is not None:
            print(f" Decision threshold: {threshold:.3f}")
        if metrics.get("accuracy") is not None:
            print(f" Accuracy: {metrics['accuracy']:.3f}")
        if metrics.get("logloss") is not None:
            print(f" LogLoss: {metrics['logloss']:.3f}")

        print(f" Model saved to: models/{symbol}_intraday_mom_xgb.pkl\n")
        return True

    except Exception as e:
        print(f"❌ Training failed for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    symbols = ["NVDA", "AAPL", "ABBV", "PLTR", "SPY"]

    print(f"\n{'=' * 60}")
    print("🔄 MOMENTUM MODEL RETRAINING")
    print(f"{'=' * 60}")
    print(f"Symbols: {', '.join(symbols)}")
    print("Period: 60 days (Yahoo Finance limit for 15min data)")
    print("Strategy: use patched train_model() internal momentum regime filtering")
    print(f"{'=' * 60}\n")

    success_count = 0
    failed_symbols = []

    for sym in symbols:
        ok = retrain_momentum_model(sym, period="60d")
        if ok:
            success_count += 1
        else:
            failed_symbols.append(sym)

    print(f"\n{'=' * 60}")
    print("📊 RETRAINING SUMMARY")
    print(f"{'=' * 60}")
    print(f"✅ Successfully retrained: {success_count}/{len(symbols)} symbols")

    if failed_symbols:
        print(f"❌ Failed: {', '.join(failed_symbols)}")
    else:
        print("🎉 All models retrained successfully!")

    print("\n⚠️ NEXT STEPS:")
    print("1. Test models with: python main.py")
    print("2. Validate with: python diagnose_intraday_models.py")
    print("3. Confirm schema validation and artifact thresholds are working before production")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
