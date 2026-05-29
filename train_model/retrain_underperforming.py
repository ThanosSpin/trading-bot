#!/usr/bin/env python
"""
Retrain Underperforming Models

Automatically identifies and retrains models below an accuracy threshold.

Patched behavior:
- Uses patched train_model(...) artifact flow.
- Evaluates historical predictions with the saved artifact threshold,
  not a hardcoded 0.5 threshold.
- Saves the full artifact unchanged.
"""

import os
import sys
import argparse
from datetime import datetime

import joblib
import pandas as pd

try:
    from predictive_model.data_loader import fetch_historical_data
    from predictive_model.features import _clean_columns
    from predictive_model.model_xgb import train_model
    from config.config import SYMBOL
    try:
        from config.config import LOGS_DIR, MODEL_DIR
    except ImportError:
        LOGS_DIR = "logs"
        MODEL_DIR = "models"
        print(f"[WARN] Using default directories: LOGS_DIR={LOGS_DIR}, MODEL_DIR={MODEL_DIR}")
except ImportError as e:
    print(f"[ERROR] Missing required module: {e}")
    print("Ensure data_loader.py, features.py, model_xgb.py, and config.py exist")
    sys.exit(1)

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def _artifact_path(symbol: str, model_type: str) -> str:
    return os.path.join(MODEL_DIR, f"{symbol}_{model_type}_xgb.pkl")


def _load_artifact_threshold(symbol: str, model_type: str, default: float = 0.5) -> float:
    """
    Load the saved decision threshold from the patched model artifact.

    Priority:
    1. artifact['decision_threshold']
    2. artifact['threshold_optimization']['best_threshold']
    3. default
    """
    path = _artifact_path(symbol, model_type)
    if not os.path.exists(path):
        return default

    try:
        artifact = joblib.load(path)
        if not isinstance(artifact, dict):
            return default

        if artifact.get("decision_threshold") is not None:
            return float(artifact["decision_threshold"])

        threshold_opt = artifact.get("threshold_optimization", {}) or {}
        if threshold_opt.get("best_threshold") is not None:
            return float(threshold_opt["best_threshold"])

        return default

    except Exception as e:
        print(f"[WARN] Could not load threshold for {symbol}/{model_type}: {e}")
        return default


def analyze_model_performance(symbol: str, model_type: str) -> dict:
    """
    Analyze actual performance from prediction logs using the saved artifact threshold.
    """
    log_file = os.path.join(LOGS_DIR, f"predictions_{symbol}_{model_type}.csv")

    if not os.path.exists(log_file):
        return {
            "accuracy": None,
            "total_predictions": 0,
            "evaluated_predictions": 0,
            "has_outcomes": False,
            "decision_threshold": None,
        }

    try:
        df = pd.read_csv(log_file)

        if len(df) == 0:
            return {
                "accuracy": None,
                "total_predictions": 0,
                "evaluated_predictions": 0,
                "has_outcomes": False,
                "decision_threshold": None,
            }

        if "actual_outcome" not in df.columns:
            return {
                "accuracy": None,
                "total_predictions": len(df),
                "evaluated_predictions": 0,
                "has_outcomes": False,
                "decision_threshold": None,
            }

        df_eval = df[df["actual_outcome"].notna()].copy()

        if len(df_eval) == 0:
            return {
                "accuracy": None,
                "total_predictions": len(df),
                "evaluated_predictions": 0,
                "has_outcomes": False,
                "decision_threshold": None,
            }

        threshold = _load_artifact_threshold(symbol, model_type, default=0.5)

        df_eval["predicted_outcome"] = (df_eval["predicted_prob"] >= threshold).astype(int)
        accuracy = (df_eval["predicted_outcome"] == df_eval["actual_outcome"]).mean()

        return {
            "accuracy": float(accuracy),
            "total_predictions": int(len(df)),
            "evaluated_predictions": int(len(df_eval)),
            "has_outcomes": True,
            "decision_threshold": threshold,
        }

    except Exception as e:
        print(f"[ERROR] analyze_model_performance({symbol}, {model_type}): {e}")
        return {
            "accuracy": None,
            "total_predictions": 0,
            "evaluated_predictions": 0,
            "has_outcomes": False,
            "decision_threshold": None,
        }


def identify_underperformers(symbols: list = None, threshold: float = 0.52) -> list:
    """
    Identify models that need retraining.
    """
    if symbols is None:
        symbols = SYMBOL

    model_types = ["intraday_mom", "intraday_mr"]
    underperformers = []

    print(f"\n{'=' * 70}")
    print("🔍 ANALYZING MODEL PERFORMANCE")
    print(f"{'=' * 70}")
    print(f"Threshold: {threshold * 100:.1f}% accuracy")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"{'=' * 70}\n")

    for symbol in symbols:
        for model_type in model_types:
            perf = analyze_model_performance(symbol, model_type)

            if not perf["has_outcomes"]:
                print(f"⏭️ {symbol:6} {model_type:15} - No outcomes yet")
                continue

            acc = perf["accuracy"]
            model_threshold = perf.get("decision_threshold")

            if acc is None:
                print(f"⏭️ {symbol:6} {model_type:15} - No data")
                continue

            threshold_text = f" | model_threshold={model_threshold:.3f}" if model_threshold is not None else ""

            if acc < threshold:
                print(
                    f"⚠️ {symbol:6} {model_type:15} - {acc * 100:5.1f}% (BELOW THRESHOLD){threshold_text}"
                )
                underperformers.append((symbol, model_type, acc))
            else:
                print(
                    f"✅ {symbol:6} {model_type:15} - {acc * 100:5.1f}% (OK){threshold_text}"
                )

    print(f"\n{'=' * 70}")
    print(f"Found {len(underperformers)} model(s) needing retraining")
    print(f"{'=' * 70}\n")

    return underperformers


def retrain_model(symbol: str, model_type: str, tune: bool = False, lookback_days: int = 60) -> bool:
    """
    Retrain a specific model with fresh data using patched train_model().
    """
    print(f"\n{'=' * 70}")
    print(f"🔄 RETRAINING: {symbol} - {model_type}")
    print(f"{'=' * 70}")

    if "intraday" in model_type and lookback_days > 60:
        print("[WARN] Yahoo Finance limits intraday data to 60 days")
        print(f"[WARN] Reducing lookback from {lookback_days} to 60 days")
        lookback_days = 60

    try:
        print(f"[DATA] Fetching {lookback_days} days of intraday data...")
        df = fetch_historical_data(
            symbol,
            period=f"{lookback_days}d",
            interval="15m",
        )

        if df is None or len(df) == 0:
            print(f"[ERROR] No data available for {symbol}")
            return False

        print(f"[DATA] Loaded {len(df)} bars")

        df = _clean_columns(df)

        print(f"[TRAIN] Training {model_type} model...")
        if tune:
            print("[TRAIN] Note: any hyperparameter tuning must be implemented inside train_model().")

        artifact = train_model(df, symbol=symbol, mode=model_type)

        if artifact is None:
            print(f"[ERROR] Training failed for {symbol} {model_type}")
            return False

        model_path = _artifact_path(symbol, model_type)
        joblib.dump(artifact, model_path)
        print(f"[SUCCESS] Model saved: {model_path}")

        metrics = artifact.get("metrics", {}) or {}
        decision_threshold = artifact.get(
            "decision_threshold",
            (artifact.get("threshold_optimization", {}) or {}).get("best_threshold"),
        )

        print("\n[METRICS] New model performance:")
        for key in ["accuracy", "logloss", "rocauc", "f1", "precision", "recall", "brier_score"]:
            val = metrics.get(key)
            if val is not None:
                print(f" {key}: {val:.3f}")

        if decision_threshold is not None:
            print(f" decision_threshold: {decision_threshold:.3f}")

        return True

    except Exception as e:
        print(f"[ERROR] Retraining failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def retrain_underperformers(
    symbols: list = None,
    threshold: float = 0.52,
    tune: bool = False,
    lookback_days: int = 60,
    force_symbols: list = None,
):
    """
    Main workflow: identify and retrain underperforming models.
    """
    print(f"\n{'=' * 70}")
    print("🤖 AUTOMATED MODEL RETRAINING")
    print(f"{'=' * 70}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Threshold: {threshold * 100:.1f}%")
    print(f"Tuning: {'Enabled' if tune else 'Disabled'}")
    print(f"Lookback: {lookback_days} days")
    print(f"{'=' * 70}\n")

    if force_symbols:
        to_retrain = [(s, m, None) for s, m in force_symbols]
        print(f"[FORCED] Retraining {len(to_retrain)} model(s):")
        for s, m, _ in to_retrain:
            print(f" - {s} {m}")
    else:
        to_retrain = identify_underperformers(symbols, threshold)

    if len(to_retrain) == 0:
        print("\n✅ All models performing above threshold!")
        print("💡 Consider:")
        print(" - Lowering threshold to retrain marginal models")
        print(" - Using --force to retrain specific models")
        return

    print(f"\n{'=' * 70}")
    print(f"⚠️ About to retrain {len(to_retrain)} model(s)")
    print(f"{'=' * 70}")

    if not force_symbols:
        response = input("\nProceed? [y/N]: ")
        if response.lower() != "y":
            print("Cancelled.")
            return

    results = []

    for symbol, model_type, old_acc in to_retrain:
        success = retrain_model(symbol, model_type, tune=tune, lookback_days=lookback_days)
        results.append((symbol, model_type, old_acc, success))
        print()

    print(f"\n{'=' * 70}")
    print("📊 RETRAINING SUMMARY")
    print(f"{'=' * 70}\n")

    successful = sum(1 for _, _, _, s in results if s)
    failed = len(results) - successful

    for symbol, model_type, old_acc, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        acc_str = f"(was {old_acc * 100:.1f}%)" if old_acc is not None else ""
        print(f"{status:12} {symbol:6} {model_type:15} {acc_str}")

    print(f"\n{'=' * 70}")
    print(f"✅ Successful: {successful}/{len(results)}")
    print(f"❌ Failed: {failed}/{len(results)}")
    print(f"{'=' * 70}\n")

    if successful > 0:
        print("💡 Next steps:")
        print(" 1. Wait for new predictions to accumulate")
        print(" 2. Run outcome_tracker.py to fill in results")
        print(" 3. Run diagnose_intraday_models.py to verify improvement")
        print("\n python outcome_tracker.py")
        print(" python diagnose_intraday_models.py")


def main():
    parser = argparse.ArgumentParser(description="Automatically retrain underperforming models")

    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Symbols to analyze (default: all from config)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.52,
        help="Accuracy threshold for retraining (default: 0.52)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable hyperparameter tuning (slower but better)",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=60,
        help="Days of historical data to use (default: 60, max for intraday)",
    )
    parser.add_argument(
        "--force",
        nargs="+",
        metavar="SYMBOL:MODEL",
        help="Force retrain specific models (e.g., NVDA:intraday_mom SPY:intraday_mr)",
    )

    args = parser.parse_args()

    force_symbols = None
    if args.force:
        force_symbols = []
        for item in args.force:
            try:
                symbol, model = item.split(":")
                force_symbols.append((symbol.upper(), model))
            except ValueError:
                print(f"[ERROR] Invalid format: {item}")
                print("Use format: SYMBOL:MODEL (e.g., NVDA:intraday_mom)")
                sys.exit(1)

    try:
        retrain_underperformers(
            symbols=args.symbols,
            threshold=args.threshold,
            tune=args.tune,
            lookback_days=args.lookback,
            force_symbols=force_symbols,
        )
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
