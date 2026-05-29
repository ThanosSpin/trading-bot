# retrain_intraday_dual.py
"""
Weekend intraday dual-model retraining script.

Trains both regime models (mean-reversion + momentum) for all symbols
using the patched training pipeline in predictive_model.model_xgb.

Important:
- Do NOT pre-filter regimes in this script.
- train_model(...) now owns feature generation, regime filtering,
  calibration, threshold optimization, and final artifact creation.
"""

import os
import sys
from datetime import datetime
import traceback
import joblib

from predictive_model.data_loader import fetch_historical_data
from predictive_model.features import _clean_columns
from predictive_model.model_xgb import train_model, MODEL_DIR
from config.config import TRAIN_SYMBOLS, USE_MULTICLASS_MODELS

INTRADAY_LOOKBACK_DAYS = 60
INTRADAY_INTERVAL = "15m"
USE_MULTICLASS = USE_MULTICLASS_MODELS
FORCE_RETRAIN = True


def _save_artifact(artifact, symbol: str, mode: str) -> bool:
    """Save full artifact returned by patched train_model()."""
    model_dir = MODEL_DIR if MODEL_DIR else "models"
    os.makedirs(model_dir, exist_ok=True)

    path = os.path.join(model_dir, f"{symbol}_{mode}_xgb.pkl")
    abs_path = os.path.abspath(path)

    try:
        joblib.dump(artifact, path)

        if os.path.exists(path):
            file_size_kb = os.path.getsize(path) / 1024.0
            print(f"✅ Saved {symbol} {mode}: {abs_path} ({file_size_kb:.1f} KB)")
            return True

        print(f"❌ Failed to save {symbol} {mode} - file not found after save")
        return False

    except Exception as e:
        print(f"❌ Error saving {symbol} {mode}: {e}")
        traceback.print_exc()
        return False


def _print_artifact_summary(artifact: dict, symbol: str, mode: str) -> None:
    """Print concise summary from patched artifact."""
    metrics = artifact.get("metrics", {}) or {}
    split_metadata = artifact.get("split_metadata", {}) or {}
    threshold_info = artifact.get("threshold_optimization", {}) or {}
    features = artifact.get("features", []) or []
    full_feature_schema = artifact.get("full_feature_schema", []) or []

    print("\n📊 Model Summary")
    print(f"   Symbol: {symbol}")
    print(f"   Mode: {mode}")
    print(f"   Calibrated: {artifact.get('calibrated', False)}")
    print(f"   Target type: {artifact.get('target_type', 'binary')}")
    print(f"   Num classes: {artifact.get('num_classes', 2)}")
    print(f"   Selected features: {len(features)}")
    print(f"   Full feature schema: {len(full_feature_schema)}")

    if split_metadata:
        print(
            "   Split sizes: "
            f"train={split_metadata.get('train_rows')} "
            f"cal={split_metadata.get('cal_rows')} "
            f"test={split_metadata.get('test_rows')}"
        )

    if threshold_info:
        print(
            "   Threshold: "
            f"{threshold_info.get('best_threshold', artifact.get('decision_threshold', 0.5))}"
        )

    for key in ["accuracy", "logloss", "rocauc", "f1", "precision", "recall", "brier_score"]:
        val = metrics.get(key)
        if val is not None:
            print(f"   {key}: {val:.4f}")


def _train_regime_model(df, symbol: str, mode: str, use_multiclass: bool = True):
    """
    Train a single regime model using patched train_model().

    train_model() is responsible for:
    - building features
    - applying regime filtering
    - creating targets
    - chronological split
    - SHAP feature selection on calibration only
    - calibration
    - threshold optimization
    - final artifact assembly
    """
    print(f"\n{'=' * 60}")
    print(f"Training {symbol} - {mode.upper()}")
    print(f"{'=' * 60}")

    try:
        artifact = train_model(
            df,
            symbol=symbol,
            mode=mode,
            use_multiclass=use_multiclass,
        )

        if artifact is None:
            print(f"❌ Training returned None for {symbol} {mode}")
            return None

        required_keys = [
            "model",
            "features",
            "metrics",
            "trained_at",
            "full_feature_schema",
        ]
        missing_keys = [k for k in required_keys if k not in artifact]
        if missing_keys:
            print(f"⚠️ Warning: Artifact missing keys: {missing_keys}")

        _print_artifact_summary(artifact, symbol, mode)
        return artifact

    except ValueError as e:
        error_msg = str(e)
        if (
            "Insufficient class diversity" in error_msg
            or "not enough filtered data" in error_msg
            or "insufficient rows" in error_msg.lower()
        ):
            print(f"⚠️ Expected training skip for {symbol} {mode}:")
            print(f"   {error_msg}")
            print("   This can be normal when regime-filtered data is too small.")
        else:
            print(f"⚠️ ValueError for {symbol} {mode}: {e}")
        return None

    except Exception as e:
        print(f"❌ Unexpected error training {symbol} {mode}:")
        print(f"   {type(e).__name__}: {e}")
        traceback.print_exc()
        return None


def _print_summary(results: dict) -> None:
    """Print overall training summary."""
    print(f"\n{'=' * 60}")
    print("TRAINING SUMMARY")
    print(f"{'=' * 60}")

    total_models = len(results)
    successful = sum(1 for v in results.values() if v["success"])
    failed = total_models - successful

    print(f"Total models: {total_models}")
    print(f"Successful: {successful} ({(successful / total_models * 100):.1f}%)" if total_models else "Successful: 0")
    print(f"Failed: {failed} ({(failed / total_models * 100):.1f}%)" if total_models else "Failed: 0")

    print("\nDetailed Results:")
    print(f"{'Symbol':<8} {'Mode':<15} {'Status':<10} {'Reason'}")
    print(f"{'-' * 70}")

    for (sym, mode), info in sorted(results.items()):
        status = "✅ Success" if info["success"] else "❌ Failed"
        reason = info.get("reason", "")
        print(f"{sym:<8} {mode:<15} {status:<10} {reason}")

    print(f"{'=' * 60}\n")


def main():
    print(f"\n{'=' * 70}")
    print("🔄 WEEKEND INTRADAY DUAL-MODEL TRAINING")
    print(f"{'=' * 70}")
    print(f"📂 Model directory: {os.path.abspath(MODEL_DIR if MODEL_DIR else 'models')}")
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Multi-class mode: {'Enabled (5 classes)' if USE_MULTICLASS else 'Disabled (binary)'}")

    symbols = TRAIN_SYMBOLS if isinstance(TRAIN_SYMBOLS, list) else [TRAIN_SYMBOLS]
    print(f"📌 Symbols ({len(symbols)}): {', '.join(symbols)}")
    print(f"{'=' * 70}\n")

    results = {}

    for i, sym in enumerate(symbols, 1):
        print(f"\n{'#' * 70}")
        print(f"# SYMBOL {i}/{len(symbols)}: {sym}")
        print(f"{'#' * 70}")

        print(f"\n[DATA] Fetching {INTRADAY_LOOKBACK_DAYS}d of {INTRADAY_INTERVAL} data for {sym}...")

        try:
            df = fetch_historical_data(
                sym,
                period=f"{INTRADAY_LOOKBACK_DAYS}d",
                interval=INTRADAY_INTERVAL,
            )
        except Exception as e:
            print(f"❌ Data fetch failed for {sym}: {e}")
            results[(sym, "intraday_mr")] = {"success": False, "reason": "Data fetch failed"}
            results[(sym, "intraday_mom")] = {"success": False, "reason": "Data fetch failed"}
            continue

        if df is None or df.empty:
            print(f"❌ No intraday data for {sym}, skipping.")
            results[(sym, "intraday_mr")] = {"success": False, "reason": "No data"}
            results[(sym, "intraday_mom")] = {"success": False, "reason": "No data"}
            continue

        print(f"[DATA] ✅ Fetched {len(df)} bars for {sym}")

        df = _clean_columns(df)

        artifact_mr = _train_regime_model(
            df,
            symbol=sym,
            mode="intraday_mr",
            use_multiclass=USE_MULTICLASS,
        )
        if artifact_mr:
            success = _save_artifact(artifact_mr, sym, "intraday_mr")
            results[(sym, "intraday_mr")] = {
                "success": success,
                "reason": "Saved successfully" if success else "Save failed",
            }
        else:
            results[(sym, "intraday_mr")] = {
                "success": False,
                "reason": "Training failed",
            }

        artifact_mom = _train_regime_model(
            df,
            symbol=sym,
            mode="intraday_mom",
            use_multiclass=USE_MULTICLASS,
        )
        if artifact_mom:
            success = _save_artifact(artifact_mom, sym, "intraday_mom")
            results[(sym, "intraday_mom")] = {
                "success": success,
                "reason": "Saved successfully" if success else "Save failed",
            }
        else:
            results[(sym, "intraday_mom")] = {
                "success": False,
                "reason": "Training failed",
            }

    _print_summary(results)

    print(f"{'=' * 70}")
    print("🎉 Weekend intraday training complete!")
    print(f"📅 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print("\n\n❌ Fatal error in main():")
        print(f"   {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
