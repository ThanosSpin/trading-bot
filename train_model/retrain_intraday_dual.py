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

from predictive_model.data_loader import fetch_intraday_history
from predictive_model.features import _clean_columns
from predictive_model.model_xgb import train_model, MODEL_DIR
from predictive_model.model_evaluation import promotion_decision
from config import TRAIN_SYMBOLS, USE_MULTICLASS_MODELS, USE_TWO_STAGE_TARGETS

INTRADAY_LOOKBACK = 40000
INTRADAY_INTERVAL = "15m"
USE_MULTICLASS = USE_MULTICLASS_MODELS
USE_TWO_STAGE = USE_TWO_STAGE_TARGETS
FORCE_RETRAIN = True


def _save_artifact(artifact, symbol: str, mode: str) -> bool:
    model_dir = MODEL_DIR if MODEL_DIR else "models"
    os.makedirs(model_dir, exist_ok=True)

    active_path = os.path.join(
        model_dir,
        f"{symbol}_{mode}_xgb.pkl",
    )
    challenger_path = os.path.join(
        model_dir,
        f"{symbol}_{mode}_xgb_challenger.pkl",
    )

    try:
        champion = (
            joblib.load(active_path)
            if os.path.exists(active_path)
            else None
        )

        decision = promotion_decision(artifact, champion)
        artifact["promotion_evaluation"] = decision

        if not decision["accepted"]:
            challenger_path = os.path.join(
                model_dir,
                f"{symbol}_{mode}_xgb_rejected.pkl",
            )

            joblib.dump(artifact, challenger_path)

            print(
                f"[PROMOTION REJECTED] {symbol}/{mode}: "
                + "; ".join(decision["reasons"])
            )
            print(f"Rejected challenger saved to: {challenger_path}")
            return False

        temporary_path = f"{active_path}.challenger"
        joblib.dump(artifact, temporary_path)
        os.replace(temporary_path, active_path)

        print(
            f"✅ Promoted {symbol} {mode}: {active_path} "
            f"({decision['kind']}, "
            f"overlap={decision['overlap_samples']})"
        )
        return True

    except Exception as e:
        print(f"❌ Error saving {symbol} {mode}: {e}")
        traceback.print_exc()
        return False


def _print_artifact_summary(artifact: dict, symbol: str, mode: str) -> None:
    """Print concise summary from a trained model artifact."""
    metrics = artifact.get("metrics") or {}
    split_metadata = artifact.get("split_metadata") or {}
    feature_schema = artifact.get("feature_schema") or {}
    full_feature_schema = artifact.get("full_feature_schema") or {}

    selected_features = []
    if isinstance(feature_schema, dict):
        selected_features = feature_schema.get("features", []) or []
    if not selected_features:
        selected_features = artifact.get("features", []) or []
        if isinstance(selected_features, dict):
            selected_features = selected_features.get("features", []) or []

    if isinstance(full_feature_schema, dict):
        full_features = full_feature_schema.get("features", []) or []
        full_feature_count = full_feature_schema.get("feature_count", len(full_features))
    else:
        full_features = full_feature_schema or []
        full_feature_count = len(full_features)

    if not isinstance(split_metadata, dict):
        split_metadata = {}

    train_rows = split_metadata.get("train_samples", split_metadata.get("train_rows"))
    cal_rows = split_metadata.get("calibration_samples", split_metadata.get("cal_rows"))
    test_rows = split_metadata.get("test_samples", split_metadata.get("test_rows"))

    selected_count = len(selected_features)

    try:
        full_feature_count = int(full_feature_count)
    except (TypeError, ValueError):
        full_feature_count = len(full_features)

    print("\n📊 Model Summary")
    print(f"   Symbol: {symbol}")
    print(f"   Mode: {mode}")
    print(f"   Calibrated: {artifact.get('calibrated', False)}")
    print(f"   Target type: {artifact.get('target_type', 'binary')}")
    print(f"   Num classes: {artifact.get('num_classes', 2)}")
    print(f"   Selected features: {selected_count}")
    print(f"   Full feature schema: {full_feature_count}")
    print(f"   Split sizes: train={train_rows} cal={cal_rows} test={test_rows}")

    threshold = artifact.get("decision_threshold")
    if threshold is not None:
        print(f"   Threshold: {float(threshold):.2f}")

    metric_aliases = {
        "accuracy": "accuracy",
        "logloss": "logloss",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "brier_score": "brier_score",
    }

    for label, key in metric_aliases.items():
        value = metrics.get(key)
        if value is not None:
            try:
                print(f"   {label}: {float(value):.4f}")
            except (TypeError, ValueError):
                print(f"   {label}: {value}")


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
            use_two_stage=USE_TWO_STAGE,
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
        msg = str(e)

        expected_skip_reasons = (
            "Insufficient class diversity",
            "not enough filtered data",
            "Not enough rows for robust train/cal/test split",
            "At least 100 chronological rows are required",
            "Invalid chronological split boundaries",
            "Train set has fewer than 2 classes",
            "Calibration set has fewer than 2 classes",
            "Two-stage movement model requires two train/calibration classes",
            "No valid regime rows",
            "Insufficient pre-split rows to fit regime thresholds",
            "Regime thresholds are not finite",
            "lacks two classes in fit/calibration",
        )

        if (
            any(reason in msg for reason in expected_skip_reasons)
            or "walk-forward" in msg.lower()
            or "fold" in msg.lower()
            or "insufficient rows" in msg.lower()
        ):
            print(f"⚠️ Skipped {symbol}/{mode}: {msg}")
            return None

        print(f"⚠️ ValueError for {symbol}/{mode}: {e}")
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
    successful = sum(1 for v in results.values() if v.get("status") == "promoted")
    rejected = sum(1 for v in results.values() if v.get("status") == "rejected")
    failed = sum(
        1 for v in results.values()
        if v.get("status") in ("training_failed", "save_error")
    )

    print(f"Total models: {total_models}")
    print(f"Successful: {successful} ({(successful / total_models * 100):.1f}%)" if total_models else "Successful: 0")
    print(f"Rejected: {rejected} ({(rejected / total_models * 100):.1f}%)" if total_models else "Rejected: 0")
    print(f"Failed: {failed} ({(failed / total_models * 100):.1f}%)" if total_models else "Failed: 0")

    print("\nDetailed Results:")
    print(f"{'Symbol':<8} {'Mode':<15} {'Status':<12} {'Reason'}")
    print(f"{'-' * 80}")

    for (sym, mode), info in sorted(results.items()):
        raw_status = info.get("status")
        status_map = {
            "promoted": "✅ Success",
            "rejected": "⛔ Rejected",
            "training_failed": "❌ Failed",
            "save_error": "❌ SaveErr",
        }
        status = status_map.get(raw_status, f"⚠️ Unknown({raw_status})")
        reason = info.get("reason", "")
        print(f"{sym:<8} {mode:<15} {status:<18} {reason}")

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

        print(f"\n[DATA] Fetching {INTRADAY_LOOKBACK}m of {INTRADAY_INTERVAL} data for {sym}...")

        try:
            df = fetch_intraday_history(
                sym,
                lookback_minutes=INTRADAY_LOOKBACK,
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
            promoted = _save_artifact(artifact_mr, sym, "intraday_mr")
            results[(sym, "intraday_mr")] = {
                "success": bool(promoted),
                "status": "Promoted" if promoted else "Promotion rejected",
                "reason": "Promoted" if promoted else "Promotion rejected",
            }
        else:
            results[(sym, "intraday_mr")] = {
                "success": False,
                "status": "Training failed",
                "reason": "Training failed",
            }

        artifact_mom = _train_regime_model(
            df,
            symbol=sym,
            mode="intraday_mom",
            use_multiclass=USE_MULTICLASS,
        )
        if artifact_mom:
            promoted = _save_artifact(artifact_mom, sym, "intraday_mom")
            results[(sym, "intraday_mom")] = {
                "success": bool(promoted),
                "status": "Promoted" if promoted else "Promotion rejected",
                "reason": "Promoted" if promoted else "Promotion rejected",
            }
        else:
            results[(sym, "intraday_mom")] = {
                "success": False,
                "status": "Training failed",
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
