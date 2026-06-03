#!/usr/bin/env python3
"""
auto_retrain.py

Automatic retraining script that:
- Analyzes recent live performance from prediction logs
- Decides which models are degraded
- Retrains them using the patched retrain_model.py pipeline
- Validates new models using accuracy + calibration_error + brier_score
- Keeps or rolls back models based on those metrics
"""

import os
import sys
from datetime import datetime
import joblib
import traceback
import shutil

import pandas as pd
import numpy as np

from config import (
    TRAIN_SYMBOLS,
    LOGS_DIR,
    MODEL_DIR,
)

# Use the patched retrain functions
from train_model.retrain_model import (
    train_daily_model,
    train_intraday_models,
)

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

DAILY_DEGRADATION_MIN_SAMPLES = 40
DAILY_DEGRADATION_MIN_ACC = 0.55
DAILY_DEGRADATION_MAX_CAL_ERR = 0.12  # for flagging as degraded

# Acceptance criteria for new daily models (from artifact metrics)
MIN_ACC = 0.55
MAX_CAL_ERR = 0.07   # 7%
MAX_BRIER = 0.27

INTRADAY_MIN_OUTCOMES = 25  # to even consider intraday performance

# -------------------------------------------------------------------
# HELPERS FOR READING LOGS
# -------------------------------------------------------------------

def _daily_log_path(symbol: str) -> str:
    return os.path.join(LOGS_DIR, f"predictions_{symbol}_daily.csv")

def _load_daily_eval(symbol: str):
    """
    Load recent daily prediction log for a symbol.
    Returns (accuracy, calibration_error, count) or (None, None, 0).
    """
    path = _daily_log_path(symbol)
    if not os.path.exists(path):
        return None, None, 0

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Failed to read daily log for {symbol}: {e}")
        return None, None, 0

    if "actual_outcome" not in df.columns or "predicted_prob" not in df.columns:
        return None, None, len(df)

    df = df.copy()
    df = df.dropna(subset=["actual_outcome", "predicted_prob"])
    if len(df) == 0:
        return None, None, 0

    y = df["actual_outcome"].astype(int)
    p = df["predicted_prob"].astype(float)

    # Simple 0.5 threshold accuracy for diagnostics
    y_hat = (p >= 0.5).astype(int)
    acc = (y_hat == y).mean()

    # Calibration error: |mean(p) - mean(y)|
    cal_err = float(abs(p.mean() - y.mean()))

    return float(acc), float(cal_err), int(len(df))


def _intraday_log_path(symbol: str, mode: str) -> str:
    # e.g. predictions_SPY_intraday_mr.csv
    return os.path.join(LOGS_DIR, f"predictions_{symbol}_{mode}.csv")

def _load_intraday_eval(symbol: str, mode: str):
    path = _intraday_log_path(symbol, mode)
    if not os.path.exists(path):
        return None, 0

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Failed to read intraday log for {symbol}/{mode}: {e}")
        return None, 0

    if "actual_outcome" not in df.columns or "predicted_prob" not in df.columns:
        return None, len(df)

    df = df.copy()
    df = df.dropna(subset=["actual_outcome", "predicted_prob"])
    if len(df) == 0:
        return None, 0

    y = df["actual_outcome"].astype(int)
    p = df["predicted_prob"].astype(float)
    y_hat = (p >= 0.5).astype(int)
    acc = (y_hat == y).mean()
    return float(acc), int(len(df))


# -------------------------------------------------------------------
# MODEL ARTIFACT HELPERS
# -------------------------------------------------------------------

def _artifact_path(symbol: str, mode: str) -> str:
    """
    Current active model path (patched artifact).
    """
    return os.path.join(MODEL_DIR, f"{symbol}_{mode}_xgb.pkl")

def _backup_path(symbol: str, mode: str) -> str:
    """
    Temporary backup used by auto_retrain for rollback.
    """
    return os.path.join(MODEL_DIR, f"{symbol}_{mode}_xgb.pkl.backup")


def _backup_active_model(symbol: str, mode: str) -> bool:
    """
    Make a .backup copy of the current active model, if it exists.
    """
    active = _artifact_path(symbol, mode)
    backup = _backup_path(symbol, mode)

    if not os.path.exists(active):
        print(f"  [INFO] No active {symbol}/{mode} model to back up.")
        return False

    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        # simple overwrite backup
        if os.path.exists(backup):
            os.remove(backup)
        shutil.copy2(active, backup)
        print(f"  💾 Backed up existing model to {backup}")
        return True
    except Exception as e:
        print(f"  [WARN] Failed to create backup for {symbol}/{mode}: {e}")
        return False


def _restore_backup_model(symbol: str, mode: str) -> None:
    """
    Restore from .backup if a new retrain fails validation.
    """
    active = _artifact_path(symbol, mode)
    backup = _backup_path(symbol, mode)

    if not os.path.exists(backup):
        print(f"  [WARN] No backup found to restore for {symbol}/{mode}.")
        return

    try:
        if os.path.exists(active):
            os.remove(active)
        shutil.copy2(backup, active)
        print(f"  ↩️ Restored backup model")
    except Exception as e:
        print(f"  [ERROR] Failed to restore backup for {symbol}/{mode}: {e}")


def _load_artifact(symbol: str, mode: str):
    path = _artifact_path(symbol, mode)
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"[WARN] Failed to load artifact {path}: {e}")
        return None


# -------------------------------------------------------------------
# MAIN CHECK + RETRAIN LOGIC
# -------------------------------------------------------------------

def check_and_retrain():
    symbols = TRAIN_SYMBOLS if isinstance(TRAIN_SYMBOLS, list) else [TRAIN_SYMBOLS]

    print("\n" + "=" * 80)
    print(f"🔍 AUTO-RETRAIN CHECK - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")

    degraded_daily = []   # list of symbols
    # intraday degradation detection is currently informational only
    intraday_info = {}

    # ------------------ DIAGNOSTICS PHASE -------------------

    for sym in symbols:
        print(f"\n📊 Checking {sym}...")

        # Daily
        daily_acc, daily_cal_err, daily_n = _load_daily_eval(sym)
        if daily_acc is not None:
            print(f"  Daily: {daily_n} samples, acc={daily_acc*100:.1f}%, "
                  f"cal_err={daily_cal_err*100:.1f}%")
        else:
            print(f"  Daily: No usable outcomes")

        degraded = False
        if daily_acc is not None and daily_n >= DAILY_DEGRADATION_MIN_SAMPLES:
            if daily_acc < DAILY_DEGRADATION_MIN_ACC or daily_cal_err > DAILY_DEGRADATION_MAX_CAL_ERR:
                print("  ⚠️ Daily model degraded!")
                degraded = True

        if degraded:
            degraded_daily.append(sym)

        # Intraday (just log for now)
        intraday_info[sym] = {}
        for mode in ["intraday_mr", "intraday_mom"]:
            acc_i, n_i = _load_intraday_eval(sym, mode)
            if acc_i is None or n_i < INTRADAY_MIN_OUTCOMES:
                print(f"  {mode}: Insufficient data ({n_i} samples)")
                intraday_info[sym][mode] = ("insufficient", acc_i, n_i)
            else:
                print(f"  {mode}: {n_i} samples, acc={acc_i*100:.1f}%")
                intraday_info[sym][mode] = ("ok", acc_i, n_i)

    print("\n" + "=" * 80)
    print("🔄 RETRAINING SUMMARY")
    print("=" * 80 + "\n")

    # ------------------ RETRAINING PHASE (DAILY) -------------------

    for sym in degraded_daily:
        mode = "daily"
        print("─" * 60)
        print(f"🔄 Retraining {sym} {mode} model...")
        print("─" * 60)

        # Backup current active model
        _backup_active_model(sym, mode)

        try:
            # This calls the patched train_daily_model, which:
            # - fetches data
            # - calls train_model(...)
            # - saves the artifact
            # - prints its own summary
            train_daily_model(sym)

            # Now load the newly trained artifact to validate
            artifact = _load_artifact(sym, mode)
            if artifact is None:
                print(f"  ❌ Could not load new {sym}/{mode} artifact after retrain.")
                _restore_backup_model(sym, mode)
                continue

            metrics = artifact.get("metrics", {}) or {}
            acc = metrics.get("accuracy")
            logloss = metrics.get("logloss")
            cal_err = metrics.get("calibration_error")
            brier = metrics.get("brier_score")

            print("\n" + "─" * 60)
            print(f"📊 {sym} - {mode.upper()}")
            print("─" * 60)
            print(f"  Target type: {artifact.get('target_type', 'binary')}")
            print(f"  Classes: {artifact.get('num_classes', 2)}")
            print(f"  Calibrated: {artifact.get('calibrated', False)}")
            print(f"  Selected features: {len(artifact.get('features', []))}")
            full_schema = artifact.get('full_feature_schema') or {}
            print(f"  Full feature schema: {len(full_schema) if isinstance(full_schema, dict) else full_schema}")

            print("─" * 60)
            print("  📊 Validation metrics:")
            if acc is not None:
                print(f"     Accuracy: {acc:.3f}")
            else:
                print("     Accuracy: N/A")
            if logloss is not None:
                print(f"     LogLoss: {logloss:.3f}")
            else:
                print("     LogLoss: N/A")
            if cal_err is not None:
                print(f"     Calibration Error: {cal_err:.3f}")
            else:
                print("     Calibration Error: N/A")
            if brier is not None:
                print(f"     Brier Score: {brier:.3f}")
            else:
                print("     Brier Score: N/A")

            # Combined acceptance decision
            ok = True

            if acc is None or cal_err is None or brier is None:
                ok = False
            else:
                if acc < MIN_ACC:
                    ok = False
                if cal_err > MAX_CAL_ERR:
                    ok = False
                if brier > MAX_BRIER:
                    ok = False

            if ok:
                print(f"  ✅ PASS: Model meets quality standards")
                print(f"✅ {sym} {mode} retrained and validated")
                # keep new model (backup stays as historical snapshot)
            else:
                print(f"  ❌ REJECT: "
                      f"Accuracy {acc*100:.1f}% < {MIN_ACC*100:.1f}% "
                      f"or Calibration Error/Brier too high")
                print(f"❌ {sym} {mode} FAILED VALIDATION - restoring backup")
                _restore_backup_model(sym, mode)
                print(f"⚠️ Manual review required for {sym} {mode} model!")

        except Exception as e:
            print(f"❌ {sym} {mode} training failed: {e}")
            traceback.print_exc()
            _restore_backup_model(sym, mode)

    print("\n" + "=" * 80)
    print("✅ Auto-retrain complete!")
    print("=" * 80 + "\n")


def main():
    try:
        check_and_retrain()
        return 0
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
        return 1
    except Exception as e:
        print(f"[ERROR] Fatal error in auto_retrain.main(): {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())