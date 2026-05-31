#!/usr/bin/env python3
# retrain_model.py
"""
Legacy compatibility wrapper for retraining models while using the patched
train_model() pipeline.

Patched behavior:
- Uses train_model(...) as the source of truth.
- Saves the full returned artifact unchanged.
- Does not apply duplicate legacy regime filtering outside train_model().
- Keeps monthly backups.
"""

import os
import glob
import shutil
from datetime import datetime
import sys
import smtplib
from email.message import EmailMessage

import joblib

from predictive_model.data_loader import fetch_historical_data, fetch_intraday_history
from predictive_model.model_xgb import train_model, MODEL_DIR
from config.config import TRAIN_SYMBOLS, USE_MULTICLASS_MODELS


MAX_BACKUPS = 6
USE_MULTICLASS = USE_MULTICLASS_MODELS

DAILY_PERIOD = "3y"
DAILY_INTERVAL = "1d"

INTRADAY_LOOKBACK = 2400
INTRADAY_INTERVAL = "15m"


# ---------------- existing helpers (unchanged) ----------------

def save_model_with_backup(artifact, symbol: str, mode: str = "daily"):
    """Save full patched artifact and maintain rolling monthly backups."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    active_path = os.path.join(MODEL_DIR, f"{symbol}_{mode}_xgb.pkl")
    joblib.dump(artifact, active_path)
    print(f"✅ Active {mode} model saved: {active_path}")

    month_tag = datetime.now().strftime("%Y-%m")
    backup_path = os.path.join(MODEL_DIR, f"{symbol}_{mode}_xgb_{month_tag}.pkl")

    if not os.path.exists(backup_path):
        shutil.copy2(active_path, backup_path)
        print(f"📦 Monthly backup created: {backup_path}")
    else:
        print(f"ℹ️ Monthly backup for {symbol} ({mode}) in {month_tag} already exists — skipping.")

    cleanup_old_backups(symbol, mode)


def cleanup_old_backups(symbol: str, mode: str):
    """Remove old backups, keeping only MAX_BACKUPS most recent."""
    pattern = os.path.join(MODEL_DIR, f"{symbol}_{mode}_xgb_*.pkl")
    backups = sorted(glob.glob(pattern))

    if len(backups) > MAX_BACKUPS:
        to_delete = backups[:-MAX_BACKUPS]
        for old in to_delete:
            try:
                os.remove(old)
                print(f"🗑️ Removed old backup: {os.path.basename(old)}")
            except Exception as e:
                print(f"[WARN] Could not remove backup {old}: {e}")


def print_model_summary(artifact: dict, symbol: str, mode: str):
    """Print concise summary from patched artifact."""
    metrics = artifact.get("metrics", {}) or {}
    split_metadata = artifact.get("split_metadata", {}) or {}
    threshold_info = artifact.get("threshold_optimization", {}) or {}

    print(f"\n{'─' * 60}")
    print(f"📊 {symbol} - {mode.upper()}")
    print(f"{'─' * 60}")
    print(f" Target type: {artifact.get('target_type', 'binary')}")
    print(f" Classes: {artifact.get('num_classes', 2)}")
    print(f" Calibrated: {artifact.get('calibrated', False)}")
    print(f" Selected features: {len(artifact.get('features', []))}")
    print(f" Full feature schema: {len(artifact.get('full_feature_schema', []))}")

    if split_metadata:
        print(
            f" Split sizes: train={split_metadata.get('train_rows')} "
            f"cal={split_metadata.get('cal_rows')} "
            f"test={split_metadata.get('test_rows')}"
        )

    best_threshold = artifact.get("decision_threshold")
    if best_threshold is None:
        best_threshold = threshold_info.get("best_threshold")
    if best_threshold is not None:
        print(f" Decision threshold: {best_threshold:.3f}")

    for key in ["accuracy", "logloss", "rocauc", "f1", "precision", "recall", "brier_score"]:
        val = metrics.get(key)
        if val is not None:
            print(f" {key}: {val:.3f}")

    print(f"{'─' * 60}")


def train_daily_model(sym: str):
    """Train daily model using patched train_model()."""
    print(f"\n{'=' * 60}")
    print(f"🔄 Training {sym} - DAILY MODEL")
    print(f"{'=' * 60}")

    df_daily = fetch_historical_data(sym, period=DAILY_PERIOD, interval=DAILY_INTERVAL)

    if df_daily is None or df_daily.empty:
        print(f"[ERROR] No daily data for {sym}. Skipping.")
        return

    try:
        artifact = train_model(
            df_daily,
            symbol=sym,
            mode="daily",
            use_multiclass=USE_MULTICLASS,
        )

        save_model_with_backup(artifact, symbol=sym, mode="daily")
        print_model_summary(artifact, sym, "daily")

    except Exception as e:
        print(f"[ERROR] Failed to train daily model for {sym}: {e}")
        import traceback
        traceback.print_exc()


def train_intraday_models(sym: str):
    """Train intraday MR and MOM models using patched train_model()."""
    print(f"\n{'=' * 60}")
    print(f"🔄 Training {sym} - INTRADAY MODELS")
    print(f"{'=' * 60}")

    df_intra = fetch_intraday_history(
        sym,
        lookback_minutes=INTRADAY_LOOKBACK,
        interval=INTRADAY_INTERVAL,
    )

    if df_intra is None or df_intra.empty:
        print(f"[ERROR] No intraday data for {sym}. Skipping.")
        return

    for mode in ["intraday_mr", "intraday_mom"]:
        try:
            print(f"\n🔧 Training {sym} {mode.upper()}...")
            artifact = train_model(
                df_intra,
                symbol=sym,
                mode=mode,
                use_multiclass=USE_MULTICLASS,
            )

            save_model_with_backup(artifact, symbol=sym, mode=mode)
            print_model_summary(artifact, sym, mode)

        except ValueError as e:
            msg = str(e)
            if (
                "Insufficient class diversity" in msg
                or "not enough filtered data" in msg
                or "Not enough rows for robust train/cal/test split" in msg
            ):
                print(f"⚠️ Skipped {sym}/{mode}: {msg}")
                continue
            else:
                print(f"[ERROR] Failed to train {mode} for {sym}: {e}")
                import traceback
                traceback.print_exc()

        except Exception as e:
            print(f"[ERROR] Failed to train {mode} for {sym}: {e}")
            import traceback
            traceback.print_exc()


# ---------------- email + main orchestration ----------------

def send_status_email(success: bool, details: str) -> None:
    """
    Send status email for daily retrain.
    Uses SMTP_* and RETRAIN_EMAIL_* environment variables.
    """
    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    status_symbol = "✅" if success else "❌"
    status_word = "Success" if success else "FAILED"

    subject = f"Model Retrain {status_symbol} {status_word} - {now_str}"
    body = (
        f"Model retrain status: {status_word}\n"
        f"Time: {now_str}\n\n"
        f"{details}\n"
    )

    sender = os.environ.get("RETRAIN_EMAIL_SENDER", os.environ.get("SMTP_USER", ""))
    receiver = os.environ.get("RETRAIN_EMAIL_RECEIVER", "roispinola@gmail.com")

    if not sender:
        print("[WARN] No RETRAIN_EMAIL_SENDER/SMTP_USER set; skipping status email.")
        return

    smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER", sender)
    smtp_pass = os.environ.get("SMTP_PASS", "")

    if not smtp_pass:
        print("[WARN] SMTP_PASS not set; skipping status email.")
        return

    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = receiver
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as s:
            s.starttls()
            s.login(smtp_user, smtp_pass)
            s.send_message(msg)
        print(f"[INFO] Retrain status email sent to {receiver}")
    except Exception as e:
        print(f"[ERROR] Failed to send retrain status email: {e}")


def main() -> int:
    symbols = TRAIN_SYMBOLS if isinstance(TRAIN_SYMBOLS, list) else [TRAIN_SYMBOLS]

    print("\n" + "=" * 80)
    print("MODEL RETRAINING (Patched artifact-compatible mode)")
    print("=" * 80)
    print(f"📌 Symbols: {symbols}")
    print(f"📂 MODEL_DIR: {os.path.abspath(MODEL_DIR)}")
    print(f"🎯 Target: {'Multi-class (5 classes)' if USE_MULTICLASS else 'Binary (2 classes)'}")
    print(f"🕒 {datetime.now().isoformat()}\n")

    try:
        for sym in symbols:
            train_daily_model(sym)
            train_intraday_models(sym)

        print("\n" + "=" * 80)
        print("🎉 All retraining tasks complete.")
        print(f"📂 Models saved to: {os.path.abspath(MODEL_DIR)}")
        print("=" * 80)

        details = "All models retrained successfully."
        send_status_email(success=True, details=details)
        return 0

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("[ERROR] Fatal error in retrain_model.main():", e)
        print(tb)
        details = f"Fatal error in retrain_model.main():\n{e}\n\n{tb}"
        send_status_email(success=False, details=details)
        return 1


if __name__ == "__main__":
    sys.exit(main())