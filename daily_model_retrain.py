#!/usr/bin/env python3
"""Guarded daily retraining for degraded daily models.

The job reads horizon-resolved daily predictions from the unified prediction
logs, collapses repeated cycles to one observation per New York trading date,
and retrains only degraded daily models. A candidate is staged and validated
before it replaces the active artifact. Intraday models remain owned by the
weekly dual-regime trainer and the validated monthly full rebuild.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import smtplib
import sys
import traceback
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

import config
from monthly_model_retrain import _artifact_summary, validate_artifact
from predictive_model.data_loader import fetch_historical_data
from predictive_model.model_monitor import _prediction_logs_dir
from predictive_model.model_evaluation import promotion_decision
from predictive_model.model_xgb import train_model


DAILY_PERIOD = "3y"
DAILY_INTERVAL = "1d"
MIN_UNIQUE_DAILY_OUTCOMES = 20
MAX_MONITOR_DAYS = 60
MIN_MODEL_AGE_DAYS = 7

DEGRADED_MIN_ACCURACY = 0.52
DEGRADED_MAX_CALIBRATION_ERROR = 0.12
DEGRADED_MAX_BRIER = 0.27

CANDIDATE_MIN_ACCURACY = 0.52
CANDIDATE_MAX_CALIBRATION_ERROR = 0.10
CANDIDATE_MAX_BRIER = 0.27

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = Path(config.MODEL_DIR).resolve()
REPORT_DIR = PROJECT_ROOT / "logs" / "daily_retrain"
USE_MULTICLASS = bool(config.USE_MULTICLASS_MODELS)


def _symbols(requested: Optional[Iterable[str]] = None) -> List[str]:
    source = requested or config.TRAIN_SYMBOLS
    values = [str(symbol).strip().upper() for symbol in source if str(symbol).strip()]
    spy = str(config.SPY_SYMBOL).strip().upper()
    if spy and spy not in values:
        values.append(spy)
    return list(dict.fromkeys(values))


def _artifact_path(symbol: str) -> Path:
    return MODEL_DIR / f"{symbol}_daily_xgb.pkl"


def _load_artifact(symbol: str) -> dict:
    path = _artifact_path(symbol)
    if not path.exists():
        raise FileNotFoundError(f"active daily artifact not found: {path}")
    artifact = joblib.load(path)
    # Existing champions from correctness phase 1 do not yet contain the new
    # walk-forward metadata. They remain loadable for the one-time transition.
    validate_artifact(artifact, symbol, "daily", require_walk_forward=False)
    return artifact


def _model_age_days(artifact: dict) -> float:
    trained_at = pd.to_datetime(artifact.get("trained_at"), utc=True, errors="coerce")
    if pd.isna(trained_at):
        return float("inf")
    now = pd.Timestamp.now(tz="UTC")
    return max(0.0, (now - trained_at).total_seconds() / 86400.0)


def _prediction_path(symbol: str) -> Path:
    return Path(_prediction_logs_dir()) / f"predictions_{symbol}.csv"


def _daily_observations(symbol: str) -> pd.DataFrame:
    """Return one resolved daily prediction per New York trading date."""
    path = _prediction_path(symbol)
    if not path.exists():
        print(f"[MONITOR] {symbol}: prediction log not found yet: {path}")
        return pd.DataFrame(
            columns=[
                "timestamp",
                "mode",
                "predicted_prob",
                "actual_outcome",
                "prediction_date_ny",
            ]
        )

    frame = pd.read_csv(path)
    required = {"timestamp", "mode", "predicted_prob"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"prediction log missing columns: {missing}")

    if "actual_outcome" not in frame.columns:
        print(
            f"[MONITOR] {symbol}: actual_outcome column is not available yet; "
            "treating as no resolved daily outcomes"
        )
        return pd.DataFrame(
            columns=[
                "timestamp",
                "mode",
                "predicted_prob",
                "actual_outcome",
                "prediction_date_ny",
            ]
        )

    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["predicted_prob"] = pd.to_numeric(frame["predicted_prob"], errors="coerce")
    frame["actual_outcome"] = pd.to_numeric(frame["actual_outcome"], errors="coerce")
    frame = frame.loc[frame["mode"].astype(str).str.lower() == "daily"]
    frame = frame.dropna(subset=["timestamp", "predicted_prob", "actual_outcome"])
    frame = frame.loc[frame["actual_outcome"].isin([0, 1])]

    if frame.empty:
        return frame

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=MAX_MONITOR_DAYS)
    frame = frame.loc[frame["timestamp"] >= cutoff].copy()
    if frame.empty:
        return frame

    frame["prediction_date_ny"] = (
        frame["timestamp"].dt.tz_convert("America/New_York").dt.date
    )
    # Daily probability is normally stable within a cycle day. Keeping the last
    # resolved row prevents six-minute cycles from inflating the sample count.
    frame = (
        frame.sort_values("timestamp")
        .groupby("prediction_date_ny", as_index=False)
        .tail(1)
        .sort_values("timestamp")
    )
    return frame


def _monitor_metrics(frame: pd.DataFrame, decision_threshold: float) -> dict:
    probabilities = frame["predicted_prob"].astype(float).clip(0.0, 1.0).to_numpy()
    outcomes = frame["actual_outcome"].astype(int).to_numpy()
    predicted = (probabilities >= float(decision_threshold)).astype(int)
    return {
        "unique_daily_outcomes": int(len(frame)),
        "accuracy": float(np.mean(predicted == outcomes)),
        "calibration_error": float(abs(np.mean(probabilities) - np.mean(outcomes))),
        "brier_score": float(np.mean((probabilities - outcomes) ** 2)),
        "decision_threshold": float(decision_threshold),
        "first_prediction": frame["timestamp"].min().isoformat(),
        "last_prediction": frame["timestamp"].max().isoformat(),
    }


def _is_degraded(metrics: dict) -> Tuple[bool, List[str]]:
    reasons = []
    if metrics["accuracy"] < DEGRADED_MIN_ACCURACY:
        reasons.append(
            f"accuracy {metrics['accuracy']:.3f} < {DEGRADED_MIN_ACCURACY:.3f}"
        )
    if metrics["calibration_error"] > DEGRADED_MAX_CALIBRATION_ERROR:
        reasons.append(
            "calibration_error "
            f"{metrics['calibration_error']:.3f} > {DEGRADED_MAX_CALIBRATION_ERROR:.3f}"
        )
    if metrics["brier_score"] > DEGRADED_MAX_BRIER:
        reasons.append(
            f"brier_score {metrics['brier_score']:.3f} > {DEGRADED_MAX_BRIER:.3f}"
        )
    return bool(reasons), reasons


def _candidate_quality(artifact: dict) -> Tuple[bool, List[str]]:
    metrics = artifact.get("metrics") or {}
    reasons = []

    def number(name: str):
        value = metrics.get(name)
        try:
            value = float(value)
        except (TypeError, ValueError):
            reasons.append(f"candidate metric {name} is unavailable")
            return None
        if not math.isfinite(value):
            reasons.append(f"candidate metric {name} is not finite")
            return None
        return value

    accuracy = number("accuracy")
    calibration_error = number("calibration_error")
    brier = number("brier_score")

    if accuracy is not None and accuracy < CANDIDATE_MIN_ACCURACY:
        reasons.append(f"candidate accuracy {accuracy:.3f} < {CANDIDATE_MIN_ACCURACY:.3f}")
    if calibration_error is not None and calibration_error > CANDIDATE_MAX_CALIBRATION_ERROR:
        reasons.append(
            "candidate calibration_error "
            f"{calibration_error:.3f} > {CANDIDATE_MAX_CALIBRATION_ERROR:.3f}"
        )
    if brier is not None and brier > CANDIDATE_MAX_BRIER:
        reasons.append(f"candidate brier_score {brier:.3f} > {CANDIDATE_MAX_BRIER:.3f}")
    return not reasons, reasons


def _train_candidate(symbol: str, run_dir: Path) -> Tuple[dict, Path]:
    data = fetch_historical_data(symbol, period=DAILY_PERIOD, interval=DAILY_INTERVAL)
    if data is None or data.empty:
        raise ValueError("no daily training data returned")

    print(f"[DATA] {symbol}/daily: {len(data)} rows")
    artifact = train_model(
        data,
        symbol=symbol,
        mode="daily",
        use_multiclass=USE_MULTICLASS,
    )
    validate_artifact(artifact, symbol, "daily")
    accepted, reasons = _candidate_quality(artifact)
    if not accepted:
        raise ValueError("candidate rejected: " + "; ".join(reasons))

    active_path = _artifact_path(symbol)
    champion = joblib.load(active_path) if active_path.exists() else None
    decision = promotion_decision(artifact, champion)
    artifact["promotion_evaluation"] = decision
    if not decision["accepted"]:
        raise ValueError(
            "candidate rejected by champion/challenger gate: "
            + "; ".join(decision["reasons"])
        )
    print(
        f"[PROMOTION GATE] {symbol}/daily: accepted "
        f"({decision['kind']}, overlap={decision['overlap_samples']})"
    )

    candidate_path = run_dir / f"{symbol}_daily_xgb.pkl"
    joblib.dump(artifact, candidate_path)
    reloaded = joblib.load(candidate_path)
    validate_artifact(reloaded, symbol, "daily")
    accepted, reasons = _candidate_quality(reloaded)
    if not accepted:
        raise ValueError("serialized candidate rejected: " + "; ".join(reasons))
    return reloaded, candidate_path


def _promote_candidate(symbol: str, candidate_path: Path, run_id: str) -> Path:
    active = _artifact_path(symbol)
    backup_dir = MODEL_DIR / "daily_retrain_backups" / run_id
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / active.name

    if active.exists():
        shutil.copy2(active, backup)

    try:
        os.replace(candidate_path, active)
        validate_artifact(joblib.load(active), symbol, "daily")
        return backup
    except Exception:
        if backup.exists():
            shutil.copy2(backup, active)
        elif active.exists():
            active.unlink()
        raise


def _write_report(report: dict) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"daily_retrain_{report['run_id']}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    return path


def _send_email(success: bool, report: dict, report_path: Path) -> None:
    sender = getattr(config, "EMAIL_SENDER", None)
    password = getattr(config, "EMAIL_PASSWORD", None)
    receiver = getattr(config, "EMAIL_RECEIVER", None)
    if not all((sender, password, receiver)):
        print("[EMAIL] Email configuration incomplete; skipping notification.")
        return

    status = "SUCCESS" if success else "FAILED"
    lines = [
        f"Daily model guard: {status}",
        f"Run: {report['run_id']}",
        f"Retrained: {', '.join(report['retrained']) or 'none'}",
        f"Report: {report_path}",
    ]
    if report.get("failures"):
        lines.extend(["", "Failures:", *report["failures"]])

    message = EmailMessage()
    message["Subject"] = f"Daily model guard {status} - {report['run_id']}"
    message["From"] = sender
    message["To"] = receiver
    message.set_content("\n".join(lines))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(message)
        print("[EMAIL] Daily model guard notification sent.")
    except Exception as exc:
        print(f"[EMAIL] Notification failed: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", help="Optional symbol subset")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Evaluate degradation without training or replacing models",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Train requested symbols regardless of degradation or model age",
    )
    parser.add_argument("--no-email", action="store_true")
    args = parser.parse_args()

    symbols = _symbols(args.symbols)
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = MODEL_DIR / ".daily_retrain_staging" / run_id
    if not args.check_only:
        run_dir.mkdir(parents=True, exist_ok=False)

    report = {
        "run_id": run_id,
        "prediction_logs_dir": _prediction_logs_dir(),
        "symbols": symbols,
        "check_only": args.check_only,
        "force": args.force,
        "monitoring": {},
        "retrained": [],
        "skipped": [],
        "failures": [],
    }

    print(f"[DAILY] Run ID: {run_id}")
    print(f"[DAILY] Prediction logs: {_prediction_logs_dir()}")
    print(f"[DAILY] Model directory: {MODEL_DIR}")

    for symbol in symbols:
        print(f"\n{'=' * 72}\nDAILY MODEL CHECK: {symbol}\n{'=' * 72}")
        try:
            active = _load_artifact(symbol)
            age_days = _model_age_days(active)
            observations = _daily_observations(symbol)
            threshold = float(active.get("decision_threshold", 0.5))

            if len(observations) < MIN_UNIQUE_DAILY_OUTCOMES and not args.force:
                reason = (
                    f"insufficient unique daily outcomes: {len(observations)} "
                    f"< {MIN_UNIQUE_DAILY_OUTCOMES}"
                )
                print(f"[SKIP] {symbol}: {reason}")
                report["skipped"].append(f"{symbol}: {reason}")
                continue

            if observations.empty:
                metrics = {"unique_daily_outcomes": 0}
                degraded, reasons = False, []
            else:
                metrics = _monitor_metrics(observations, threshold)
                degraded, reasons = _is_degraded(metrics)

            metrics["model_age_days"] = age_days
            report["monitoring"][symbol] = metrics
            print(f"[MONITOR] {symbol}: {metrics}")

            if args.check_only:
                status = "degraded" if degraded else "healthy"
                report["skipped"].append(f"{symbol}: check-only ({status})")
                continue
            if not args.force and not degraded:
                report["skipped"].append(f"{symbol}: healthy")
                print(f"[SKIP] {symbol}: model is healthy")
                continue
            if not args.force and age_days < MIN_MODEL_AGE_DAYS:
                reason = f"degraded but only {age_days:.1f} days old"
                report["skipped"].append(f"{symbol}: {reason}")
                print(f"[SKIP] {symbol}: {reason}")
                continue

            trigger = "forced" if args.force else "; ".join(reasons)
            print(f"[RETRAIN] {symbol}: {trigger}")
            candidate, candidate_path = _train_candidate(symbol, run_dir)
            backup = _promote_candidate(symbol, candidate_path, run_id)
            report["retrained"].append(symbol)
            report.setdefault("candidates", {})[symbol] = _artifact_summary(candidate)
            print(f"[PROMOTED] {symbol}/daily; backup={backup}")
        except Exception as exc:
            message = f"{symbol}: {type(exc).__name__}: {exc}"
            report["failures"].append(message)
            print(f"[FAILED] {message}")
            traceback.print_exc()

    report["status"] = "failed" if report["failures"] else "success"
    report_path = _write_report(report)
    print(f"[DAILY] Report: {report_path}")

    if not args.no_email:
        _send_email(not report["failures"], report, report_path)

    if run_dir.exists() and not report["failures"]:
        shutil.rmtree(run_dir)

    if report["failures"]:
        print("[DAILY] FAILED")
        return 1
    print(f"[DAILY] SUCCESS - retrained {len(report['retrained'])} model(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
