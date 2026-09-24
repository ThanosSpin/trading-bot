#!/usr/bin/env python3
"""Validated monthly retraining for daily and dual intraday models.

This job is intentionally separate from the weekly trainers. It always builds a
complete batch, validates every artifact, and promotes nothing unless all
expected models train successfully.
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
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib

import config
from predictive_model.data_loader import fetch_historical_data
from predictive_model.model_evaluation import promotion_decision
from predictive_model.model_xgb import train_model


DAILY_PERIOD = "3y"
DAILY_INTERVAL = "1d"
INTRADAY_PERIOD = "60d"
INTRADAY_INTERVAL = "15m"
MODES = ("daily", "intraday_mr", "intraday_mom")
MAX_MONTHLY_BACKUPS = 6

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = Path(config.MODEL_DIR).resolve()
REPORT_DIR = PROJECT_ROOT / "logs" / "monthly_retrain"
USE_MULTICLASS = bool(config.USE_MULTICLASS_MODELS)
USE_TWO_STAGE = bool(config.USE_TWO_STAGE_TARGETS)


def _symbols(requested: Optional[Iterable[str]] = None) -> List[str]:
    source = requested if requested is not None else config.TRAIN_SYMBOLS
    values = [str(symbol).strip().upper() for symbol in source if str(symbol).strip()]
    if requested is None:
        spy = str(config.SPY_SYMBOL).strip().upper()
        if spy and spy not in values:
            values.append(spy)
    return list(dict.fromkeys(values))


def _finite_positive(value) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0


def validate_artifact(
    artifact: dict,
    symbol: str,
    mode: str,
    require_walk_forward: bool = True,
) -> None:
    """Raise ValueError when an artifact is unsafe to promote."""
    if not isinstance(artifact, dict):
        raise ValueError("training did not return an artifact dictionary")

    required = {
        "model",
        "features",
        "full_feature_schema",
        "split_metadata",
        "metrics",
        "trained_at",
        "symbol",
        "mode",
        "calibrated",
    }
    missing = sorted(required.difference(artifact))
    if missing:
        raise ValueError(f"artifact missing required keys: {missing}")

    if str(artifact.get("symbol", "")).upper() != symbol:
        raise ValueError(f"artifact symbol mismatch: {artifact.get('symbol')!r}")
    if artifact.get("mode") != mode:
        raise ValueError(f"artifact mode mismatch: {artifact.get('mode')!r}")
    if not artifact.get("features"):
        raise ValueError("artifact has no selected features")
    if not artifact.get("calibrated"):
        raise ValueError("artifact calibration did not succeed")
    if artifact.get("target_type") == "two_stage":
        if artifact.get("movement_model") is None:
            raise ValueError("two-stage artifact has no movement_model")
        if not _finite_positive(artifact.get("movement_threshold")):
            raise ValueError("two-stage artifact has an invalid movement_threshold")
        expected_horizon = "next_trading_session" if mode == "daily" else "60min"
        if artifact.get("target_horizon") != expected_horizon:
            raise ValueError(
                f"two-stage target horizon mismatch: {artifact.get('target_horizon')!r}"
            )

    split = artifact.get("split_metadata") or {}
    for key in ("train_samples", "calibration_samples", "test_samples"):
        if not _finite_positive(split.get(key)):
            raise ValueError(f"invalid split metadata: {key}={split.get(key)!r}")
    if split.get("windows_overlap") is not False:
        raise ValueError("artifact does not confirm non-overlapping time windows")

    if require_walk_forward:
        walk_forward = artifact.get("walk_forward_evaluation")
        if not isinstance(walk_forward, dict):
            raise ValueError("artifact has no walk_forward_evaluation")
        if int(walk_forward.get("n_splits", 0)) < 3:
            raise ValueError("artifact has fewer than three walk-forward folds")
        if int(walk_forward.get("gap_bars", 0)) < 1:
            raise ValueError("artifact walk-forward evaluation has no embargo gap")
        if not isinstance(walk_forward.get("aggregate"), dict):
            raise ValueError("artifact has no aggregate walk-forward metrics")
        if not walk_forward.get("predictions"):
            raise ValueError("artifact has no out-of-fold prediction records")

    if mode.startswith("intraday_"):
        regime = artifact.get("regime_config")
        if not isinstance(regime, dict):
            raise ValueError("intraday artifact has no persisted regime_config")
        for key in ("momentum_threshold", "volatility_threshold"):
            if not _finite_positive(regime.get(key)):
                raise ValueError(f"invalid regime threshold: {key}={regime.get(key)!r}")
        if not regime.get("fit_end"):
            raise ValueError("regime_config is missing fit_end")


def _artifact_summary(artifact: dict) -> dict:
    split = artifact.get("split_metadata") or {}
    metrics = artifact.get("metrics") or {}
    regime = artifact.get("regime_config") or {}
    walk_forward = artifact.get("walk_forward_evaluation") or {}
    wf_metrics = walk_forward.get("aggregate") or {}
    promotion = artifact.get("promotion_evaluation") or {}
    return {
        "trained_at": artifact.get("trained_at"),
        "calibrated": artifact.get("calibrated"),
        "target_type": artifact.get("target_type"),
        "feature_count": len(artifact.get("features") or []),
        "train_samples": split.get("train_samples"),
        "calibration_samples": split.get("calibration_samples"),
        "test_samples": split.get("test_samples"),
        "decision_threshold": artifact.get("decision_threshold"),
        "accuracy": metrics.get("accuracy"),
        "logloss": metrics.get("logloss"),
        "brier_score": metrics.get("brier_score"),
        "momentum_threshold": regime.get("momentum_threshold"),
        "volatility_threshold": regime.get("volatility_threshold"),
        "regime_fit_end": regime.get("fit_end"),
        "walk_forward_folds": walk_forward.get("n_splits"),
        "walk_forward_gap_bars": walk_forward.get("gap_bars"),
        "walk_forward_cost_bps": walk_forward.get("cost_bps"),
        "walk_forward_trades": wf_metrics.get("trade_count"),
        "walk_forward_net_return": wf_metrics.get("net_return"),
        "walk_forward_profit_factor": wf_metrics.get("profit_factor"),
        "walk_forward_max_drawdown": wf_metrics.get("max_drawdown"),
        "walk_forward_brier": wf_metrics.get("brier_score"),
        "promotion": promotion,
    }


def _fetch_training_data(symbol: str, mode: str):
    if mode == "daily":
        return fetch_historical_data(symbol, period=DAILY_PERIOD, interval=DAILY_INTERVAL)
    return fetch_historical_data(symbol, period=INTRADAY_PERIOD, interval=INTRADAY_INTERVAL)


def _train_to_stage(
    symbols: List[str],
    stage_dir: Path,
    enforce_promotion_gate: bool = True,
) -> Tuple[dict, List[str]]:
    summaries: Dict[str, dict] = {}
    failures: List[str] = []

    for symbol in symbols:
        intraday_data = None
        for mode in MODES:
            label = f"{symbol}/{mode}"
            print(f"\n{'=' * 72}\nMONTHLY TRAINING: {label}\n{'=' * 72}")
            try:
                if mode == "daily":
                    data = _fetch_training_data(symbol, mode)
                else:
                    if intraday_data is None:
                        intraday_data = _fetch_training_data(symbol, mode)
                    data = intraday_data

                if data is None or data.empty:
                    raise ValueError("no training data returned")
                print(f"[DATA] {label}: {len(data)} rows")

                artifact = train_model(
                    data,
                    symbol=symbol,
                    mode=mode,
                    use_multiclass=USE_MULTICLASS,
                    use_two_stage=USE_TWO_STAGE,
                )
                validate_artifact(artifact, symbol, mode)

                active_path = MODEL_DIR / f"{symbol}_{mode}_xgb.pkl"
                champion = joblib.load(active_path) if active_path.exists() else None
                decision = promotion_decision(artifact, champion)
                artifact["promotion_evaluation"] = decision
                if enforce_promotion_gate and not decision["accepted"]:
                    raise ValueError(
                        "challenger rejected: " + "; ".join(decision["reasons"])
                    )
                gate_status = "accepted" if decision["accepted"] else "rejected"
                print(
                    f"[PROMOTION GATE] {label}: {gate_status} "
                    f"({decision['kind']}, overlap={decision['overlap_samples']})"
                )
                if not decision["accepted"]:
                    print("[PROMOTION GATE] " + "; ".join(decision["reasons"]))

                stage_path = stage_dir / f"{symbol}_{mode}_xgb.pkl"
                joblib.dump(artifact, stage_path)

                # Validate the serialized object, not only the in-memory result.
                reloaded = joblib.load(stage_path)
                validate_artifact(reloaded, symbol, mode)
                summaries[label] = _artifact_summary(reloaded)
                print(f"[STAGED] {label}: {stage_path}")
            except Exception as exc:
                message = f"{label}: {type(exc).__name__}: {exc}"
                failures.append(message)
                print(f"[FAILED] {message}")
                traceback.print_exc()

    return summaries, failures


def _cleanup_old_backups() -> None:
    backup_root = MODEL_DIR / "monthly_backups"
    if not backup_root.exists():
        return
    month_dirs = sorted(path for path in backup_root.iterdir() if path.is_dir())
    for old_dir in month_dirs[:-MAX_MONTHLY_BACKUPS]:
        shutil.rmtree(old_dir)
        print(f"[BACKUP] Removed old monthly backup: {old_dir}")


def _promote_batch(stage_dir: Path, symbols: List[str]) -> None:
    """Back up active artifacts and atomically replace each file."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    month_dir = MODEL_DIR / "monthly_backups" / datetime.now().strftime("%Y-%m")
    rollback_dir = stage_dir / "rollback"
    month_dir.mkdir(parents=True, exist_ok=True)
    rollback_dir.mkdir(parents=True, exist_ok=True)

    filenames = [f"{symbol}_{mode}_xgb.pkl" for symbol in symbols for mode in MODES]
    existed = {}

    for filename in filenames:
        active = MODEL_DIR / filename
        existed[filename] = active.exists()
        if active.exists():
            shutil.copy2(active, rollback_dir / filename)
            monthly_backup = month_dir / filename
            if not monthly_backup.exists():
                shutil.copy2(active, monthly_backup)

    promoted: List[str] = []
    try:
        for filename in filenames:
            staged = stage_dir / filename
            active = MODEL_DIR / filename
            if not staged.exists():
                raise FileNotFoundError(f"staged artifact missing: {staged}")
            os.replace(staged, active)
            promoted.append(filename)
            print(f"[PROMOTED] {active}")
    except Exception:
        print("[ROLLBACK] Promotion failed; restoring previous active artifacts.")
        for filename in promoted:
            active = MODEL_DIR / filename
            rollback = rollback_dir / filename
            if existed.get(filename) and rollback.exists():
                shutil.copy2(rollback, active)
            elif active.exists():
                active.unlink()
        raise

    _cleanup_old_backups()


def _write_report(report: dict) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"monthly_retrain_{report['run_id']}.json"
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
        f"Monthly model retrain: {status}",
        f"Run: {report['run_id']}",
        f"Models expected: {report['expected_models']}",
        f"Models validated: {len(report.get('artifacts', {}))}",
        f"Report: {report_path}",
    ]
    if report.get("failures"):
        lines.extend(["", "Failures:", *report["failures"]])

    message = EmailMessage()
    message["Subject"] = f"Monthly model retrain {status} - {report['run_id']}"
    message["From"] = sender
    message["To"] = receiver
    message.set_content("\n".join(lines))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(message)
        print("[EMAIL] Monthly retrain notification sent.")
    except Exception as exc:
        print(f"[EMAIL] Notification failed: {exc}")


def _validate_existing(symbols: List[str]) -> Tuple[dict, List[str]]:
    summaries = {}
    failures = []
    for symbol in symbols:
        for mode in MODES:
            label = f"{symbol}/{mode}"
            path = MODEL_DIR / f"{symbol}_{mode}_xgb.pkl"
            try:
                artifact = joblib.load(path)
                validate_artifact(artifact, symbol, mode)
                summaries[label] = _artifact_summary(artifact)
                print(f"[VALID] {label}: {path}")
            except Exception as exc:
                message = f"{label}: {type(exc).__name__}: {exc}"
                failures.append(message)
                print(f"[INVALID] {message}")
    return summaries, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", help="Optional symbol subset")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate active artifacts without fetching data or training",
    )
    parser.add_argument(
        "--shadow",
        action="store_true",
        help=(
            "Train and evaluate all challengers, record promotion decisions, "
            "and replace no active artifacts"
        ),
    )
    parser.add_argument("--no-email", action="store_true")
    args = parser.parse_args()
    if args.validate_only and args.shadow:
        parser.error("--validate-only and --shadow cannot be used together")

    symbols = _symbols(args.symbols)
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    expected_models = len(symbols) * len(MODES)
    stage_dir = MODEL_DIR / ".monthly_retrain_staging" / run_id
    if not args.validate_only:
        stage_dir.mkdir(parents=True, exist_ok=False)

    print(f"[MONTHLY] Run ID: {run_id}")
    print(f"[MONTHLY] Symbols: {', '.join(symbols)}")
    print(f"[MONTHLY] Modes: {', '.join(MODES)}")
    print(f"[MONTHLY] Model directory: {MODEL_DIR}")
    print(f"[MONTHLY] Intraday history: {INTRADAY_PERIOD} @ {INTRADAY_INTERVAL}")
    print(f"[MONTHLY] Multiclass: {USE_MULTICLASS}")
    print(f"[MONTHLY] Two-stage targets: {USE_TWO_STAGE}")

    if args.validate_only:
        summaries, failures = _validate_existing(symbols)
    else:
        summaries, failures = _train_to_stage(
            symbols,
            stage_dir,
            enforce_promotion_gate=not args.shadow,
        )

    report = {
        "run_id": run_id,
        "status": "failed" if failures else "success",
        "validate_only": args.validate_only,
        "shadow": args.shadow,
        "symbols": symbols,
        "modes": list(MODES),
        "expected_models": expected_models,
        "artifacts": summaries,
        "failures": failures,
    }

    if not failures and len(summaries) != expected_models:
        failures.append(
            f"expected {expected_models} validated artifacts, got {len(summaries)}"
        )
        report["status"] = "failed"

    if not failures and not args.validate_only and not args.shadow:
        try:
            _promote_batch(stage_dir, symbols)
            report["promoted"] = True
        except Exception as exc:
            failures.append(f"promotion: {type(exc).__name__}: {exc}")
            report["status"] = "failed"
            report["promoted"] = False
            traceback.print_exc()
    else:
        report["promoted"] = False

    report_path = _write_report(report)
    print(f"[MONTHLY] Report: {report_path}")

    if not args.no_email:
        _send_email(not failures, report, report_path)

    if failures:
        print("[MONTHLY] FAILED - active models were not intentionally replaced.")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    if stage_dir.exists():
        shutil.rmtree(stage_dir)

    if args.validate_only:
        action = "validated"
    elif args.shadow:
        action = "trained and evaluated in shadow mode; none promoted"
    else:
        action = "trained, validated, and promoted"
    print(f"[MONTHLY] SUCCESS - {expected_models} models {action}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
