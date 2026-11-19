#!/usr/bin/env python3
"""Periodically retrain intraday challengers as active artifacts age."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib

from config import (
    INTRADAY_MODEL_RETRAIN_AGE_DAYS,
    INTRADAY_MODEL_RETRAIN_RETRY_DAYS,
    MODEL_DIR,
    TRAIN_SYMBOLS,
)
from train_model.retrain_model import train_intraday_models


INTRADAY_MODES = ("intraday_mr", "intraday_mom")
MODEL_ROOT = Path(MODEL_DIR)
ATTEMPT_STATE_PATH = MODEL_ROOT / "intraday_retrain_attempts.json"


def _parse_timestamp(value):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _artifact_age_days(symbol: str, mode: str, now: datetime):
    path = MODEL_ROOT / f"{symbol}_{mode}_xgb.pkl"
    if not path.exists():
        return None, "artifact_missing"
    try:
        artifact = joblib.load(path)
    except Exception as exc:
        return None, f"artifact_unreadable:{exc}"

    trained_at = _parse_timestamp((artifact or {}).get("trained_at"))
    if trained_at is None:
        return None, "trained_at_missing_or_invalid"
    return (now - trained_at).total_seconds() / 86400.0, None


def _load_attempt_state():
    try:
        with ATTEMPT_STATE_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as exc:
        print(f"[WARN] Could not read retrain attempt state: {exc}")
        return {}


def _save_attempt_state(state):
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    temporary_path = ATTEMPT_STATE_PATH.with_suffix(".json.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_path, ATTEMPT_STATE_PATH)


def _retry_due(last_attempt, now: datetime) -> bool:
    attempted_at = _parse_timestamp(last_attempt)
    if attempted_at is None:
        return True
    retry_after = timedelta(days=int(INTRADAY_MODEL_RETRAIN_RETRY_DAYS))
    return now - attempted_at >= retry_after


def main() -> int:
    now = datetime.now(timezone.utc)
    attempts = _load_attempt_state()
    due_by_symbol = {}

    print(
        f"[STALE INTRADAY RETRAIN] checked_at={now.isoformat()} "
        f"trigger_age_days={INTRADAY_MODEL_RETRAIN_AGE_DAYS} "
        f"retry_days={INTRADAY_MODEL_RETRAIN_RETRY_DAYS}"
    )

    for symbol in TRAIN_SYMBOLS:
        due_modes = []
        for mode in INTRADAY_MODES:
            key = f"{symbol.upper()}/{mode}"
            age_days, issue = _artifact_age_days(symbol.upper(), mode, now)
            if issue:
                print(f"[STALE INTRADAY RETRAIN] {key}: due ({issue})")
                due = True
            else:
                due = age_days >= float(INTRADAY_MODEL_RETRAIN_AGE_DAYS)
                print(
                    f"[STALE INTRADAY RETRAIN] {key}: age={age_days:.1f}d "
                    f"due={due}"
                )

            if not due:
                continue
            if not _retry_due(attempts.get(key), now):
                print(
                    f"[STALE INTRADAY RETRAIN] {key}: retry cooldown active; "
                    f"last_attempt={attempts.get(key)}"
                )
                continue
            due_modes.append(mode)
            attempts[key] = now.isoformat()

        if due_modes:
            due_by_symbol[symbol.upper()] = due_modes

    if not due_by_symbol:
        print("[STALE INTRADAY RETRAIN] No models are due.")
        return 0

    _save_attempt_state(attempts)
    failures = []
    for symbol, modes in due_by_symbol.items():
        print(
            f"[STALE INTRADAY RETRAIN] Training {symbol}: "
            f"modes={','.join(modes)}"
        )
        try:
            train_intraday_models(symbol, modes=modes)
        except Exception as exc:
            failures.append(f"{symbol}: {exc}")
            print(f"[ERROR] Stale intraday retrain failed for {symbol}: {exc}")

    if failures:
        print("[STALE INTRADAY RETRAIN] Failures: " + "; ".join(failures))
        return 1

    print(
        "[STALE INTRADAY RETRAIN] Attempts complete. Promotion gates remain "
        "authoritative; rejected challengers do not replace active models."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
