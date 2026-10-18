#!/usr/bin/env python3
"""Safely activate the September binary model artifacts.

The command is a dry run unless --apply is supplied. All requested artifacts
are loaded and validated before any active model is changed. Applying creates
a timestamped backup and writes an integrity manifest for rollback/audit.
"""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib


DEFAULT_SYMBOLS = ("NVDA", "AAPL")
MODES = ("daily", "intraday_mr", "intraday_mom")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_artifact(path: Path, symbol: str, mode: str) -> dict:
    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        raise ValueError(f"{path}: artifact is not a dictionary")
    if str(artifact.get("symbol", "")).upper() != symbol:
        raise ValueError(f"{path}: symbol is {artifact.get('symbol')!r}, expected {symbol}")
    if artifact.get("mode") != mode:
        raise ValueError(f"{path}: mode is {artifact.get('mode')!r}, expected {mode}")
    if artifact.get("target_type", "binary") != "binary":
        raise ValueError(f"{path}: target_type must be binary")
    if artifact.get("model") is None:
        raise ValueError(f"{path}: model is missing")
    if not artifact.get("features"):
        raise ValueError(f"{path}: feature list is empty")
    threshold = artifact.get("decision_threshold")
    try:
        threshold = float(threshold)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: invalid decision threshold {threshold!r}") from error
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"{path}: decision threshold is outside (0, 1)")
    return artifact


def activate(source_dir: Path, model_dir: Path, symbols, apply: bool) -> dict:
    requested = []
    for symbol in symbols:
        symbol = symbol.upper()
        for mode in MODES:
            name = f"{symbol}_{mode}_xgb.pkl"
            source = source_dir / name
            target = model_dir / name
            if not source.is_file():
                raise FileNotFoundError(f"missing rollback artifact: {source}")
            if not target.is_file():
                raise FileNotFoundError(f"missing active artifact: {target}")
            artifact = validate_artifact(source, symbol, mode)
            requested.append(
                {
                    "symbol": symbol,
                    "mode": mode,
                    "source": str(source.resolve()),
                    "target": str(target.resolve()),
                    "source_sha256": sha256(source),
                    "active_sha256_before": sha256(target),
                    "decision_threshold": float(artifact["decision_threshold"]),
                }
            )

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "applied": False,
        "artifacts": requested,
    }
    if not apply:
        return result

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = model_dir / "production_rollback_backups" / timestamp
    backup_dir.mkdir(parents=True, exist_ok=False)

    # Stage every replacement on the target filesystem before changing any
    # active file. os.replace then provides an atomic update per artifact.
    stage_dir = Path(tempfile.mkdtemp(prefix=".binary_activation_", dir=model_dir))
    applied_targets = []
    try:
        for item in requested:
            source = Path(item["source"])
            target = Path(item["target"])
            backup = backup_dir / target.name
            staged = stage_dir / target.name
            shutil.copy2(target, backup)
            shutil.copy2(source, staged)
            validate_artifact(staged, item["symbol"], item["mode"])
            item["backup"] = str(backup.resolve())

        for item in requested:
            target = Path(item["target"])
            staged = stage_dir / target.name
            os.replace(staged, target)
            applied_targets.append(item)
            item["active_sha256_after"] = sha256(target)

        result["applied"] = True
        result["backup_dir"] = str(backup_dir.resolve())
        manifest = backup_dir / "activation_manifest.json"
        manifest.write_text(json.dumps(result, indent=2) + "\n")
        result["manifest"] = str(manifest.resolve())
        return result
    except Exception:
        # Restore any targets already swapped if an unexpected filesystem
        # failure interrupts the activation sequence.
        for item in reversed(applied_targets):
            backup = Path(item["backup"])
            target = Path(item["target"])
            if backup.is_file():
                shutil.copy2(backup, target)
        raise
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("models/monthly_backups/2026-09"),
    )
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = activate(args.source_dir, args.model_dir, args.symbols, args.apply)
    print(json.dumps(result, indent=2))
    if not args.apply:
        print("[DRY RUN] No active artifacts changed. Re-run with --apply to activate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
