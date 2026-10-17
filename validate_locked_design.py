#!/usr/bin/env python3
"""Evaluate locked unified designs once on the reserved historical period."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV

import config
from predictive_model.model_evaluation import optimize_cost_aware_threshold
from unified_intraday_research import (
    aggregate_non_overlapping_trades,
    load_frame,
    model_factory,
    prepare_dataset,
    split_development_holdout,
    verify_dataset_manifest,
)


def evaluate_selection(symbol: str, selection: dict, lock: dict, data_dir: Path) -> dict:
    symbol_frame = load_frame(data_dir, symbol)
    _, raw_holdout, holdout_start = split_development_holdout(
        symbol_frame, int(lock["holdout_sessions"])
    )
    spy = load_frame(data_dir, config.SPY_SYMBOL)
    sector_symbol = selection.get("sector_benchmark")
    sector = load_frame(data_dir, sector_symbol) if sector_symbol else None
    X, y, forward_returns, _ = prepare_dataset(
        symbol_frame,
        spy,
        sector,
        selection["sector_feature_set"],
        int(selection["horizon_bars"]),
        float(lock["cost_bps"]),
        float(selection["volatility_multiplier"]),
    )
    holdout_start_timestamp = raw_holdout.index.min()
    horizon = int(selection["horizon_bars"])
    development_mask = X.index < holdout_start_timestamp
    holdout_mask = X.index >= holdout_start_timestamp
    X_development, y_development = X.loc[development_mask], y.loc[development_mask]
    # Purge observations whose forward target would cross the holdout boundary.
    X_development = X_development.iloc[:-horizon]
    y_development = y_development.loc[X_development.index]
    X_holdout = X.loc[holdout_mask]
    holdout_returns = forward_returns.loc[X_holdout.index]
    calibration_rows = max(100, int(len(X_development) * 0.15))
    fit_end = len(X_development) - calibration_rows - horizon
    X_fit, y_fit = X_development.iloc[:fit_end], y_development.iloc[:fit_end]
    X_cal = X_development.iloc[fit_end + horizon :]
    y_cal = y_development.loc[X_cal.index]
    if y_fit.nunique() < 2 or y_cal.nunique() < 2:
        raise ValueError(f"{symbol} development fit/calibration lacks two classes")
    estimator = model_factory(selection["model_kind"])
    estimator.fit(X_fit, y_fit)
    calibrated = CalibratedClassifierCV(estimator, method="sigmoid", cv="prefit", n_jobs=1)
    calibrated.fit(X_cal, y_cal)
    calibration_probability = calibrated.predict_proba(X_cal)[:, 1]
    threshold_result = optimize_cost_aware_threshold(
        y_cal.to_numpy(),
        calibration_probability,
        forward_returns.loc[X_cal.index].to_numpy(),
        cost_bps=float(lock["cost_bps"]),
        minimum_trades=max(10, len(X_cal) // 100),
        periods_per_year=max(1, (252 * 26) // horizon),
    )
    if not threshold_result.get("eligible"):
        raise ValueError(f"{symbol} locked threshold calibration is ineligible")
    threshold = float(threshold_result["best_threshold"])
    X_scored = X_holdout.iloc[::horizon]
    probability = calibrated.predict_proba(X_scored)[:, 1]
    trade_mask = probability >= threshold
    metrics = aggregate_non_overlapping_trades(
        holdout_returns.loc[X_scored.index].to_numpy(), trade_mask, float(lock["cost_bps"])
    )
    return {
        "historical_validation_start": holdout_start,
        "historical_validation_end": str(X_scored.index.max()),
        "observations": len(X_scored),
        "decision_threshold": threshold,
        "threshold_metric": "cost_aware_development_calibration",
        "metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lock")
    parser.add_argument("--data-dir", default="research_data/intraday_15m")
    parser.add_argument("--output-dir", default="logs/unified_intraday_validation")
    args = parser.parse_args()
    lock_path = Path(args.lock).resolve()
    lock_hash = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    lock = json.loads(lock_path.read_text())
    data_dir = Path(args.data_dir).resolve()
    actual_dataset_id = verify_dataset_manifest(data_dir, Path(lock["manifest"]))
    if actual_dataset_id != lock["dataset_id"]:
        raise ValueError("lock and frozen dataset IDs do not match")
    if not lock.get("selections"):
        raise ValueError("lock contains no stable development selections")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for prior_path in output_dir.glob("historical_validation_*.json"):
        prior = json.loads(prior_path.read_text())
        if prior.get("lock_sha256") == lock_hash:
            raise RuntimeError(
                f"this lock was already evaluated in {prior_path}; historical validation is one-shot"
            )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "historical_validation",
        "lock": str(lock_path),
        "lock_sha256": lock_hash,
        "dataset_id": actual_dataset_id,
        "results": {},
    }
    for symbol, selection in lock["selections"].items():
        result = evaluate_selection(symbol, selection, lock, data_dir)
        report["results"][symbol] = result
        metrics = result["metrics"]
        print(
            f"[VALIDATION] {symbol}: trades={metrics['trade_count']} "
            f"net={metrics['net_return']:.2%} pf={metrics['profit_factor']:.3f} "
            f"dd={metrics['max_drawdown']:.2%} threshold={result['decision_threshold']:.3f}"
        )
    output = output_dir / f"historical_validation_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[VALIDATION] Report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
