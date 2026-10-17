#!/usr/bin/env python3
"""Lock at most one stable development design per symbol."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def select_designs(report: dict) -> dict:
    selections = {}
    for symbol, symbol_result in report.get("results", {}).items():
        candidates = []
        for design_name, design in symbol_result.get("designs", {}).items():
            for model_kind, metrics in design.get("models", {}).items():
                if metrics.get("stable_candidate"):
                    candidates.append(
                        (
                            metrics["net_return"],
                            metrics["profit_factor"],
                            design_name,
                            model_kind,
                            design,
                        )
                    )
        if not candidates:
            continue
        _, _, design_name, model_kind, design = max(candidates)
        development_metrics = {
            key: value
            for key, value in design["models"][model_kind].items()
            if key != "records"
        }
        selections[symbol] = {
            "model_kind": model_kind,
            "target_design": design_name,
            "horizon_bars": design["horizon_bars"],
            "horizon_minutes": design["horizon_minutes"],
            "volatility_multiplier": design["volatility_multiplier"],
            "sector_benchmark": symbol_result.get("sector_benchmark"),
            "sector_feature_set": symbol_result.get("sector_feature_set"),
            "development_metrics": development_metrics,
        }
    return selections


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    parser.add_argument("--output", default="research_locks/unified_intraday_lock.json")
    args = parser.parse_args()
    report_path = Path(args.report).resolve()
    report = json.loads(report_path.read_text())
    if report.get("phase") != "development":
        raise ValueError("only a development report can be locked")
    lock = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_report": str(report_path),
        "dataset_id": report["dataset_id"],
        "manifest": report["manifest"],
        "cost_bps": report["cost_bps"],
        "holdout_sessions": report["holdout_sessions"],
        "selections": select_designs(report),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(lock, indent=2) + "\n")
    print(f"[LOCK] selections={len(lock['selections'])} path={output}")
    for symbol, selection in lock["selections"].items():
        print(f"[LOCK] {symbol}: {selection['target_design']}/{selection['model_kind']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
