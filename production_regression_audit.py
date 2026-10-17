#!/usr/bin/env python3
"""Audit actual completed trades before building a strategy replay."""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from trade_attribution import attribution_summary, build_trade_cycles, normalize_trade_log


PROVENANCE_COLUMNS = (
    "source_commit",
    "source_checkout",
    "execution_id",
    "client_order_id",
    "broker_order_id",
)


def parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"source must be LABEL=PATH, received {value!r}")
    label, raw_path = value.split("=", 1)
    if not label.strip() or not raw_path.strip():
        raise ValueError(f"source must be LABEL=PATH, received {value!r}")
    return label.strip(), Path(raw_path).expanduser().resolve()


def _text(value, default="unknown") -> str:
    if value is None or pd.isna(value):
        return default
    value = str(value).strip()
    return value if value and value.lower() != "nan" else default


def discover_trade_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.glob("trades_*.csv"))


def load_source(label: str, root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cycle_frames = []
    execution_frames = []
    for path in discover_trade_files(root):
        raw = pd.read_csv(path)
        if raw.empty:
            continue
        symbol = path.stem.removeprefix("trades_").upper()
        normalized = normalize_trade_log(raw)
        if normalized.empty:
            continue
        normalized["audit_source"] = label
        normalized["audit_file"] = str(path)
        if "symbol" not in normalized.columns:
            normalized["symbol"] = symbol
        else:
            normalized["symbol"] = normalized["symbol"].fillna(symbol)
        execution_frames.append(normalized)

        cycles = build_trade_cycles(raw, symbol=symbol)
        if cycles.empty:
            continue
        normalized_by_time = normalized.drop_duplicates("timestamp").set_index(
            "timestamp", drop=False
        )
        for column in PROVENANCE_COLUMNS:
            if column not in normalized_by_time.columns:
                normalized_by_time[column] = np.nan
            cycles[f"entry_{column}"] = cycles["entry_time"].map(
                lambda timestamp: _text(
                    normalized_by_time.loc[timestamp, column]
                    if timestamp in normalized_by_time.index else None
                )
            )
            cycles[f"exit_{column}"] = cycles["exit_time"].map(
                lambda timestamp: _text(
                    normalized_by_time.loc[timestamp, column]
                    if timestamp in normalized_by_time.index else None
                )
            )
        cycles["audit_source"] = label
        cycles["audit_file"] = str(path)
        cycles["pyramid_additions"] = (cycles["buy_count"] - 1).clip(lower=0)
        cycles["entry_probability_edge"] = (
            cycles["decision_probability"] - cycles["buy_threshold"]
        )
        cycle_frames.append(cycles)
    cycles = pd.concat(cycle_frames, ignore_index=True) if cycle_frames else pd.DataFrame()
    executions = (
        pd.concat(execution_frames, ignore_index=True)
        if execution_frames else pd.DataFrame()
    )
    return cycles, executions


def assign_periods(cycles: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    frames = []
    windows = (
        ("post_cutoff", cutoff, None),
        ("pre_cutoff_90d", cutoff - timedelta(days=90), cutoff),
        ("last_12m", cutoff - timedelta(days=365), None),
    )
    for name, start, end in windows:
        mask = cycles["entry_time"] >= start
        if end is not None:
            mask &= cycles["entry_time"] < end
        selected = cycles.loc[mask].copy()
        selected["period"] = name
        frames.append(selected)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_periods(period_cycles: pd.DataFrame) -> pd.DataFrame:
    if period_cycles.empty:
        return pd.DataFrame()
    return attribution_summary(
        period_cycles,
        group_columns=["period", "audit_source", "symbol"],
    )


def summarize_provenance(cycles: pd.DataFrame) -> pd.DataFrame:
    if cycles.empty:
        return pd.DataFrame()
    return attribution_summary(
        cycles,
        group_columns=[
            "audit_source",
            "entry_source_commit",
            "symbol",
            "model_used",
            "regime",
            "entry_reason",
            "exit_reason",
        ],
    )


def find_near_duplicate_orders(executions: pd.DataFrame, seconds: int = 3) -> pd.DataFrame:
    if executions.empty:
        return pd.DataFrame()
    rows = []
    ordered = executions.sort_values(
        ["audit_source", "symbol", "action", "timestamp"]
    )
    for _, group in ordered.groupby(["audit_source", "symbol", "action"]):
        previous = None
        for _, row in group.iterrows():
            if previous is not None:
                elapsed = (row["timestamp"] - previous["timestamp"]).total_seconds()
                same_quantity = np.isclose(row["exec_qty"], previous["exec_qty"])
                if 0 <= elapsed <= seconds and same_quantity:
                    rows.append(
                        {
                            "audit_source": row["audit_source"],
                            "symbol": row["symbol"],
                            "action": row["action"],
                            "qty": float(row["exec_qty"]),
                            "first_timestamp": previous["timestamp"],
                            "second_timestamp": row["timestamp"],
                            "seconds_apart": elapsed,
                            "first_broker_order_id": _text(
                                previous.get("broker_order_id")
                            ),
                            "second_broker_order_id": _text(
                                row.get("broker_order_id")
                            ),
                            "first_source_commit": _text(
                                previous.get("source_commit")
                            ),
                            "second_source_commit": _text(row.get("source_commit")),
                        }
                    )
            previous = row
    return pd.DataFrame(rows)


def diagnostics(cycles: pd.DataFrame, duplicates: pd.DataFrame) -> dict:
    if cycles.empty:
        return {"cycles": 0, "near_duplicate_orders": len(duplicates)}
    return {
        "cycles": int(len(cycles)),
        "start": str(cycles["entry_time"].min()),
        "end": str(cycles["exit_time"].max()),
        "unknown_model_pct": float(cycles["model_used"].eq("unknown").mean() * 100),
        "unknown_regime_pct": float(cycles["regime"].eq("unknown").mean() * 100),
        "unknown_entry_commit_pct": float(
            cycles["entry_source_commit"].eq("unknown").mean() * 100
        ),
        "pyramided_cycles": int((cycles["pyramid_additions"] > 0).sum()),
        "near_duplicate_orders": int(len(duplicates)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Trade directory as LABEL=PATH; may be repeated.",
    )
    parser.add_argument("--cutoff", default="2026-09-24")
    parser.add_argument("--output-dir", default="logs/production_regression_audit")
    args = parser.parse_args()

    all_cycles = []
    all_executions = []
    for source_value in args.source:
        label, root = parse_source(source_value)
        cycles, executions = load_source(label, root)
        if not cycles.empty:
            all_cycles.append(cycles)
        if not executions.empty:
            all_executions.append(executions)
        print(
            f"[AUDIT] {label}: files={len(discover_trade_files(root))} "
            f"cycles={len(cycles)} executions={len(executions)}"
        )
    cycles = pd.concat(all_cycles, ignore_index=True) if all_cycles else pd.DataFrame()
    executions = (
        pd.concat(all_executions, ignore_index=True)
        if all_executions else pd.DataFrame()
    )
    if cycles.empty:
        raise ValueError("no completed flat-to-flat cycles found")

    cutoff = pd.Timestamp(args.cutoff, tz="UTC")
    period_cycles = assign_periods(cycles, cutoff)
    period_summary = summarize_periods(period_cycles)
    provenance_summary = summarize_provenance(cycles)
    duplicates = find_near_duplicate_orders(executions)
    audit_diagnostics = diagnostics(cycles, duplicates)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cycles.to_csv(output_dir / "closed_cycles.csv", index=False)
    period_summary.to_csv(output_dir / "period_symbol_summary.csv", index=False)
    provenance_summary.to_csv(output_dir / "provenance_summary.csv", index=False)
    duplicates.to_csv(output_dir / "near_duplicate_orders.csv", index=False)
    (output_dir / "diagnostics.json").write_text(
        json.dumps(audit_diagnostics, indent=2) + "\n"
    )

    print("\n[AUDIT] PERIOD/SYMBOL SUMMARY")
    print(period_summary.to_string(index=False))
    print("\n[AUDIT] DIAGNOSTICS")
    print(json.dumps(audit_diagnostics, indent=2))
    print(f"[AUDIT] Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
