#!/usr/bin/env python3
"""Monitor intraday model modes from durable, non-overlapping outcomes."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config import SYMBOL
from model_outcome_store import load_outcome_history, outcome_history_path


MIN_OUTCOMES = 30
MIN_SESSIONS = 5
MIN_ACCURACY = 0.48
MAX_BRIER = 0.30
MIN_PROFIT_FACTOR = 1.0
MAX_DRAWDOWN = -0.15
MAX_SESSION_CONCENTRATION = 0.50


def _horizon_minutes(row: pd.Series) -> int:
    value = str(row.get("outcome_horizon") or row.get("target_horizon") or "15min")
    digits = "".join(character for character in value if character.isdigit())
    return max(1, int(digits or 15))


def non_overlapping_intraday_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep chronologically distinct prediction horizons per symbol and mode."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=list(frame.columns) if frame is not None else [])

    work = frame.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work = work.dropna(subset=["timestamp"])
    work = work.loc[work["mode"].astype(str).str.startswith("intraday")]
    work["session_date_ny"] = (
        work["timestamp"].dt.tz_convert("America/New_York").dt.date
    )
    work["horizon_minutes"] = work.apply(_horizon_minutes, axis=1)
    work = work.sort_values(["symbol", "mode", "session_date_ny", "timestamp"])

    selected_indices = []
    group_columns = ["symbol", "mode", "session_date_ny"]
    for _, group in work.groupby(group_columns, sort=False):
        next_available = None
        for index, row in group.iterrows():
            timestamp = row["timestamp"]
            if next_available is not None and timestamp < next_available:
                continue
            selected_indices.append(index)
            next_available = timestamp + pd.Timedelta(
                minutes=int(row["horizon_minutes"])
            )

    return work.loc[selected_indices].sort_values("timestamp").reset_index(drop=True)


def evaluate_intraday_mode(frame: pd.DataFrame, cost_bps: float = 10.0) -> dict:
    if frame is None or frame.empty:
        return {
            "outcomes": 0,
            "sessions": 0,
            "entry_signals": 0,
            "eligible": False,
            "shadow_retrain_recommended": False,
            "reasons": ["no resolved non-overlapping outcomes"],
        }

    work = frame.copy()
    work["predicted_prob"] = pd.to_numeric(work["predicted_prob"], errors="coerce")
    work["actual_outcome"] = pd.to_numeric(work["actual_outcome"], errors="coerce")
    work["return_pct"] = pd.to_numeric(work["return_pct"], errors="coerce")
    work = work.dropna(subset=["predicted_prob", "actual_outcome", "return_pct"])
    if work.empty:
        return evaluate_intraday_mode(pd.DataFrame(), cost_bps=cost_bps)

    probabilities = work["predicted_prob"].clip(0.0, 1.0).to_numpy(float)
    outcomes = work["actual_outcome"].astype(int).to_numpy()
    predicted = (probabilities >= 0.5).astype(int)
    raw_returns = work["return_pct"].to_numpy(float) / 100.0
    entry_mask = predicted == 1
    # Production is long-only. Cost-aware return therefore evaluates bullish
    # entry signals; bearish classifications remain represented in accuracy
    # and Brier score rather than being treated as synthetic short trades.
    trade_returns = raw_returns[entry_mask] - (float(cost_bps) / 10000.0)

    gains = float(trade_returns[trade_returns > 0].sum())
    losses = float(-trade_returns[trade_returns < 0].sum())
    # Keep reports strict-JSON compatible while still representing a run with
    # gains and no losses as an excellent finite profit factor.
    profit_factor = gains / losses if losses > 0 else (10.0 if gains > 0 else 0.0)
    equity_curve = pd.Series(1.0 + trade_returns).cumprod()
    drawdown = equity_curve / equity_curve.cummax() - 1.0

    if "session_date_ny" not in work.columns:
        work["session_date_ny"] = (
            pd.to_datetime(work["timestamp"], utc=True)
            .dt.tz_convert("America/New_York")
            .dt.date
        )
    trades = work.loc[entry_mask].copy()
    trades["net_return"] = trade_returns
    session_returns = trades.groupby("session_date_ny")["net_return"].sum()
    absolute_total = float(session_returns.abs().sum())
    concentration = (
        float(session_returns.abs().max() / absolute_total)
        if absolute_total > 0
        else 1.0
    )

    count = int(len(work))
    sessions = int(work["session_date_ny"].nunique())
    accuracy = float(np.mean(predicted == outcomes))
    brier = float(np.mean((probabilities - outcomes) ** 2))
    net_return = float(equity_curve.iloc[-1] - 1.0) if len(equity_curve) else 0.0
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0
    eligible = count >= MIN_OUTCOMES and sessions >= MIN_SESSIONS

    reasons = []
    if count < MIN_OUTCOMES:
        reasons.append(f"non-overlapping outcomes {count} < {MIN_OUTCOMES}")
    if sessions < MIN_SESSIONS:
        reasons.append(f"sessions {sessions} < {MIN_SESSIONS}")
    if eligible:
        if accuracy < MIN_ACCURACY:
            reasons.append(f"accuracy {accuracy:.3f} < {MIN_ACCURACY:.3f}")
        if brier > MAX_BRIER:
            reasons.append(f"brier {brier:.3f} > {MAX_BRIER:.3f}")
        if net_return <= 0:
            reasons.append("cost-adjusted net return is not positive")
        if profit_factor < MIN_PROFIT_FACTOR:
            reasons.append(f"profit factor {profit_factor:.3f} < {MIN_PROFIT_FACTOR:.3f}")
        if max_drawdown < MAX_DRAWDOWN:
            reasons.append(f"drawdown {max_drawdown:.2%} < {MAX_DRAWDOWN:.2%}")
        if concentration > MAX_SESSION_CONCENTRATION:
            reasons.append(
                f"session concentration {concentration:.2%} > "
                f"{MAX_SESSION_CONCENTRATION:.2%}"
            )

    return {
        "outcomes": count,
        "sessions": sessions,
        "entry_signals": int(entry_mask.sum()),
        "eligible": eligible,
        "accuracy": accuracy,
        "brier_score": brier,
        "cost_adjusted_net_return": net_return,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "session_concentration": concentration,
        "shadow_retrain_recommended": bool(eligible and reasons),
        "reasons": reasons,
    }


def build_report(symbols, lookback_days: int = 60, cost_bps: float = 10.0) -> dict:
    history = load_outcome_history()
    if not history.empty:
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=lookback_days)
        history = history.loc[history["timestamp"] >= cutoff]
    history = non_overlapping_intraday_outcomes(history)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "history_path": str(outcome_history_path()),
        "lookback_days": int(lookback_days),
        "cost_bps": float(cost_bps),
        "results": {},
    }
    for symbol in symbols:
        symbol = str(symbol).upper()
        report["results"][symbol] = {}
        for mode in ("intraday_mr", "intraday_mom"):
            selected = history.loc[
                (history["symbol"] == symbol) & (history["mode"] == mode)
            ]
            report["results"][symbol][mode] = evaluate_intraday_mode(
                selected,
                cost_bps=cost_bps,
            )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--lookback-days", type=int, default=60)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--output-dir", default="logs/intraday_monitor")
    args = parser.parse_args()

    symbols = args.symbols or list(dict.fromkeys([*SYMBOL, "SPY"]))
    report = build_report(symbols, args.lookback_days, args.cost_bps)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    report_path = output_dir / f"intraday_monitor_{run_id}.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False))

    for symbol, modes in report["results"].items():
        for mode, metrics in modes.items():
            status = (
                "SHADOW RETRAIN RECOMMENDED"
                if metrics["shadow_retrain_recommended"]
                else ("HEALTHY" if metrics["eligible"] else "COLLECTING")
            )
            print(
                f"[INTRADAY MONITOR] {symbol}/{mode}: {status} "
                f"outcomes={metrics['outcomes']} sessions={metrics['sessions']} "
                f"reasons={'; '.join(metrics['reasons']) or 'none'}"
            )
    print(f"[INTRADAY MONITOR] Report: {report_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
