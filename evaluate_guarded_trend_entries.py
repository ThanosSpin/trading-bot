#!/usr/bin/env python3
"""Evaluate below-threshold trend entries from durable signal snapshots."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "timestamp",
    "price",
    "finalprob",
    "buythreshold",
    "sessionreturn",
    "intradaymom",
    "vwapdistance",
    "vwapslope2",
    "relativereturn4",
    "relativesessionreturn",
    "relativemarketavailable",
    "volumetimeratio",
}


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def load_signal_history(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"missing confirmation columns: {', '.join(missing)}")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    numeric = REQUIRED_COLUMNS - {"timestamp", "relativemarketavailable"}
    for name in numeric:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame["relativemarketavailable"] = _as_bool(frame["relativemarketavailable"])
    frame = frame.dropna(subset=["timestamp", "price"]).sort_values("timestamp")
    frame = frame.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    return frame


def guarded_candidates(
    frame: pd.DataFrame,
    max_probability_gap: float,
    minimum_volume_ratio: float,
    relative_rule: str = "strict",
) -> pd.DataFrame:
    if relative_rule not in {"strict", "recent"}:
        raise ValueError(f"unsupported relative rule: {relative_rule}")

    complete = frame.dropna(subset=list(REQUIRED_COLUMNS - {"timestamp", "relativemarketavailable"}))
    relative_mask = complete["relativereturn4"] > 0.0
    if relative_rule == "strict":
        relative_mask &= complete["relativesessionreturn"] > 0.0

    mask = (
        (complete["finalprob"] < complete["buythreshold"])
        & (complete["finalprob"] >= complete["buythreshold"] - max_probability_gap)
        & (complete["sessionreturn"] > 0.0)
        & (complete["intradaymom"] > 0.0)
        & (complete["vwapdistance"] > 0.0)
        & (complete["vwapslope2"] > 0.0)
        & relative_mask
        & complete["relativemarketavailable"]
        & (complete["volumetimeratio"] >= minimum_volume_ratio)
    )
    return complete.loc[mask].copy()


def guard_funnel(
    frame: pd.DataFrame,
    max_probability_gap: float,
    minimum_volume_ratio: float,
    relative_rule: str = "strict",
) -> dict:
    """Return individual and cumulative pass counts for every entry guard."""
    if relative_rule not in {"strict", "recent"}:
        raise ValueError(f"unsupported relative rule: {relative_rule}")

    complete = frame.dropna(
        subset=list(REQUIRED_COLUMNS - {"timestamp", "relativemarketavailable"})
    )
    conditions = [
        (
            "probability_gap",
            (complete["finalprob"] < complete["buythreshold"])
            & (
                complete["finalprob"]
                >= complete["buythreshold"] - max_probability_gap
            ),
        ),
        ("positive_session_return", complete["sessionreturn"] > 0.0),
        ("positive_intraday_momentum", complete["intradaymom"] > 0.0),
        ("above_vwap", complete["vwapdistance"] > 0.0),
        ("positive_vwap_slope", complete["vwapslope2"] > 0.0),
        (
            "minimum_volume_ratio",
            complete["volumetimeratio"] >= minimum_volume_ratio,
        ),
        ("relative_market_available", complete["relativemarketavailable"]),
        ("positive_recent_relative_return", complete["relativereturn4"] > 0.0),
    ]
    if relative_rule == "strict":
        conditions.append(
            (
                "positive_session_relative_return",
                complete["relativesessionreturn"] > 0.0,
            )
        )

    cumulative_mask = pd.Series(True, index=complete.index, dtype=bool)
    individual = {}
    cumulative = {}
    rejected_at = {}
    for name, condition in conditions:
        condition = condition.fillna(False).astype(bool)
        individual[name] = int(condition.sum())
        rejected_at[name] = int((cumulative_mask & ~condition).sum())
        cumulative_mask &= condition
        cumulative[name] = int(cumulative_mask.sum())

    return {
        "complete_rows": int(len(complete)),
        "individual_pass_counts": individual,
        "cumulative_pass_counts": cumulative,
        "rejected_at_guard": rejected_at,
        "candidate_rows": int(cumulative_mask.sum()),
    }


def horizon_eligible_count(candidates: pd.DataFrame, horizon_minutes: int) -> int:
    """Count candidates early enough to complete a same-session horizon."""
    if candidates.empty:
        return 0
    timestamps_ny = candidates["timestamp"].dt.tz_convert("America/New_York")
    close_ny = timestamps_ny.dt.normalize() + pd.Timedelta(hours=16)
    latest_entry_ny = close_ny - pd.Timedelta(minutes=horizon_minutes)
    return int((timestamps_ny <= latest_entry_ny).sum())


def evaluate_horizon(
    history: pd.DataFrame,
    candidates: pd.DataFrame,
    horizon_minutes: int,
    cost_bps: float,
    tolerance_minutes: int,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.assign(
            horizon_minutes=pd.Series(dtype=int),
            future_timestamp=pd.Series(dtype="datetime64[ns, UTC]"),
            future_price=pd.Series(dtype=float),
            gross_return=pd.Series(dtype=float),
            net_return=pd.Series(dtype=float),
        )

    times = pd.DatetimeIndex(history["timestamp"])
    selected = []
    next_allowed = None
    for row in candidates.itertuples(index=False):
        timestamp_ny = row.timestamp.tz_convert("America/New_York")
        close_ny = timestamp_ny.normalize() + pd.Timedelta(hours=16)
        latest_entry_ny = close_ny - pd.Timedelta(minutes=horizon_minutes)
        if timestamp_ny > latest_entry_ny:
            continue
        if next_allowed is not None and row.timestamp < next_allowed:
            continue
        target = row.timestamp + pd.Timedelta(minutes=horizon_minutes)
        index = int(times.searchsorted(target, side="left"))
        if index >= len(history):
            continue
        future = history.iloc[index]
        if future["timestamp"] > target + pd.Timedelta(minutes=tolerance_minutes):
            continue
        if future["timestamp"].tz_convert("America/New_York").date() != row.timestamp.tz_convert("America/New_York").date():
            continue

        record = row._asdict()
        record.update(
            horizon_minutes=horizon_minutes,
            future_timestamp=future["timestamp"],
            future_price=float(future["price"]),
            gross_return=float(future["price"] / row.price - 1.0),
        )
        record["net_return"] = record["gross_return"] - float(cost_bps) / 10000.0
        selected.append(record)
        next_allowed = row.timestamp + pd.Timedelta(minutes=horizon_minutes)

    return pd.DataFrame(selected)


def summarize(evaluated: pd.DataFrame, minimum_samples: int, minimum_sessions: int) -> dict:
    if evaluated.empty:
        return {
            "status": "COLLECTING",
            "observations": 0,
            "sessions": 0,
            "win_rate": None,
            "average_net_return": None,
            "compound_net_return": None,
            "profit_factor": None,
            "max_drawdown": None,
        }

    returns = evaluated["net_return"].astype(float)
    sessions = evaluated["timestamp"].dt.tz_convert("America/New_York").dt.date.nunique()
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    profit_factor = None if losses == 0 and gains > 0 else (gains / losses if losses else 0.0)
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    enough = len(evaluated) >= minimum_samples and sessions >= minimum_sessions
    profitable = returns.mean() > 0.0 and (profit_factor is None or profit_factor > 1.0)
    return {
        "status": "PASS" if enough and profitable else ("FAIL" if enough else "COLLECTING"),
        "observations": int(len(evaluated)),
        "sessions": int(sessions),
        "win_rate": float((returns > 0).mean()),
        "average_net_return": float(returns.mean()),
        "compound_net_return": float(equity.iloc[-1] - 1.0),
        "profit_factor": None if profit_factor is None else float(profit_factor),
        "max_drawdown": float(drawdown.min()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--symbols", nargs="+", default=["NVDA", "AAPL", "ABBV", "PLTR"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[30, 60])
    parser.add_argument("--max-probability-gap", type=float, default=0.05)
    parser.add_argument("--minimum-volume-ratio", type=float, default=0.80)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--tolerance-minutes", type=int, default=15)
    parser.add_argument("--minimum-samples", type=int, default=20)
    parser.add_argument("--minimum-sessions", type=int, default=5)
    parser.add_argument(
        "--relative-rules",
        nargs="+",
        choices=["strict", "recent"],
        default=["strict", "recent"],
        help=(
            "strict requires positive recent and whole-session returns versus SPY; "
            "recent requires only positive recent return versus SPY"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("logs/guarded_trend_evaluation"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    environments = {
        "live": args.root / "logs" / "live",
        "live2": args.root / "logs" / "live2",
        "paper": args.root / "logs_paper",
    }
    output_dir = args.root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation": {
            "horizons": args.horizons,
            "max_probability_gap": args.max_probability_gap,
            "minimum_volume_ratio": args.minimum_volume_ratio,
            "cost_bps": args.cost_bps,
            "common_conditions": [
                "below normal BUY threshold but within probability gap",
                "positive session return and intraday momentum",
                "above VWAP with positive two-bar VWAP slope",
                "time-normalized volume at or above minimum",
            ],
            "relative_rules": {
                "strict": "positive recent and whole-session return versus SPY",
                "recent": "positive recent return versus SPY; whole-session relative return is informational",
            },
            "latest_entry": {
                str(horizon): (
                    f"{(16 * 60 - horizon) // 60:02d}:"
                    f"{(16 * 60 - horizon) % 60:02d} America/New_York"
                )
                for horizon in args.horizons
            },
        },
        "results": {},
    }
    all_rows = []

    for environment, directory in environments.items():
        report["results"][environment] = {}
        for symbol in args.symbols:
            path = directory / f"signals_{symbol.upper()}.csv"
            symbol_result = {"source": str(path)}
            report["results"][environment][symbol.upper()] = symbol_result
            if not path.is_file():
                symbol_result["status"] = "MISSING"
                continue
            try:
                history = load_signal_history(path)
            except (ValueError, pd.errors.ParserError) as exc:
                symbol_result.update(status="COLLECTING", reason=str(exc))
                continue

            symbol_result["complete_rows"] = int(
                history[list(REQUIRED_COLUMNS - {"timestamp", "relativemarketavailable"})]
                .notna().all(axis=1).sum()
            )
            symbol_result["rules"] = {}
            for relative_rule in args.relative_rules:
                funnel = guard_funnel(
                    history,
                    args.max_probability_gap,
                    args.minimum_volume_ratio,
                    relative_rule=relative_rule,
                )
                candidates = guarded_candidates(
                    history,
                    args.max_probability_gap,
                    args.minimum_volume_ratio,
                    relative_rule=relative_rule,
                )
                rule_result = {
                    "candidate_rows_before_horizon_cutoff": int(len(candidates)),
                    "funnel": funnel,
                    "horizons": {},
                }
                symbol_result["rules"][relative_rule] = rule_result
                for horizon in args.horizons:
                    eligible_before_sampling = horizon_eligible_count(
                        candidates, horizon
                    )
                    evaluated = evaluate_horizon(
                        history,
                        candidates,
                        horizon,
                        args.cost_bps,
                        args.tolerance_minutes,
                    )
                    summary = summarize(
                        evaluated, args.minimum_samples, args.minimum_sessions
                    )
                    summary["eligible_rows_before_nonoverlap_sampling"] = (
                        eligible_before_sampling
                    )
                    rule_result["horizons"][str(horizon)] = summary
                    if not evaluated.empty:
                        evaluated.insert(0, "environment", environment)
                        evaluated.insert(1, "symbol", symbol.upper())
                        evaluated.insert(2, "relative_rule", relative_rule)
                        all_rows.append(evaluated)
                    print(
                        f"[GUARDED TREND] {environment}/{symbol.upper()}/"
                        f"{relative_rule}/{horizon}m: "
                        f"{summary['status']} observations={summary['observations']} "
                        f"sessions={summary['sessions']} "
                        f"avg_net={summary['average_net_return']} "
                        f"pf={summary['profit_factor']}"
                    )
                print(
                    f"[GUARDED TREND FUNNEL] {environment}/{symbol.upper()}/"
                    f"{relative_rule}: complete={funnel['complete_rows']} "
                    + " ".join(
                        f"{name}={count}"
                        for name, count in funnel["cumulative_pass_counts"].items()
                    )
                )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    report_path = output_dir / f"guarded_trend_{stamp}.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    candidates_path = output_dir / f"guarded_trend_{stamp}.csv"
    if all_rows:
        pd.concat(all_rows, ignore_index=True).to_csv(candidates_path, index=False)
    else:
        pd.DataFrame().to_csv(candidates_path, index=False)
    print(f"[GUARDED TREND] Report: {report_path}")
    print(f"[GUARDED TREND] Observations: {candidates_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
