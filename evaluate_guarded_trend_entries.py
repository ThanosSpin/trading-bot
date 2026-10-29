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
) -> pd.DataFrame:
    complete = frame.dropna(subset=list(REQUIRED_COLUMNS - {"timestamp", "relativemarketavailable"}))
    mask = (
        (complete["finalprob"] < complete["buythreshold"])
        & (complete["finalprob"] >= complete["buythreshold"] - max_probability_gap)
        & (complete["sessionreturn"] > 0.0)
        & (complete["intradaymom"] > 0.0)
        & (complete["vwapdistance"] > 0.0)
        & (complete["vwapslope2"] > 0.0)
        & (complete["relativereturn4"] > 0.0)
        & (complete["relativesessionreturn"] > 0.0)
        & complete["relativemarketavailable"]
        & (complete["volumetimeratio"] >= minimum_volume_ratio)
    )
    return complete.loc[mask].copy()


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
        "rule": {
            "max_probability_gap": args.max_probability_gap,
            "minimum_volume_ratio": args.minimum_volume_ratio,
            "cost_bps": args.cost_bps,
            "conditions": [
                "below normal BUY threshold but within probability gap",
                "positive session return and intraday momentum",
                "above VWAP with positive two-bar VWAP slope",
                "positive 60-minute and session relative strength",
                "time-normalized volume at or above minimum",
            ],
        },
        "results": {},
    }
    all_rows = []

    for environment, directory in environments.items():
        report["results"][environment] = {}
        for symbol in args.symbols:
            path = directory / f"signals_{symbol.upper()}.csv"
            symbol_result = {"source": str(path), "horizons": {}}
            report["results"][environment][symbol.upper()] = symbol_result
            if not path.is_file():
                symbol_result["status"] = "MISSING"
                continue
            try:
                history = load_signal_history(path)
            except (ValueError, pd.errors.ParserError) as exc:
                symbol_result.update(status="COLLECTING", reason=str(exc))
                continue

            candidates = guarded_candidates(
                history,
                args.max_probability_gap,
                args.minimum_volume_ratio,
            )
            symbol_result["complete_rows"] = int(
                history[list(REQUIRED_COLUMNS - {"timestamp", "relativemarketavailable"})]
                .notna().all(axis=1).sum()
            )
            symbol_result["candidate_rows"] = int(len(candidates))
            for horizon in args.horizons:
                evaluated = evaluate_horizon(
                    history,
                    candidates,
                    horizon,
                    args.cost_bps,
                    args.tolerance_minutes,
                )
                summary = summarize(evaluated, args.minimum_samples, args.minimum_sessions)
                symbol_result["horizons"][str(horizon)] = summary
                if not evaluated.empty:
                    evaluated.insert(0, "environment", environment)
                    evaluated.insert(1, "symbol", symbol.upper())
                    all_rows.append(evaluated)
                print(
                    f"[GUARDED TREND] {environment}/{symbol.upper()}/{horizon}m: "
                    f"{summary['status']} observations={summary['observations']} "
                    f"sessions={summary['sessions']} avg_net={summary['average_net_return']} "
                    f"pf={summary['profit_factor']}"
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
