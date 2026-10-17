#!/usr/bin/env python3
"""Compare unified logistic and regularized-XGBoost intraday challengers."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import config
from predictive_model.features import build_intraday_features
from predictive_model.model_evaluation import (
    optimize_cost_aware_threshold,
    walk_forward_splits,
)


RAW_COLUMNS = {"Open", "High", "Low", "Close", "Volume"}
TARGET_DESIGNS = (
    {"name": "30m_050vol", "horizon_bars": 2, "volatility_multiplier": 0.50},
    {"name": "30m_075vol", "horizon_bars": 2, "volatility_multiplier": 0.75},
    {"name": "60m_050vol", "horizon_bars": 4, "volatility_multiplier": 0.50},
    {"name": "60m_075vol", "horizon_bars": 4, "volatility_multiplier": 0.75},
)
HISTORICAL_VALIDATION_SESSIONS = 63


def load_frame(data_dir: Path, symbol: str) -> pd.DataFrame:
    path = data_dir / f"{symbol}_15min.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing research dataset: {path}")
    frame = pd.read_parquet(path)
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame.sort_index()


def add_causal_regime_feature(frame: pd.DataFrame) -> pd.DataFrame:
    """Represent regime without splitting the training population."""
    out = frame.copy()
    close = pd.to_numeric(out["Close"], errors="coerce")
    return_1 = close.pct_change(fill_method=None)
    momentum = close.pct_change(4, fill_method=None).abs()
    volatility = return_1.rolling(12, min_periods=12).std()
    # Thresholds use earlier bars only; the current observation cannot alter
    # the boundary used to classify itself.
    momentum_threshold = momentum.shift(1).rolling(520, min_periods=52).quantile(0.70)
    volatility_threshold = volatility.shift(1).rolling(520, min_periods=52).quantile(0.70)
    out["regime_momentum"] = (
        (momentum >= momentum_threshold) | (volatility >= volatility_threshold)
    ).astype(float)
    out["regime_momentum_threshold"] = momentum_threshold
    out["regime_volatility_threshold"] = volatility_threshold
    return out


def prepare_dataset(
    symbol_frame: pd.DataFrame,
    spy_frame: pd.DataFrame,
    sector_frame: pd.DataFrame | None,
    sector_feature_set: str,
    horizon_bars: int,
    cost_bps: float,
    volatility_multiplier: float = 0.50,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    features = build_intraday_features(
        symbol_frame,
        benchmark_df=spy_frame,
        sector_df=sector_frame,
        sector_feature_set=sector_feature_set,
    )
    features = add_causal_regime_feature(features)
    close = pd.to_numeric(features["Close"], errors="coerce")
    forward_return = close.shift(-horizon_bars) / close - 1.0
    realized_volatility = close.pct_change(fill_method=None).rolling(26, min_periods=13).std()
    move_threshold = pd.concat(
        [
            pd.Series(float(cost_bps) / 10000.0, index=features.index),
            realized_volatility * np.sqrt(horizon_bars) * volatility_multiplier,
        ],
        axis=1,
    ).max(axis=1)
    target = (forward_return > move_threshold).astype(float)

    numeric = features.select_dtypes(include=[np.number]).copy()
    feature_columns = [
        column
        for column in numeric.columns
        if column not in RAW_COLUMNS and not column.lower().startswith("target")
    ]
    X = numeric.loc[:, feature_columns].replace([np.inf, -np.inf], np.nan)
    valid = X.notna().all(axis=1) & forward_return.notna() & move_threshold.notna()
    X = X.loc[valid]
    return (
        X,
        target.loc[valid].astype(int),
        forward_return.loc[valid],
        move_threshold.loc[valid],
    )


def split_development_holdout(
    frame: pd.DataFrame,
    holdout_sessions: int = HISTORICAL_VALIDATION_SESSIONS,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Reserve the latest complete NY trading sessions for historical validation."""
    if holdout_sessions < 1:
        raise ValueError("holdout_sessions must be positive")
    index = pd.DatetimeIndex(frame.index)
    if index.tz is None:
        index = index.tz_localize("UTC")
    session_dates = pd.Index(index.tz_convert("America/New_York").date).unique()
    if len(session_dates) <= holdout_sessions:
        raise ValueError(
            f"dataset has {len(session_dates)} sessions; need more than {holdout_sessions}"
        )
    first_holdout_session = session_dates[-holdout_sessions]
    holdout_mask = np.asarray(
        index.tz_convert("America/New_York").date >= first_holdout_session,
        dtype=bool,
    )
    development = frame.loc[~holdout_mask].copy()
    holdout = frame.loc[holdout_mask].copy()
    return development, holdout, str(first_holdout_session)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_manifest_hash(manifest_path: Path) -> str:
    manifest = json.loads(manifest_path.read_text())
    canonical = json.dumps(manifest.get("files", {}), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def verify_dataset_manifest(data_dir: Path, manifest_path: Path) -> str:
    manifest = json.loads(manifest_path.read_text())
    for filename, expected in manifest.get("files", {}).items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"frozen dataset file is missing: {path}")
        actual = file_sha256(path)
        if actual != expected.get("sha256"):
            raise ValueError(f"frozen dataset changed: {filename}")
    return str(manifest.get("dataset_id") or dataset_manifest_hash(manifest_path))


def model_factory(kind: str):
    if kind == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.10,
                penalty="l2",
                class_weight="balanced",
                max_iter=2000,
                random_state=42,
            ),
        )
    if kind == "xgboost":
        return XGBClassifier(
            n_estimators=250,
            max_depth=2,
            learning_rate=0.025,
            min_child_weight=12,
            subsample=0.75,
            colsample_bytree=0.65,
            reg_alpha=1.0,
            reg_lambda=12.0,
            gamma=0.20,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=1,
            random_state=42,
        )
    raise ValueError(f"unknown model kind: {kind}")


def aggregate_non_overlapping_trades(
    forward_returns: np.ndarray,
    trade_mask: np.ndarray,
    cost_bps: float,
) -> dict:
    """Summarize independent, horizon-spaced entries after round-trip costs."""
    selected = np.asarray(forward_returns, dtype=float)[
        np.asarray(trade_mask, dtype=bool)
    ] - float(cost_bps) / 10000.0
    if not len(selected):
        return {
            "trade_count": 0,
            "net_return": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
        }
    equity = np.cumprod(1.0 + selected)
    running_max = np.maximum.accumulate(np.r_[1.0, equity])[:-1]
    drawdown = equity / running_max - 1.0
    wins = selected[selected > 0]
    losses = selected[selected <= 0]
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) else 0.0
    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else (10.0 if gross_profit > 0 else 0.0)
    )
    return {
        "trade_count": int(len(selected)),
        "net_return": float(equity[-1] - 1.0),
        "profit_factor": float(profit_factor),
        "max_drawdown": float(drawdown.min()),
    }


def evaluate_model(
    X: pd.DataFrame,
    y: pd.Series,
    forward_returns: pd.Series,
    kind: str,
    horizon_bars: int,
    cost_bps: float,
    n_splits: int = 5,
) -> dict:
    records = []
    fold_summaries = []
    folds = walk_forward_splits(len(X), n_splits=n_splits, gap_bars=horizon_bars)
    for fold in folds:
        train_end = fold["train_end"]
        calibration_rows = max(100, int(train_end * 0.15))
        fit_end = train_end - calibration_rows - horizon_bars
        if fit_end < 500:
            raise ValueError(f"fold {fold['fold']} has insufficient fit rows")
        X_fit = X.iloc[:fit_end]
        y_fit = y.iloc[:fit_end]
        X_cal = X.iloc[fit_end + horizon_bars : train_end]
        y_cal = y.iloc[fit_end + horizon_bars : train_end]
        X_test = X.iloc[fold["test_start"] : fold["test_end"]]
        y_test = y.iloc[fold["test_start"] : fold["test_end"]]
        returns_test = forward_returns.iloc[fold["test_start"] : fold["test_end"]]
        if y_fit.nunique() < 2 or y_cal.nunique() < 2:
            raise ValueError(f"fold {fold['fold']} lacks two target classes")

        estimator = model_factory(kind)
        estimator.fit(X_fit, y_fit)
        calibrated = CalibratedClassifierCV(estimator, method="sigmoid", cv="prefit", n_jobs=1)
        calibrated.fit(X_cal, y_cal)
        calibration_probability = calibrated.predict_proba(X_cal)[:, 1]
        threshold_result = optimize_cost_aware_threshold(
            y_cal.to_numpy(),
            calibration_probability,
            forward_returns.loc[X_cal.index].to_numpy(),
            cost_bps=cost_bps,
            minimum_trades=max(10, len(X_cal) // 100),
            periods_per_year=max(1, (252 * 26) // horizon_bars),
        )
        threshold = float(threshold_result["best_threshold"])
        calibration_metrics = threshold_result.get("selected_metrics") or {}
        threshold_eligible = bool(threshold_result.get("eligible"))

        # Score non-overlapping holding windows to avoid counting the same
        # forward move repeatedly.
        X_scored = X_test.iloc[::horizon_bars]
        probability = calibrated.predict_proba(X_scored)[:, 1]
        for timestamp, actual, future_return, prob in zip(
            X_scored.index,
            y_test.loc[X_scored.index],
            returns_test.loc[X_scored.index],
            probability,
        ):
            records.append(
                {
                    "timestamp": str(timestamp),
                    "fold": fold["fold"],
                    "actual": int(actual),
                    "forward_return": float(future_return),
                    "probability": float(prob),
                    "threshold": threshold,
                    "threshold_eligible": threshold_eligible,
                    "trade": bool(prob >= threshold),
                }
            )

        fold_records = [row for row in records if row["fold"] == fold["fold"]]
        fold_trade_mask = np.asarray(
            [row["trade"] for row in fold_records], dtype=bool
        )
        fold_test_metrics = aggregate_non_overlapping_trades(
            np.asarray([row["forward_return"] for row in fold_records], dtype=float),
            fold_trade_mask,
            cost_bps,
        )
        fold_summaries.append(
            {
                "fold": fold["fold"],
                "test_start": str(X_test.index.min()),
                "test_end": str(X_test.index.max()),
                "threshold": threshold,
                "threshold_eligible": threshold_eligible,
                "calibration_trade_count": int(
                    calibration_metrics.get("trade_count", 0)
                ),
                "calibration_net_return": float(
                    calibration_metrics.get("net_return", 0.0)
                ),
                "calibration_profit_factor": float(
                    calibration_metrics.get("profit_factor", 0.0)
                ),
                **{f"test_{key}": value for key, value in fold_test_metrics.items()},
            }
        )

    record_frame = pd.DataFrame(records)
    trade_mask = record_frame["trade"].to_numpy(dtype=bool)
    metrics = aggregate_non_overlapping_trades(
        record_frame["forward_return"].to_numpy(),
        trade_mask,
        cost_bps=cost_bps,
    )
    selected = record_frame.loc[record_frame["trade"]]
    eligible_folds = sum(row["threshold_eligible"] for row in fold_summaries)
    profitable_folds = sum(
        row["test_trade_count"] >= 5 and row["test_net_return"] > 0
        for row in fold_summaries
    )
    active_folds = sum(row["test_trade_count"] > 0 for row in fold_summaries)
    stable_candidate = bool(
        eligible_folds >= 3
        and profitable_folds >= 3
        and active_folds >= 4
        and metrics["trade_count"] >= 50
        and metrics["net_return"] > 0
        and metrics["profit_factor"] >= 1.20
        and metrics["max_drawdown"] >= -0.15
    )
    metrics.update(
        {
            "model_kind": kind,
            "folds": len(folds),
            "observations": len(record_frame),
            "positive_rate": float(record_frame["actual"].mean()),
            "mean_threshold": float(record_frame["threshold"].mean()),
            "trade_win_rate": (
                float((selected["forward_return"] > 0).mean())
                if not selected.empty else 0.0
            ),
            "eligible_folds": int(eligible_folds),
            "active_folds": int(active_folds),
            "profitable_folds": int(profitable_folds),
            "stable_candidate": stable_candidate,
            "fold_summaries": fold_summaries,
            "records": records,
        }
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=["NVDA", "AAPL", "ABBV", "PLTR"])
    parser.add_argument("--data-dir", default="research_data/intraday_15m")
    parser.add_argument("--output-dir", default="logs/unified_intraday")
    parser.add_argument(
        "--target-design",
        choices=["all", *[item["name"] for item in TARGET_DESIGNS]],
        default="all",
    )
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--holdout-sessions", type=int, default=HISTORICAL_VALIDATION_SESSIONS)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    manifest_path = Path(args.manifest).resolve()
    dataset_id = verify_dataset_manifest(data_dir, manifest_path)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    spy = load_frame(data_dir, config.SPY_SYMBOL)
    candidate_sets = {"AAPL": "returns", "PLTR": "all"}
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "development",
        "dataset_id": dataset_id,
        "manifest": str(manifest_path),
        "holdout_sessions": args.holdout_sessions,
        "cost_bps": args.cost_bps,
        "results": {},
    }
    for raw_symbol in args.symbols:
        symbol = str(raw_symbol).upper()
        symbol_frame = load_frame(data_dir, symbol)
        development_frame, holdout_frame, holdout_start = split_development_holdout(
            symbol_frame, args.holdout_sessions
        )
        sector_symbol = config.SECTOR_BENCHMARKS.get(symbol)
        sector = load_frame(data_dir, sector_symbol) if sector_symbol else None
        development_end = development_frame.index.max()
        spy_development = spy.loc[spy.index <= development_end]
        sector_development = (
            sector.loc[sector.index <= development_end] if sector is not None else None
        )
        feature_set = candidate_sets.get(symbol, "baseline")
        symbol_result = {
            "development_raw_rows": len(development_frame),
            "holdout_raw_rows": len(holdout_frame),
            "development_end": str(development_end),
            "historical_validation_start": holdout_start,
            "sector_benchmark": sector_symbol,
            "sector_feature_set": feature_set,
            "designs": {},
        }
        designs = [
            item for item in TARGET_DESIGNS
            if args.target_design in ("all", item["name"])
        ]
        for design in designs:
            X, y, returns, move_threshold = prepare_dataset(
                development_frame,
                spy_development,
                sector_development,
                feature_set,
                design["horizon_bars"],
                args.cost_bps,
                design["volatility_multiplier"],
            )
            design_result = {
                "horizon_bars": design["horizon_bars"],
                "horizon_minutes": design["horizon_bars"] * 15,
                "volatility_multiplier": design["volatility_multiplier"],
                "rows": len(X),
                "features": len(X.columns),
                "mean_move_threshold": float(move_threshold.mean()),
                "models": {},
            }
            for kind in ("logistic", "xgboost"):
                metrics = evaluate_model(
                    X, y, returns, kind, design["horizon_bars"], args.cost_bps
                )
                design_result["models"][kind] = metrics
                print(
                    f"[UNIFIED] {symbol}/{design['name']}/{kind}: rows={len(X)} "
                    f"trades={metrics['trade_count']} net={metrics['net_return']:.2%} "
                    f"pf={metrics['profit_factor']:.3f} "
                    f"dd={metrics['max_drawdown']:.2%} "
                    f"eligible_folds={metrics['eligible_folds']}/5 "
                    f"profitable_folds={metrics['profitable_folds']}/5 "
                    f"stable={metrics['stable_candidate']}"
                )
            symbol_result["designs"][design["name"]] = design_result
        report["results"][symbol] = symbol_result

    path = output_dir / f"unified_intraday_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"[UNIFIED] Report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
