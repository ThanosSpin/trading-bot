"""Leakage-aware model evaluation and champion/challenger promotion gates."""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss
from xgboost import XGBClassifier


EVALUATION_VERSION = 3
DEFAULT_COST_BPS = 10.0


def _finite(value, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def walk_forward_splits(
    n_samples: int,
    n_splits: int = 4,
    gap_bars: int = 1,
    min_train_fraction: float = 0.50,
) -> List[Dict[str, int]]:
    """Create expanding-window folds with an embargo before every test window."""
    if n_samples < 120:
        raise ValueError(f"walk-forward evaluation requires at least 120 rows; got {n_samples}")
    if n_splits < 2:
        raise ValueError("walk-forward evaluation requires at least two folds")
    if gap_bars < 1:
        raise ValueError("gap_bars must be at least one")

    initial_train = max(80, int(n_samples * min_train_fraction))
    available = n_samples - initial_train - gap_bars
    test_size = available // n_splits
    if test_size < 15:
        raise ValueError(
            f"walk-forward test windows are too small: {test_size} rows per fold"
        )

    folds = []
    for fold_number in range(n_splits):
        train_end = initial_train + fold_number * test_size
        test_start = train_end + gap_bars
        test_end = n_samples if fold_number == n_splits - 1 else test_start + test_size
        if test_end - test_start < 15:
            continue
        folds.append(
            {
                "fold": fold_number + 1,
                "train_start": 0,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "gap_bars": gap_bars,
            }
        )
    if len(folds) < 2:
        raise ValueError("fewer than two usable walk-forward folds")
    return folds


def cost_aware_metrics(
    actual: np.ndarray,
    probability: np.ndarray,
    forward_returns: np.ndarray,
    threshold: float,
    cost_bps: float = DEFAULT_COST_BPS,
    periods_per_year: int = 252,
    group_ids: Optional[np.ndarray] = None,
) -> dict:
    """Evaluate a held long position after an explicit round-trip cost assumption."""
    actual = np.asarray(actual, dtype=int)
    probability = np.asarray(probability, dtype=float)
    forward_returns = np.asarray(forward_returns, dtype=float)
    if not (len(actual) == len(probability) == len(forward_returns)):
        raise ValueError("evaluation arrays must have identical lengths")
    if len(actual) == 0:
        raise ValueError("cannot evaluate an empty prediction set")

    if group_ids is None:
        group_ids = np.zeros(len(actual), dtype=int)
    else:
        group_ids = np.asarray(group_ids)
        if len(group_ids) != len(actual):
            raise ValueError("group_ids must match the evaluation arrays")

    signal = probability >= float(threshold)
    per_trade_cost = float(cost_bps) / 10000.0
    previous_signal = np.r_[False, signal[:-1]]
    group_changed = np.r_[True, group_ids[1:] != group_ids[:-1]]
    entries = signal & (~previous_signal | group_changed)

    # Consecutive positive bars represent one continuously held position.
    # Charge the configured round-trip allowance once when that position opens.
    strategy_returns = np.where(signal, forward_returns, 0.0)
    strategy_returns[entries] -= per_trade_cost

    trade_returns = []
    for start in np.flatnonzero(entries):
        end = start + 1
        while (
            end < len(signal)
            and signal[end]
            and group_ids[end] == group_ids[start]
        ):
            end += 1
        trade_returns.append(float(np.prod(1.0 + strategy_returns[start:end]) - 1.0))
    trade_returns = np.asarray(trade_returns, dtype=float)
    equity = np.cumprod(1.0 + strategy_returns)
    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1.0
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns <= 0]
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) else 0.0
    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else (10.0 if gross_profit > 0 else 0.0)
    )
    std = float(np.std(strategy_returns, ddof=1)) if len(strategy_returns) > 1 else 0.0
    sharpe = (
        float(np.mean(strategy_returns) / std * np.sqrt(periods_per_year))
        if std > 0
        else 0.0
    )

    return {
        "samples": int(len(actual)),
        "accuracy": float(accuracy_score(actual, signal.astype(int))),
        "brier_score": (
            float(brier_score_loss(actual, probability))
            if len(np.unique(actual)) >= 2
            else None
        ),
        "logloss": (
            float(log_loss(actual, probability, labels=[0, 1]))
            if len(np.unique(actual)) >= 2
            else None
        ),
        "threshold": float(threshold),
        "cost_bps": float(cost_bps),
        "trade_count": int(entries.sum()),
        "bars_in_market": int(signal.sum()),
        "win_rate": float(np.mean(trade_returns > 0)) if len(trade_returns) else 0.0,
        "average_net_trade_return": float(np.mean(trade_returns)) if len(trade_returns) else 0.0,
        "net_return": float(equity[-1] - 1.0),
        "profit_factor": float(min(profit_factor, 10.0)),
        "sharpe": sharpe,
        "max_drawdown": float(np.min(drawdown)) if len(drawdown) else 0.0,
    }


def _best_threshold(actual: pd.Series, probability: np.ndarray) -> float:
    best_threshold = 0.50
    best_score = -1.0
    for threshold in np.arange(0.35, 0.651, 0.01):
        score = f1_score(actual, probability >= threshold, zero_division=0)
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold


def evaluate_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    forward_returns: pd.Series,
    model_params: dict,
    mode: str,
    n_splits: int = 4,
    gap_bars: int = 1,
    cost_bps: float = DEFAULT_COST_BPS,
    max_features: Optional[int] = None,
    training_target: Optional[pd.Series] = None,
) -> dict:
    """Fit expanding folds and test every eligible out-of-fold market bar."""
    if training_target is None:
        training_target = y.copy()
    if not (
        len(X) == len(y) == len(forward_returns) == len(training_target)
    ):
        raise ValueError("walk-forward inputs must have identical lengths")

    folds = walk_forward_splits(len(X), n_splits=n_splits, gap_bars=gap_bars)
    records = []
    fold_summaries = []
    periods_per_year = 252 if mode == "daily" else 252 * 26

    for fold in folds:
        train_end = fold["train_end"]
        calibration_rows = max(20, int(train_end * 0.15))
        fit_end = train_end - calibration_rows - gap_bars
        if fit_end < 50:
            raise ValueError(f"fold {fold['fold']} has insufficient fit rows")

        X_fit = X.iloc[:fit_end]
        y_fit = training_target.iloc[:fit_end]
        X_cal = X.iloc[fit_end + gap_bars : train_end]
        y_cal = training_target.iloc[fit_end + gap_bars : train_end]
        X_test = X.iloc[fold["test_start"] : fold["test_end"]]
        y_test = y.iloc[fold["test_start"] : fold["test_end"]]
        returns_test = forward_returns.iloc[fold["test_start"] : fold["test_end"]]

        fit_mask = y_fit.notna()
        calibration_mask = y_cal.notna()
        X_fit = X_fit.loc[fit_mask]
        y_fit = y_fit.loc[fit_mask].astype(int)
        X_cal = X_cal.loc[calibration_mask]
        y_cal = y_cal.loc[calibration_mask].astype(int)

        if y_fit.nunique() < 2 or y_cal.nunique() < 2:
            raise ValueError(f"fold {fold['fold']} lacks two classes in fit/calibration")

        estimator = XGBClassifier(**model_params)
        estimator.fit(X_fit, y_fit, eval_set=[(X_cal, y_cal)], verbose=False)

        fold_features = list(X.columns)
        if max_features and 0 < max_features < len(fold_features):
            importance = np.asarray(estimator.feature_importances_, dtype=float)
            if len(importance) != len(fold_features):
                raise ValueError(
                    f"fold {fold['fold']} feature importance/schema mismatch"
                )
            ranked = np.argsort(importance)[::-1][: int(max_features)]
            fold_features = [fold_features[index] for index in ranked]
            X_fit = X_fit.loc[:, fold_features]
            X_cal = X_cal.loc[:, fold_features]
            X_test = X_test.loc[:, fold_features]
            estimator = XGBClassifier(**model_params)
            estimator.fit(X_fit, y_fit, eval_set=[(X_cal, y_cal)], verbose=False)
        calibrated = CalibratedClassifierCV(
            estimator,
            method="sigmoid",
            cv="prefit",
            # The estimator already carries the configured worker limit.
            # Keeping calibration single-process avoids duplicating a fitted
            # XGBoost model in memory on small VMs.
            n_jobs=1,
        )
        calibrated.fit(X_cal, y_cal)
        cal_probability = calibrated.predict_proba(X_cal)[:, 1]
        threshold = _best_threshold(y_cal, cal_probability)
        probability = calibrated.predict_proba(X_test)[:, 1]
        fold_metrics = cost_aware_metrics(
            y_test.to_numpy(),
            probability,
            returns_test.to_numpy(),
            threshold=threshold,
            cost_bps=cost_bps,
            periods_per_year=periods_per_year,
        )
        fold_summaries.append(
            {
                **fold,
                "fit_rows": int(len(X_fit)),
                "calibration_rows": int(len(X_cal)),
                "test_rows_all_market_bars": int(len(X_test)),
                "feature_count": int(len(fold_features)),
                "train_end_timestamp": str(X.index[train_end - 1]),
                "test_start_timestamp": str(X_test.index[0]),
                "test_end_timestamp": str(X_test.index[-1]),
                "metrics": fold_metrics,
            }
        )
        for timestamp, target, predicted, realized in zip(
            X_test.index,
            y_test.to_numpy(),
            probability,
            returns_test.to_numpy(),
        ):
            records.append(
                {
                    "timestamp": str(timestamp),
                    "actual": int(target),
                    "probability": float(predicted),
                    "forward_return": float(realized),
                    "threshold": float(threshold),
                    "fold": int(fold["fold"]),
                }
            )

    frame = pd.DataFrame(records)
    # Aggregate using each fold's independently selected threshold.
    signals = frame["probability"].to_numpy() >= frame["threshold"].to_numpy()
    aggregate = _metrics_from_signals(
        frame,
        signals,
        cost_bps=cost_bps,
        periods_per_year=periods_per_year,
    )
    return {
        "version": EVALUATION_VERSION,
        "method": "expanding_walk_forward_with_embargo",
        "evaluation_universe": "all_eligible_bars_including_neutral_moves",
        "mode": mode,
        "n_splits": len(fold_summaries),
        "gap_bars": int(gap_bars),
        "cost_bps": float(cost_bps),
        "fold_local_feature_selection": bool(
            max_features and 0 < max_features < X.shape[1]
        ),
        "max_features": int(max_features) if max_features else None,
        "folds": fold_summaries,
        "aggregate": aggregate,
        "predictions": records,
    }


def _metrics_from_signals(
    frame: pd.DataFrame,
    signals: np.ndarray,
    cost_bps: float,
    periods_per_year: int,
) -> dict:
    probabilities = frame["probability"].to_numpy(dtype=float)
    actual = frame["actual"].to_numpy(dtype=int)
    returns = frame["forward_return"].to_numpy(dtype=float)
    # Use a synthetic probability vector that exactly reproduces per-fold signals
    # for trading metrics, then restore probability-based calibration metrics.
    synthetic = np.where(signals, 1.0, 0.0)
    metrics = cost_aware_metrics(
        actual,
        synthetic,
        returns,
        threshold=0.5,
        cost_bps=cost_bps,
        periods_per_year=periods_per_year,
        group_ids=(
            frame["fold"].to_numpy()
            if "fold" in frame.columns
            else None
        ),
    )
    metrics["accuracy"] = float(accuracy_score(actual, signals.astype(int)))
    metrics["brier_score"] = (
        float(brier_score_loss(actual, probabilities))
        if len(np.unique(actual)) >= 2
        else None
    )
    metrics["logloss"] = (
        float(log_loss(actual, probabilities, labels=[0, 1]))
        if len(np.unique(actual)) >= 2
        else None
    )
    metrics["threshold"] = "per_fold"
    return metrics


def promotion_decision(candidate: dict, champion: Optional[dict]) -> dict:
    """Decide whether a challenger has enough evidence to replace the champion."""
    evaluation = candidate.get("walk_forward_evaluation") or {}
    aggregate = evaluation.get("aggregate") or {}
    mode = str(candidate.get("mode", ""))
    minimum_trades = 8 if mode == "daily" else 20
    reasons = []

    if evaluation.get("version") != EVALUATION_VERSION:
        reasons.append("candidate has no supported walk-forward evaluation")
    if int(evaluation.get("n_splits", 0)) < 3:
        reasons.append("candidate has fewer than three walk-forward folds")
    if int(evaluation.get("gap_bars", 0)) < 1:
        reasons.append("candidate evaluation has no embargo gap")
    if int(aggregate.get("trade_count", 0)) < minimum_trades:
        reasons.append(f"candidate has fewer than {minimum_trades} evaluated trades")
    if _finite(aggregate.get("net_return"), -1.0) <= 0:
        reasons.append("candidate cost-aware net return is not positive")
    if _finite(aggregate.get("profit_factor")) < 1.0:
        reasons.append("candidate cost-aware profit factor is below 1.0")
    brier = aggregate.get("brier_score")
    if brier is None or _finite(brier, 1.0) > 0.27:
        reasons.append("candidate Brier score exceeds 0.27")
    if _finite(aggregate.get("max_drawdown"), -1.0) < -0.20:
        reasons.append("candidate walk-forward drawdown exceeds 20%")

    result = {
        "accepted": False,
        "kind": "absolute_gate",
        "reasons": reasons,
        "candidate": aggregate,
        "champion": None,
        "overlap_samples": 0,
    }
    if reasons:
        return result

    champion_eval = (champion or {}).get("walk_forward_evaluation") or {}
    champion_records = champion_eval.get("predictions") or []
    candidate_records = evaluation.get("predictions") or []
    if not champion_records:
        result["accepted"] = True
        result["reasons"] = ["champion predates walk-forward evaluation; absolute gates passed"]
        return result

    candidate_frame = pd.DataFrame(candidate_records).drop_duplicates("timestamp", keep="last")
    champion_frame = pd.DataFrame(champion_records).drop_duplicates("timestamp", keep="last")
    overlap = candidate_frame.merge(
        champion_frame,
        on="timestamp",
        suffixes=("_candidate", "_champion"),
    )
    minimum_overlap = 30 if mode == "daily" else 80
    if len(overlap) < minimum_overlap:
        result["reasons"] = [
            f"only {len(overlap)} overlapping observations; need {minimum_overlap}"
        ]
        return result

    comparison_frame = pd.DataFrame(
        {
            "actual": overlap["actual_candidate"],
            "forward_return": overlap["forward_return_candidate"],
            "fold": overlap["fold_candidate"],
        }
    )
    candidate_comparison = comparison_frame.assign(
        probability=overlap["probability_candidate"],
        threshold=overlap["threshold_candidate"],
    )
    champion_comparison = comparison_frame.assign(
        probability=overlap["probability_champion"],
        threshold=overlap["threshold_champion"],
    )
    periods_per_year = 252 if mode == "daily" else 252 * 26
    cost_bps = _finite(evaluation.get("cost_bps"), DEFAULT_COST_BPS)
    candidate_metrics = _metrics_from_signals(
        candidate_comparison,
        candidate_comparison["probability"].to_numpy()
        >= candidate_comparison["threshold"].to_numpy(),
        cost_bps,
        periods_per_year,
    )
    champion_metrics = _metrics_from_signals(
        champion_comparison,
        champion_comparison["probability"].to_numpy()
        >= champion_comparison["threshold"].to_numpy(),
        cost_bps,
        periods_per_year,
    )

    comparison_reasons = []
    if candidate_metrics["net_return"] < champion_metrics["net_return"] - 0.002:
        comparison_reasons.append("challenger net return trails champion by more than 0.20%")
    if candidate_metrics["brier_score"] > champion_metrics["brier_score"] + 0.02:
        comparison_reasons.append("challenger Brier score is more than 0.02 worse")
    if candidate_metrics["max_drawdown"] < champion_metrics["max_drawdown"] - 0.02:
        comparison_reasons.append("challenger drawdown is more than 2% worse")
    material_improvement = (
        candidate_metrics["net_return"] > champion_metrics["net_return"] + 0.001
        or candidate_metrics["brier_score"] < champion_metrics["brier_score"] - 0.01
    )
    if not material_improvement:
        comparison_reasons.append("challenger has no material return or calibration improvement")

    result.update(
        {
            "accepted": not comparison_reasons,
            "kind": "champion_challenger",
            "reasons": comparison_reasons or ["challenger passed direct comparison"],
            "candidate": candidate_metrics,
            "champion": champion_metrics,
            "overlap_samples": int(len(overlap)),
        }
    )
    return result
