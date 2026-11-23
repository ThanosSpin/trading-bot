"""Calibrated, pooled intraday-metrics fallback for stale XGBoost artifacts.

The fallback is trained from the same 15-minute feature pipeline as the main
intraday models. A pooled logistic estimator is enabled per symbol and session
only when a later chronological holdout supports it. It is never used as an
unvalidated hand-written conversion from returns to probabilities.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    INTRADAY_METRICS_FALLBACK_MAX_AGE_DAYS,
    INTRADAY_METRICS_FALLBACK_MIN_BRIER_IMPROVEMENT,
    INTRADAY_METRICS_FALLBACK_MIN_ROC_AUC,
    INTRADAY_METRICS_FALLBACK_MIN_TEST_SAMPLES,
    INTRADAY_METRICS_FALLBACK_MIN_TEST_SESSIONS,
    INTRADAY_METRICS_FALLBACK_MAX_BRIER,
    MODEL_DIR,
    SPY_SYMBOL,
    TRAIN_SYMBOLS,
)
from predictive_model.data_loader import fetch_intraday_history
from predictive_model.features import build_intraday_features
from predictive_model.time_features import intraday_session_keys, intraday_session_types


SESSIONS = ("premarket", "regular", "after_hours")
# Use return and relative-return measures only. VWAP and volume are intentionally
# excluded from this probability fallback; those remain separate strategy inputs.
METRIC_FEATURES = (
    "return_1",
    "return_5",
    "momentum_accel_5",
    "relative_return_2",
    "relative_return_4",
    "relative_session_return",
)
ARTIFACT_PATH = Path(MODEL_DIR) / "intraday_metrics_fallback.pkl"
_artifact_cache = {"mtime": None, "artifact": None}


def _symbol_list():
    symbols = list(TRAIN_SYMBOLS) if isinstance(TRAIN_SYMBOLS, (list, tuple)) else [TRAIN_SYMBOLS]
    symbols.append(SPY_SYMBOL)
    return list(dict.fromkeys(str(symbol).upper() for symbol in symbols if symbol))


def _add_indicator_columns(frame: pd.DataFrame, symbols) -> pd.DataFrame:
    result = frame.loc[:, list(METRIC_FEATURES)].copy()
    for symbol in symbols:
        result[f"symbol__{symbol}"] = (frame["symbol"] == symbol).astype(float)
    for session in SESSIONS:
        result[f"session__{session}"] = (frame["session"] == session).astype(float)
    return result


def _build_symbol_rows(symbol: str, bars: pd.DataFrame, benchmark: pd.DataFrame):
    features = build_intraday_features(
        bars,
        benchmark_df=benchmark,
        diagnostic_label=f"{symbol}/metrics_fallback",
    )
    if features is None or features.empty or "Close" not in features:
        return pd.DataFrame()

    index = pd.DatetimeIndex(features.index)
    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")
    features = features.copy()
    features.index = index

    close = pd.to_numeric(features["Close"], errors="coerce")
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    forward_return = close.shift(-1) / close - 1.0
    session_types = intraday_session_types(index)
    session_keys = intraday_session_keys(index)
    same_session = session_keys.eq(session_keys.shift(-1))
    valid = (
        forward_return.notna()
        & same_session
        & session_types.isin(SESSIONS)
    )

    missing = [name for name in METRIC_FEATURES if name not in features.columns]
    if missing:
        raise ValueError(f"{symbol}: missing fallback features {missing}")

    rows = features.loc[valid, list(METRIC_FEATURES)].copy()
    rows["symbol"] = str(symbol).upper()
    rows["session"] = session_types.loc[valid].to_numpy()
    rows["session_date"] = index[valid.to_numpy()].tz_convert(
        "America/New_York"
    ).date
    rows["target"] = (forward_return.loc[valid] > 0.0).astype(int).to_numpy()
    return rows.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)


def _logit(probabilities):
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


def _artifact_age_days(artifact, now=None):
    try:
        freshness_timestamp = artifact.get(
            "calibration_data_through", artifact.get("trained_at")
        )
        trained_at = pd.to_datetime(freshness_timestamp, utc=True, errors="coerce")
        if pd.isna(trained_at):
            return None
        now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
        if now.tzinfo is None:
            now = now.tz_localize("UTC")
        return max(0.0, (now - trained_at).total_seconds() / 86400.0)
    except Exception:
        return None


def load_metrics_fallback() -> Optional[dict]:
    """Load the validated fallback artifact, caching by file modification time."""
    try:
        mtime = ARTIFACT_PATH.stat().st_mtime
    except OSError:
        return None
    if _artifact_cache["mtime"] == mtime:
        return _artifact_cache["artifact"]
    try:
        artifact = joblib.load(ARTIFACT_PATH)
        _artifact_cache.update({"mtime": mtime, "artifact": artifact})
        return artifact
    except Exception as exc:
        print(f"[METRICS FALLBACK] Could not load artifact: {exc}")
        return None


def _scope_key(symbol: str, session: str) -> str:
    return f"{str(symbol).upper()}|{str(session).lower()}"


def metrics_fallback_is_ready(artifact: dict, symbol: str, session: str) -> bool:
    symbol = str(symbol).upper()
    session = str(session).lower()
    if session == "pre_market":
        session = "premarket"
    key = _scope_key(symbol, session)
    scope_dates = artifact.get("scope_calibration_dates") or {}
    scope_models = artifact.get("scope_models") or {}
    freshness_source = scope_dates.get(key)
    if freshness_source is None:
        freshness_source = (scope_models.get(key) or {}).get(
            "calibration_data_through"
        )
    age_artifact = artifact if freshness_source is None else {
        "calibration_data_through": freshness_source
    }
    age_days = _artifact_age_days(age_artifact)
    return bool(
        artifact
        and session in SESSIONS
        and age_days is not None
        and age_days < float(INTRADAY_METRICS_FALLBACK_MAX_AGE_DAYS)
        and (artifact.get("supported") or {}).get(symbol, {}).get(session, False)
    )


def predict_metrics_fallback(
    artifact: dict,
    symbol: str,
    session: str,
    feature_frame: pd.DataFrame,
) -> Optional[float]:
    """Return a calibrated probability only for validated, fresh artifact scopes."""
    symbol = str(symbol).upper()
    session = str(session).lower()
    if session == "pre_market":
        session = "premarket"
    if not metrics_fallback_is_ready(artifact, symbol, session):
        return None
    if feature_frame is None or feature_frame.empty:
        return None

    latest = feature_frame.iloc[-1]
    values = pd.to_numeric(latest.reindex(METRIC_FEATURES), errors="coerce")
    if int(values.notna().sum()) < 4:
        return None

    key = _scope_key(symbol, session)
    scope_model = (artifact.get("scope_models") or {}).get(key) or {}
    symbols = scope_model.get("symbols") or artifact.get("symbols") or []
    if symbol not in symbols:
        return None
    row = pd.DataFrame([values.to_dict()])
    row["symbol"] = symbol
    row["session"] = session
    features = scope_model.get("features") or artifact.get("features", [])
    x = _add_indicator_columns(row, symbols).reindex(
        columns=features, fill_value=0.0
    )
    model = scope_model.get("model") or artifact["model"]
    calibrator = (
        scope_model.get("calibrator")
        or (artifact.get("calibrators") or {}).get(key)
        or artifact["calibrator"]
    )
    raw_probability = float(model.predict_proba(x)[0, 1])
    calibrated = calibrator.predict_proba(_logit([raw_probability]))[0, 1]
    if not np.isfinite(calibrated):
        return None
    return float(np.clip(calibrated, 0.0, 1.0))


def train_metrics_fallback() -> bool:
    """Fit and validate a pooled metrics estimator; save only if session gates pass."""
    symbols = _symbol_list()
    lookback_minutes = 60 * 24 * 60
    data_rows = []
    benchmark = None

    for symbol in symbols:
        bars = fetch_intraday_history(
            symbol,
            lookback_minutes=lookback_minutes,
            interval="15m",
            include_extended_hours=True,
        )
        if bars is None or bars.empty:
            print(f"[METRICS FALLBACK] {symbol}: no Yahoo extended-session history")
            continue
        if symbol == str(SPY_SYMBOL).upper():
            benchmark = bars
        data_rows.append((symbol, bars))

    if benchmark is None:
        print("[METRICS FALLBACK] No SPY benchmark history; keeping current artifact")
        return False

    frames = []
    for symbol, bars in data_rows:
        try:
            frames.append(_build_symbol_rows(symbol, bars, benchmark))
        except Exception as exc:
            print(f"[METRICS FALLBACK] {symbol}: feature build failed: {exc}")
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        print("[METRICS FALLBACK] No labeled feature rows; keeping current artifact")
        return False

    data = pd.concat(frames, ignore_index=True).sort_values(
        "session_date", kind="stable"
    ).reset_index(drop=True)
    symbols_with_data = sorted(data["symbol"].unique().tolist())
    unique_dates = sorted(data["session_date"].unique())
    if len(unique_dates) < 40:
        print(
            f"[METRICS FALLBACK] Need at least 40 chronological sessions; "
            f"got {len(unique_dates)}. Keeping current artifact."
        )
        return False

    train_end = max(12, int(len(unique_dates) * 0.60))
    calibration_end = max(train_end + 5, int(len(unique_dates) * 0.80))
    if calibration_end >= len(unique_dates) - 5:
        print("[METRICS FALLBACK] Chronological calibration/test windows too small")
        return False
    # Keep ordered date lists for artifact metadata, and sets for efficient
    # membership filtering. Sets cannot safely supply chronological endpoints.
    train_date_list = unique_dates[:train_end]
    calibration_date_list = unique_dates[train_end:calibration_end]
    test_date_list = unique_dates[calibration_end:]
    train_dates = set(train_date_list)
    calibration_dates = set(calibration_date_list)
    test_dates = set(test_date_list)
    train = data[data["session_date"].isin(train_dates)].copy()
    calibration = data[data["session_date"].isin(calibration_dates)].copy()
    test = data[data["session_date"].isin(test_dates)].copy()

    y_train = train["target"].astype(int)
    y_cal = calibration["target"].astype(int)
    y_test = test["target"].astype(int)
    if y_train.nunique() < 2 or y_cal.nunique() < 2 or y_test.nunique() < 2:
        print("[METRICS FALLBACK] Chronological windows lack both outcome classes")
        return False

    feature_columns = list(METRIC_FEATURES) + [
        f"symbol__{symbol}" for symbol in symbols_with_data
    ] + [f"session__{session}" for session in SESSIONS]
    x_train = _add_indicator_columns(train, symbols_with_data).reindex(columns=feature_columns)
    x_cal = _add_indicator_columns(calibration, symbols_with_data).reindex(columns=feature_columns)
    x_test = _add_indicator_columns(test, symbols_with_data).reindex(columns=feature_columns)

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=0.5, max_iter=1000)),
        ]
    )
    model.fit(x_train, y_train)
    raw_cal = model.predict_proba(x_cal)[:, 1]
    global_calibrator = LogisticRegression(C=1.0, max_iter=500)
    global_calibrator.fit(_logit(raw_cal), y_cal)

    # A pooled calibrator can be biased for one ticker/session even when its
    # ranking is useful. Fit a small Platt map per scope using calibration data
    # only; scopes without enough calibration examples retain the pooled map.
    calibration = calibration.copy()
    calibration["raw_probability"] = raw_cal
    scope_calibrators = {}
    min_scope_calibration_samples = max(50, int(INTRADAY_METRICS_FALLBACK_MIN_TEST_SAMPLES) // 2)
    for symbol in symbols_with_data:
        for session in SESSIONS:
            key = _scope_key(symbol, session)
            rows = calibration[
                (calibration["symbol"] == symbol)
                & (calibration["session"] == session)
            ]
            if (
                len(rows) >= min_scope_calibration_samples
                and rows["session_date"].nunique() >= 5
                and rows["target"].nunique() == 2
            ):
                scope_calibrator = LogisticRegression(C=1.0, max_iter=500)
                scope_calibrator.fit(
                    _logit(rows["raw_probability"].to_numpy()),
                    rows["target"].astype(int),
                )
                scope_calibrators[key] = scope_calibrator

    test = test.copy()
    test["raw_probability"] = model.predict_proba(x_test)[:, 1]
    test["probability"] = np.nan
    for symbol in symbols_with_data:
        for session in SESSIONS:
            mask = (test["symbol"] == symbol) & (test["session"] == session)
            if not mask.any():
                continue
            calibrator_for_scope = scope_calibrators.get(
                _scope_key(symbol, session), global_calibrator
            )
            test.loc[mask, "probability"] = calibrator_for_scope.predict_proba(
                _logit(test.loc[mask, "raw_probability"].to_numpy())
            )[:, 1]
    supported: Dict[str, dict] = {}
    validation = {}
    for symbol in symbols_with_data:
        supported[symbol] = {}
        validation[symbol] = {}
        for session in SESSIONS:
            rows = test[(test["symbol"] == symbol) & (test["session"] == session)]
            cal_rows = calibration[
                (calibration["symbol"] == symbol)
                & (calibration["session"] == session)
            ]
            labels = rows["target"].astype(int).to_numpy()
            probabilities = rows["probability"].to_numpy(dtype=float)
            distinct_sessions = int(rows["session_date"].nunique())
            brier = (
                float(brier_score_loss(labels, probabilities))
                if len(rows) and len(np.unique(labels)) >= 2
                else None
            )
            baseline_rate = (
                float(cal_rows["target"].mean()) if not cal_rows.empty else None
            )
            baseline_brier = (
                float(brier_score_loss(labels, np.full(len(labels), baseline_rate)))
                if baseline_rate is not None and len(np.unique(labels)) >= 2
                else None
            )
            auc = (
                float(roc_auc_score(labels, probabilities))
                if len(np.unique(labels)) >= 2
                else None
            )
            ready = bool(
                len(rows) >= int(INTRADAY_METRICS_FALLBACK_MIN_TEST_SAMPLES)
                and distinct_sessions
                >= int(INTRADAY_METRICS_FALLBACK_MIN_TEST_SESSIONS)
                and brier is not None
                and brier <= float(INTRADAY_METRICS_FALLBACK_MAX_BRIER)
                and baseline_brier is not None
                and brier
                <= baseline_brier
                - float(INTRADAY_METRICS_FALLBACK_MIN_BRIER_IMPROVEMENT)
                and auc is not None
                and auc >= float(INTRADAY_METRICS_FALLBACK_MIN_ROC_AUC)
            )
            supported[symbol][session] = ready
            validation[symbol][session] = {
                "test_samples": int(len(rows)),
                "test_sessions": distinct_sessions,
                "brier_score": brier,
                "calibration_baseline_brier": baseline_brier,
                "roc_auc": auc,
                "ready": ready,
            }
            print(
                f"[METRICS FALLBACK VALIDATION] {symbol}/{session}: "
                f"n={len(rows)} sessions={distinct_sessions} brier={brier} "
                f"baseline_brier={baseline_brier} auc={auc} ready={ready}"
            )

    if not any(any(session_map.values()) for session_map in supported.values()):
        print("[METRICS FALLBACK] No symbol/session passed validation; keeping current artifact")
        return False

    scope_calibration_dates = {
        _scope_key(symbol, session): calibration_date_list[-1].isoformat()
        for symbol, sessions in supported.items()
        for session, ready in sessions.items()
        if ready
    }
    scope_models = {}
    candidate = {
        "model": model,
        "calibrator": global_calibrator,
        "calibrators": scope_calibrators,
        "features": feature_columns,
        "metric_features": list(METRIC_FEATURES),
        "symbols": symbols_with_data,
        "supported": supported,
        "scope_calibration_dates": scope_calibration_dates,
        "scope_models": scope_models,
        "validation": validation,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_data_through": train_date_list[-1].isoformat(),
        "calibration_data_through": calibration_date_list[-1].isoformat(),
        "validation_data_through": test_date_list[-1].isoformat(),
        "fit_window": {
            "train_sessions": len(train_date_list),
            "calibration_sessions": len(calibration_date_list),
            "test_sessions": len(test_date_list),
        },
        "target": "next_15m_close_up_same_session",
    }

    current = load_metrics_fallback()
    if current:
        current_supported = current.get("supported", {})
        current_scope_models = current.get("scope_models", {})
        current_scope_dates = current.get("scope_calibration_dates", {})
        for old_symbol, sessions in current_supported.items():
            for old_session, enabled in sessions.items():
                if not enabled or candidate["supported"].get(old_symbol, {}).get(old_session, False):
                    continue
                key = _scope_key(old_symbol, old_session)
                old_scope = current_scope_models.get(key)
                old_date = current_scope_dates.get(key)
                if old_scope is None:
                    old_date = old_date or current.get("calibration_data_through") or current.get("trained_at")
                    old_calibrator = (current.get("calibrators") or {}).get(key) or current.get("calibrator")
                    old_scope = {
                        "model": current.get("model"),
                        "calibrator": old_calibrator,
                        "features": current.get("features", []),
                        "symbols": current.get("symbols", []),
                        "calibration_data_through": old_date,
                    }
                else:
                    old_date = old_date or old_scope.get("calibration_data_through")
                old_scope_age = _artifact_age_days({"calibration_data_through": old_date})
                if (
                    old_scope.get("model") is None
                    or old_scope.get("calibrator") is None
                    or old_scope_age is None
                    or old_scope_age >= float(INTRADAY_METRICS_FALLBACK_MAX_AGE_DAYS)
                ):
                    continue
                old_scope = dict(old_scope)
                old_scope["calibration_data_through"] = old_date
                candidate["scope_models"][key] = old_scope
                candidate["scope_calibration_dates"][key] = old_date
                candidate["supported"].setdefault(old_symbol, {})[old_session] = True
                validation.setdefault(old_symbol, {}).setdefault(old_session, {})[
                    "retained_incumbent"
                ] = True
                print(
                    f"[METRICS FALLBACK] Retaining validated incumbent scope "
                    f"{old_symbol}/{old_session} while updating other scopes"
                )

    if not any(any(session_map.values()) for session_map in candidate["supported"].values()):
        print("[METRICS FALLBACK] No fresh symbol/session passed validation; keeping current artifact")
        return False

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = ARTIFACT_PATH.with_suffix(".pkl.tmp")
    joblib.dump(candidate, temporary)
    os.replace(temporary, ARTIFACT_PATH)
    _artifact_cache.update({"mtime": None, "artifact": None})
    print(f"[METRICS FALLBACK] Validated artifact saved: {ARTIFACT_PATH}")
    return True
