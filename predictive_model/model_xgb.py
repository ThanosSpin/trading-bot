# model_xgb.py
import os
import traceback
import warnings
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, Union
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from predictive_model.adaptive_thresholds import get_adaptive_regime_thresholds
from predictive_model.data_loader import (
    fetch_historical_data,
    fetch_intraday_history,
    fetch_latest_price,
    fetch_intraday_price_fallback,
)
from predictive_model.features import build_daily_features, build_intraday_features
from predictive_model.time_features import (
    intraday_session_keys,
    intraday_session_types,
)
from predictive_model.model_monitor import evaluate_predictions, log_prediction
from predictive_model.model_evaluation import evaluate_walk_forward
from predictive_model.target_labels import backtest_threshold, create_target_label
from predictive_model.trading_metrics import (
    calculate_financial_metrics,
    print_trading_report,
)

from config import (
    INTRADAY_MOM_TRIG,
    INTRADAY_REGIME_OVERRIDES,
    INTRADAY_VOL_TRIG,
    INTRADAY_WEIGHT,
    MIN_INTRADAY_BARS_FOR_FEATURES,
    MODEL_EVAL_GAP_BARS,
    MODEL_EVAL_TRANSACTION_COST_BPS,
    MODEL_EVAL_WALK_FORWARD_FOLDS,
    MODEL_TRAIN_N_JOBS,
    MODEL_DIR as CONFIG_MODEL_DIR,
    USE_TWO_STAGE_TARGETS,
    EXTENDED_HOURS_MAX_FALLBACK_PRICE_AGE_SECONDS,
    EXTENDED_HOURS_MAX_INTRADAY_BAR_AGE_MINUTES,
    EXTENDED_HOURS_MIN_MOMENTUM_BARS,
)
from market import get_current_session_start, get_market_session

# Resolve artifacts from the checkout, independent of the caller's current
# working directory.  Retraining and promotion code use config.MODEL_DIR too,
# so inference must read from that same location.
MODEL_DIR = Path(CONFIG_MODEL_DIR).resolve()
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CALIBRATOR_DIR = MODEL_DIR / "calibrators"
_calibrator_cache = {}

warnings.filterwarnings("ignore", message=".*cv='prefit'.*")

INTRADAY_TARGET_OVERRIDES = {
    ("SPY", "intraday_mr"): {
        "min_move": 0.0005,
    },
}

_intraday_regime_state = {}

# Artifacts produced by the former enhanced-indicator pipeline stored these
# column names with an initial capital letter.  Current feature generation
# correctly keeps them lowercase.  Inference may provide these exact aliases
# for old artifacts without weakening schema checks for unrelated features.
_LEGACY_CAPITALIZED_INDICATORS = {
    "adx",
    "adx_neg",
    "adx_pos",
    "adx_strong_trend",
    "adx_weak_trend",
    "aroon_down",
    "aroon_indicator",
    "aroon_up",
    "atr",
    "atr_pct",
    "bb_lower",
    "bb_mid",
    "bb_pct",
    "bb_upper",
    "bb_width",
    "cmf",
    "ema14_above_ema50",
    "ema_14",
    "ema_50",
    "keltner_lower",
    "keltner_mid",
    "keltner_upper",
    "macd",
    "macd_hist",
    "macd_signal",
    "obv",
    "price_above_bb_upper",
    "price_below_bb_lower",
    "roc",
    "rsi",
    "rsi_overbought",
    "rsi_oversold",
    "sma_20",
    "stoch_d",
    "stoch_k",
    "stoch_overbought",
    "stoch_oversold",
    "volume_ratio",
    "volume_sma_20",
}

# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------
def load_platt_calibrator(symbol: str, mode: str):
    """
    Load Platt calibrator for (symbol, mode) from models/calibrators.
    Returns None if missing. Uses simple in-memory cache.
    """
    sym = symbol.upper()
    key = (sym, mode)
    if key in _calibrator_cache:
        return _calibrator_cache[key]

    path = CALIBRATOR_DIR / f"{sym}_{mode}_platt.joblib"
    if path.exists():
        try:
            platt = joblib.load(path)
        except Exception:
            platt = None
    else:
        platt = None

    _calibrator_cache[key] = platt
    return platt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _clean_feature_dataframe(
    df: pd.DataFrame,
    *,
    drop_all_nan: bool = True,
) -> pd.DataFrame:
    """
    Clean DataFrame to ensure all columns are proper 1D Series.
    This fixes issues with XGBoost complaining about DataFrame columns.
    """
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(x) for x in col if str(x) != ""]).strip("_")
            for col in df.columns
        ]

    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]

    for col in list(df.columns):
        if isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]

        if hasattr(df[col], "values") and getattr(df[col].values, "ndim", 1) > 1:
            vals = df[col].values
            if vals.shape[0] == len(df):
                df[col] = pd.Series(vals[:, 0], index=df.index)

        df[col] = pd.to_numeric(df[col], errors="coerce")

    if drop_all_nan:
        df = df.dropna(axis=1, how="all")
    return df


def _validate_exact_feature_schema(
    df_features: pd.DataFrame,
    expected_features: list,
    allow_extra: bool = True,
) -> pd.DataFrame:
    """
    Enforce exact feature names and order for inference.
    This prevents silent train/live drift from fuzzy matching.
    """
    if df_features is None or df_features.empty:
        raise ValueError("Feature dataframe is empty.")

    if not expected_features:
        raise ValueError("Expected feature schema is empty.")

    # Compatibility is deliberately restricted to the known capitalization
    # change above.  Missing unrelated columns still reject the artifact.
    compatibility_aliases = {}
    for lowercase_name in _LEGACY_CAPITALIZED_INDICATORS:
        legacy_name = lowercase_name.capitalize()
        if (
            legacy_name in expected_features
            and legacy_name not in df_features.columns
            and lowercase_name in df_features.columns
        ):
            compatibility_aliases[legacy_name] = df_features[lowercase_name]

    if compatibility_aliases:
        df_features = df_features.assign(**compatibility_aliases)

    legacy_clock_features = {
        "hour_sin",
        "hour_cos",
        "minute_sin",
        "minute_cos",
    }
    required_clock_features = legacy_clock_features.intersection(expected_features)
    missing_clock_features = required_clock_features.difference(df_features.columns)
    if missing_clock_features:
        if not isinstance(df_features.index, pd.DatetimeIndex):
            raise ValueError(
                "Legacy clock features require a DatetimeIndex. "
                f"Missing={sorted(missing_clock_features)}"
            )

        hour = df_features.index.hour.astype(float)
        minute = df_features.index.minute.astype(float)
        clock_aliases = {
            "hour_sin": np.sin(2 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0),
            "minute_sin": np.sin(2 * np.pi * minute / 60.0),
            "minute_cos": np.cos(2 * np.pi * minute / 60.0),
        }
        df_features = df_features.assign(
            **{
                name: clock_aliases[name]
                for name in missing_clock_features
            }
        )

    actual = list(df_features.columns)
    missing = [c for c in expected_features if c not in actual]
    extra = [c for c in actual if c not in expected_features]

    if missing:
        raise ValueError(
            "Feature schema mismatch. "
            f"Missing={missing[:10]} "
            f"Expected_count={len(expected_features)} Actual_count={len(actual)}"
        )

    if extra and not allow_extra:
        raise ValueError(
            "Unexpected extra features present. "
            f"Extra={extra[:10]} Expected_count={len(expected_features)} Actual_count={len(actual)}"
        )

    aligned = df_features.loc[:, expected_features].copy()
    # Keep schema columns even when an extended-hours sample has no usable
    # values for them. Training explicitly permits these VWAP features to be
    # missing, and XGBoost handles NaNs through its learned missing-value path.
    aligned = _clean_feature_dataframe(aligned, drop_all_nan=False)

    unnamed_cols = [
        col
        for col in aligned.columns
        if col == "" or pd.isna(col) or str(col).strip() == ""
    ]
    if unnamed_cols:
        raise ValueError(
            f"Unnamed columns present in inference features: {unnamed_cols[:10]}"
        )

    non_numeric = aligned.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(f"Non-numeric features present: {non_numeric[:10]}")

    if aligned.isna().any().any():
        optional_nan_features = {
            "vwap_distance",
            "vwap_slope_2",
            "vwap_slope_4",
            "relative_vwap_strength",
        }
        nan_cols = aligned.columns[aligned.isna().any()].tolist()
        unexpected_nan_cols = [
            column for column in nan_cols if column not in optional_nan_features
        ]
        if unexpected_nan_cols:
            raise ValueError(
                "NaNs present in inference features: "
                f"{unexpected_nan_cols[:10]}"
            )

    return aligned


def _time_ordered_train_cal_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_frac: float = 0.60,
    cal_frac: float = 0.20,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")

    n = len(X)
    if n < 100:
        raise ValueError(
            "At least 100 chronological rows are required for independent "
            f"train/calibration/test windows; received n={n}."
        )

    # Calibration and final test data must never overlap.
    train_end = int(n * train_frac)
    cal_end = int(n * (train_frac + cal_frac))

    if train_end <= 0 or cal_end <= train_end or cal_end >= n:
        raise ValueError(f"Invalid chronological split boundaries for n={n}")

    X_train = X.iloc[:train_end].copy()
    y_train = y.iloc[:train_end].copy()
    X_cal = X.iloc[train_end:cal_end].copy()
    y_cal = y.iloc[train_end:cal_end].copy()
    X_test = X.iloc[cal_end:].copy()
    y_test = y.iloc[cal_end:].copy()

    return X_train, y_train, X_cal, y_cal, X_test, y_test


def _chronological_class_counts(
    y: pd.Series,
    train_frac: float = 0.60,
    cal_frac: float = 0.20,
) -> Dict[str, Dict[int, int]]:
    """
    Return chronological class counts for train/cal/test windows without
    requiring a full X matrix. Useful for pre-flight diagnostics before
    walk-forward and split validation.
    """
    y = pd.Series(y).dropna().astype(int)
    n = len(y)
    if n <= 0:
        return {
            "train": {},
            "cal": {},
            "test": {},
            "n": 0,
        }

    train_end = int(n * train_frac)
    cal_end = int(n * (train_frac + cal_frac))

    y_train = y.iloc[:train_end]
    y_cal = y.iloc[train_end:cal_end]
    y_test = y.iloc[cal_end:]

    return {
        "train": y_train.value_counts().sort_index().astype(int).to_dict(),
        "cal": y_cal.value_counts().sort_index().astype(int).to_dict(),
        "test": y_test.value_counts().sort_index().astype(int).to_dict(),
        "n": int(n),
    }


def _print_chronological_class_counts(
    label: str,
    y: pd.Series,
    train_frac: float = 0.60,
    cal_frac: float = 0.20,
) -> Dict[str, Dict[int, int]]:
    counts = _chronological_class_counts(
        y,
        train_frac=train_frac,
        cal_frac=cal_frac,
    )
    print(
        f"[CLASS TIMELINE] {label}: "
        f"n={counts['n']} "
        f"train={counts['train']} "
        f"cal={counts['cal']} "
        f"test={counts['test']}"
    )
    return counts


def _build_thresholded_binary_target(
    df_feat: pd.DataFrame,
    mode: str,
    use_two_stage: bool = False,
    min_move_override: Optional[float] = None,
) -> pd.DataFrame:
    """
    Create the binary training target and evaluation target.

    Training target:
      1 = forward return above +min_move
      0 = forward return below -min_move
      NaN = neutral/no-trade observation

    Evaluation target:
      1 = non-negative movement target
      0 = otherwise, including neutral observations
    """
    df_feat = df_feat.copy()

    if "Close" not in df_feat.columns:
        raise ValueError("Close column is required for target creation")

    horizon_bars = 1 if mode == "daily" else (4 if use_two_stage else 1)

    close = pd.to_numeric(df_feat["Close"], errors="coerce")
    next_ret = close.shift(-horizon_bars) / close - 1.0
    if mode != "daily" and isinstance(df_feat.index, pd.DatetimeIndex):
        session_keys = intraday_session_keys(df_feat.index)
        same_session = session_keys.eq(session_keys.shift(-horizon_bars))
        next_ret = next_ret.where(same_session)

    default_min_move = 0.002 if mode == "daily" or use_two_stage else 0.0008
    min_move = (
        float(min_move_override) if min_move_override is not None else default_min_move
    )

    df_feat["forward_return"] = next_ret

    valid_return = next_ret.notna()

    df_feat["evaluation_target"] = np.where(
        valid_return,
        (next_ret > 0.0).astype(int),
        np.nan,
    )

    df_feat["movement_target"] = np.where(
        valid_return,
        (next_ret.abs() >= min_move).astype(int),
        np.nan,
    )

    df_feat["target"] = np.where(
        next_ret > min_move,
        1,
        np.where(next_ret < -min_move, 0, np.nan),
    )

    valid_count = int(valid_return.sum())
    neutral_count = int((valid_return & (next_ret.abs() <= min_move)).sum())
    labeled_count = int(df_feat["target"].notna().sum())

    print(
        f"[TARGET DIAGNOSTICS] mode={mode} "
        f"horizon_bars={horizon_bars} "
        f"min_move={min_move:.5f} "
        f"eligible={valid_count} "
        f"labeled={labeled_count} "
        f"neutral={neutral_count} "
        f"neutral_rate="
        f"{neutral_count / valid_count:.2%}"
        if valid_count
        else f"[TARGET DIAGNOSTICS] mode={mode} "
        f"horizon_bars={horizon_bars} "
        f"min_move={min_move:.5f} "
        f"eligible=0 labeled=0 neutral=0 neutral_rate=0.00%"
    )

    return df_feat


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Proper OHLCV resample: Open=first, High=max, Low=min, Close=last, Volume=sum"""
    df = df.copy()
    required = ["Open", "High", "Low", "Close", "Volume"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing {c} for resample")

    o = df["Open"].resample(rule).first()
    h = df["High"].resample(rule).max()
    l = df["Low"].resample(rule).min()
    c = df["Close"].resample(rule).last()
    v = df["Volume"].resample(rule).sum(min_count=1)

    out = pd.concat([o, h, l, c, v], axis=1)
    out.columns = ["Open", "High", "Low", "Close", "Volume"]
    out = out.dropna()
    return out


def _add_intraday_regime_cols(df_feat: pd.DataFrame) -> pd.DataFrame:
    """Add session-aware regime measures used by model training."""
    df_feat = df_feat.copy()
    close = df_feat["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce")
    if isinstance(df_feat.index, pd.DatetimeIndex):
        session_keys = intraday_session_keys(df_feat.index)
        session_types = intraday_session_types(df_feat.index)
        returns = close.pct_change().where(session_keys.eq(session_keys.shift(1)))
        mom_4 = close.pct_change(4).where(session_keys.eq(session_keys.shift(4)))
        grouped_returns = returns.groupby(session_keys)
        regular_vol = grouped_returns.transform(
            lambda values: values.rolling(12, min_periods=12).std()
        )
        extended_vol = grouped_returns.transform(
            lambda values: values.rolling(4, min_periods=2).std()
        )
        df_feat["vol_12"] = regular_vol
        df_feat["vol_extended"] = extended_vol.where(
            session_types.isin(["premarket", "after_hours"])
        )
        df_feat["vol_regime"] = regular_vol.where(
            session_types.eq("regular"), df_feat["vol_extended"]
        ).where(session_types.isin(["regular", "premarket", "after_hours"]))
    else:
        returns = close.pct_change()
        mom_4 = close.pct_change(4)
        df_feat["vol_12"] = returns.rolling(12, min_periods=12).std()
        df_feat["vol_extended"] = np.nan
        df_feat["vol_regime"] = df_feat["vol_12"]
    df_feat["mom_4_abs"] = mom_4.abs()
    return df_feat


def _latest_extended_session_volatility(frame: Optional[pd.DataFrame]) -> Optional[float]:
    """Calculate four-bar volatility from fresh 15-minute bars in this session."""
    if frame is None or frame.empty or "Close" not in frame.columns:
        return None
    close = frame["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    timestamps = pd.to_datetime(frame.index, utc=True, errors="coerce")
    close = pd.Series(
        pd.to_numeric(close, errors="coerce").to_numpy(), index=timestamps
    ).dropna()
    close = close.loc[~close.index.isna()].sort_index()
    close = close[~close.index.duplicated(keep="last")]
    if len(close) < 3:
        return None

    age_minutes = (
        pd.Timestamp.now(tz="UTC") - close.index[-1]
    ).total_seconds() / 60.0
    if age_minutes < -5.0 or age_minutes > float(EXTENDED_HOURS_MAX_INTRADAY_BAR_AGE_MINUTES):
        return None

    session_types = intraday_session_types(close.index)
    session_keys = intraday_session_keys(close.index)
    latest_session_type = session_types.iloc[-1]
    if latest_session_type not in ("premarket", "after_hours"):
        return None
    close = close.loc[session_keys.eq(session_keys.iloc[-1]).to_numpy()]
    close = close.resample("15min").last().dropna()
    returns = close.pct_change()
    if returns.notna().sum() < 2:
        return None
    value = returns.rolling(4, min_periods=2).std().iloc[-1]
    return float(value) if pd.notna(value) and np.isfinite(value) else None


def _filter_intraday_rows_by_mode(
    df_feat: pd.DataFrame,
    mode: str,
    train_frac: float = 0.70,
    percentile: float = 0.70,
    eligible_mask: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Fit regime thresholds on the earliest training portion only, then apply
    the identical rule used by live model routing.
    """
    if mode not in ("intraday_mom", "intraday_mr"):
        return df_feat.copy(), {}

    with_regime = _add_intraday_regime_cols(df_feat)
    eligible_count = len(with_regime)
    if eligible_mask is not None:
        eligible_mask = np.asarray(eligible_mask, dtype=bool)
        if len(eligible_mask) != len(with_regime):
            raise ValueError(
                "Intraday target-eligibility mask length does not match feature rows"
            )
        eligible_count = int(eligible_mask.sum())
        with_regime = with_regime.iloc[eligible_mask]
    with_regime = with_regime.replace([np.inf, -np.inf], np.nan)
    with_regime = with_regime.dropna(subset=["mom_4_abs", "vol_regime"])
    if with_regime.empty:
        raise ValueError(
            f"No valid target-eligible regime rows for mode={mode} "
            f"(target_eligible={eligible_count})"
        )
    _log_intraday_session_coverage(
        "REGIME", mode, "valid_regime_metrics", with_regime.index
    )

    if isinstance(with_regime.index, pd.DatetimeIndex):
        session_types = intraday_session_types(with_regime.index)
        extended_mask = session_types.isin(["premarket", "after_hours"])
        regular_threshold_rows = with_regime.loc[
            ~session_types.eq("after_hours")
        ].dropna(subset=["mom_4_abs", "vol_12"])
        fit_end = int(len(regular_threshold_rows) * train_frac)
        if fit_end < 20:
            raise ValueError(
                f"Insufficient pre-split rows to fit regular thresholds: {fit_end}"
            )
        regular_fit = regular_threshold_rows.iloc[:fit_end]
        fit_cutoff = regular_fit.index[-1]
        extended_fit = with_regime.loc[
            extended_mask & (with_regime.index <= fit_cutoff)
        ].dropna(subset=["mom_4_abs", "vol_extended"])
    else:
        session_types = pd.Series("regular", index=with_regime.index)
        extended_mask = pd.Series(False, index=with_regime.index)
        regular_threshold_rows = with_regime.dropna(
            subset=["mom_4_abs", "vol_12"]
        )
        fit_end = int(len(regular_threshold_rows) * train_frac)
        if fit_end < 20:
            raise ValueError(
                f"Insufficient pre-split rows to fit regular thresholds: {fit_end}"
            )
        regular_fit = regular_threshold_rows.iloc[:fit_end]
        fit_cutoff = regular_fit.index[-1]
        extended_fit = with_regime.iloc[0:0]

    mom_threshold = float(regular_fit["mom_4_abs"].quantile(percentile))
    vol_threshold = float(regular_fit["vol_12"].quantile(percentile))
    if len(extended_fit) >= 20:
        extended_mom_threshold = float(
            extended_fit["mom_4_abs"].quantile(percentile)
        )
        extended_vol_threshold = float(
            extended_fit["vol_extended"].quantile(percentile)
        )
    else:
        raise ValueError(
            "Insufficient extended-session rows to fit regime thresholds: "
            f"{len(extended_fit)}"
        )
    if not all(np.isfinite(value) for value in (
        mom_threshold, vol_threshold, extended_mom_threshold, extended_vol_threshold
    )):
        raise ValueError("Regime thresholds are not finite")

    row_mom_threshold = pd.Series(
        np.where(extended_mask, extended_mom_threshold, mom_threshold),
        index=with_regime.index,
    )
    row_vol_threshold = pd.Series(
        np.where(extended_mask, extended_vol_threshold, vol_threshold),
        index=with_regime.index,
    )
    momentum_mask = (
        with_regime["mom_4_abs"] >= row_mom_threshold
    ) | (with_regime["vol_regime"] >= row_vol_threshold)
    selected_mask = momentum_mask if mode == "intraday_mom" else ~momentum_mask
    filtered = with_regime.loc[selected_mask].copy()

    regime_config = {
        "version": 2,
        "momentum_bars": 4,
        "volatility_bars": 12,
        "extended_volatility_bars": 4,
        "extended_volatility_min_periods": 2,
        "percentile": float(percentile),
        "momentum_threshold": mom_threshold,
        "volatility_threshold": vol_threshold,
        "extended_momentum_threshold": extended_mom_threshold,
        "extended_volatility_threshold": extended_vol_threshold,
        "fit_rows": fit_end,
        "fit_end": str(fit_cutoff),
        "rule": "session_specific_volatility; mom_when_abs_momentum_or_volatility_meets_threshold",
    }
    print(
        f"[REGIME] {mode}: {len(with_regime)} -> {len(filtered)} rows "
        f"| target-eligible={eligible_count} "
        f"| regular mom={mom_threshold:.6f} vol12={vol_threshold:.6f} "
        f"| extended mom={extended_mom_threshold:.6f} "
        f"vol4={extended_vol_threshold:.6f} min_periods=2"
    )
    return filtered, regime_config


def _intraday_target_eligibility_mask(
    df_feat: pd.DataFrame,
    horizon_bars: int,
) -> np.ndarray:
    """Select rows whose forward horizon stays within one intraday session."""
    close = df_feat["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce")
    eligible = (
        close.notna() & close.shift(-horizon_bars).notna()
    ).to_numpy(dtype=bool)

    if isinstance(df_feat.index, pd.DatetimeIndex):
        session_values = intraday_session_keys(df_feat.index).to_numpy()
        same_session = np.zeros(len(session_values), dtype=bool)
        if horizon_bars < len(session_values):
            same_session[:-horizon_bars] = (
                session_values[:-horizon_bars] == session_values[horizon_bars:]
            )
        eligible &= same_session

    return eligible


def _log_intraday_session_coverage(
    symbol: str,
    mode: str,
    stage: str,
    index: pd.Index,
) -> None:
    """Log New York session counts for rows surviving a training stage."""
    if not isinstance(index, pd.DatetimeIndex):
        print(
            f"[SESSION COVERAGE] {symbol}/{mode} stage={stage}: "
            f"rows={len(index)} index_type={type(index).__name__}"
        )
        return

    session_types = intraday_session_types(index)
    counts = session_types.value_counts()
    print(
        f"[SESSION COVERAGE] {symbol}/{mode} stage={stage}: "
        f"rows={len(index)} tz={index.tz} "
        f"premarket={int(counts.get('premarket', 0))} "
        f"regular={int(counts.get('regular', 0))} "
        f"after_hours={int(counts.get('after_hours', 0))} "
        f"closed={int(counts.get('closed', 0))}"
    )


def _artifact_age_days(artifact: dict):
    if not artifact:
        return None

    trained_at = artifact.get("trained_at")
    if not trained_at:
        return None

    try:
        dt = datetime.fromisoformat(str(trained_at).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return (now - dt).total_seconds() / 86400.0
    except Exception as e:
        print(f"[WARN] Could not parse trained_at={trained_at!r}: {e}")
        return None


def _artifact_is_fresh(artifact: dict, max_age_days: int = 30) -> bool:
    age = _artifact_age_days(artifact)
    if age is None:
        return False
    return age <= float(max_age_days)


def _order_intraday_model_candidates(preferred_order, fresh_models, stale_models):
    """Keep regime preference ahead of cross-regime artifact freshness."""
    fresh_by_name = {name: (obj, age) for name, obj, age in fresh_models}
    stale_by_name = {name: (obj, age) for name, obj, age in stale_models}
    candidates = []

    for name in preferred_order:
        if name in fresh_by_name:
            obj, age = fresh_by_name[name]
            candidates.append((name, obj, age, False))
        elif name in stale_by_name:
            obj, age = stale_by_name[name]
            candidates.append((name, obj, age, True))

    return candidates


# ---------------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------------
def train_model(
    df: pd.DataFrame,
    symbol: str,
    mode: str = "daily",
    use_multiclass: bool = False,
    use_two_stage: Optional[bool] = None,
    benchmark_df: Optional[pd.DataFrame] = None,
):
    """
    Train XGB model with chronological train/calibration/test split.
    Calibration is performed only on the calibration window, never on the final test window.
    """
    from class_balancing import analyze_class_balance, calculate_scale_pos_weight
    from target_engineering import create_multiclass_target, print_target_distribution
    from predictive_model.feature_selection import select_features_with_shap

    if use_two_stage is None:
        use_two_stage = bool(USE_TWO_STAGE_TARGETS)

    spy_intraday_mr_force_two_stage = (
        str(symbol).upper() == "SPY" and mode == "intraday_mr"
    )
    if spy_intraday_mr_force_two_stage and not use_multiclass:
        use_two_stage = True
        print(
            "[TARGET CONFIG] SPY/intraday_mr forcing two-stage target "
            "(4-bar / 60min horizon) for denser MR labels"
        )

    if use_multiclass and use_two_stage:
        raise ValueError("multiclass and two-stage targets cannot be enabled together")

    df = df.copy()
    regime_config = None
    binary_evaluation_frame = None
    extended_training_rows = {"premarket": 0, "after_hours": 0}
    if mode != "daily" and isinstance(df.index, pd.DatetimeIndex):
        session_types = intraday_session_types(df.index)
        extended_training_rows = {
            "premarket": int(session_types.eq("premarket").sum()),
            "after_hours": int(session_types.eq("after_hours").sum()),
        }
    extended_hours_trained = all(
        count >= 40 for count in extended_training_rows.values()
    )
    if mode != "daily":
        print(
            f"[EXTENDED-HOURS TRAINING COVERAGE] {symbol}/{mode}: "
            f"premarket={extended_training_rows['premarket']} "
            f"after_hours={extended_training_rows['after_hours']} "
            f"ready={extended_hours_trained}"
        )
        _log_intraday_session_coverage(symbol, mode, "raw_input", df.index)

    print(f"\n[FEATURES] Building engineered features for {symbol}/{mode}...")
    regime_selected_index = None
    if mode == "daily":
        df_feat = build_daily_features(df)
    elif mode in ("intraday", "intraday_mr", "intraday_mom"):
        if benchmark_df is None:
            benchmark_df = (
                df
                if symbol.upper() == "SPY"
                else fetch_historical_data("SPY", period="60d", interval="15m")
            )
        df_feat = build_intraday_features(
            df,
            benchmark_df=benchmark_df,
            diagnostic_label=f"{symbol}/{mode}",
        )
        _log_intraday_session_coverage(
            symbol, mode, "engineered_features", df_feat.index
        )
        if mode in ("intraday_mr", "intraday_mom"):
            regime_target_horizon = (
                1 if use_multiclass else (4 if use_two_stage else 1)
            )
            target_eligible_mask = _intraday_target_eligibility_mask(
                df_feat,
                horizon_bars=regime_target_horizon,
            )
            _log_intraday_session_coverage(
                symbol,
                mode,
                "target_horizon_eligible",
                df_feat.index[target_eligible_mask],
            )
            regime_rows, regime_config = _filter_intraday_rows_by_mode(
                df_feat,
                mode=mode,
                eligible_mask=target_eligible_mask,
            )
            regime_selected_index = regime_rows.index
            _log_intraday_session_coverage(
                symbol, mode, "regime_selected", regime_selected_index
            )
            # Keep the full chronological feature frame until forward targets
            # are computed. Filtering first makes a four-bar horizon jump over
            # omitted regime rows and can invalidate every label.
            df_feat = _add_intraday_regime_cols(df_feat)
        for c in ["ret_12", "mom_12_abs", "mom_4_abs", "vol_12", "vol_extended", "vol_regime"]:
            if c in df_feat.columns:
                df_feat = df_feat.drop(columns=[c])
    else:
        raise ValueError(
            "mode must be 'daily' or one of 'intraday', 'intraday_mr', 'intraday_mom'"
        )

    print(
        f"[FEATURES] Built {len(df_feat.columns)} total columns after feature engineering"
    )

    print("\n[CLEANUP] Cleaning DataFrame columns...")
    df_feat = _clean_feature_dataframe(df_feat)
    print(f"[CLEANUP] Rows after clean: {len(df_feat)} | Cols: {len(df_feat.columns)}")

    if use_multiclass:
        print("\n[TARGET] Creating multi-class target...")
        thresholds = (
            (-0.015, -0.003, 0.003, 0.015)
            if mode == "daily"
            else (-0.008, -0.002, 0.002, 0.008)
        )
        df_feat = create_multiclass_target(
            df_feat, forward_periods=1, thresholds=thresholds
        )
        target_col = "Target_multiclass"
        if mode != "daily" and isinstance(df_feat.index, pd.DatetimeIndex):
            session_keys = intraday_session_keys(df_feat.index)
            same_session = session_keys.eq(session_keys.shift(-1))
            df_feat.loc[~same_session, "Target_multiclass"] = np.nan
        if regime_selected_index is not None:
            regime_mask = df_feat.index.isin(regime_selected_index)
            print(
                f"[REGIME TARGET] {symbol}/{mode}: "
                f"frame_rows={len(df_feat)} "
                f"regime_index_rows={len(regime_selected_index)} "
                f"index_overlap={int(regime_mask.sum())} "
                f"labeled_before={int(df_feat[target_col].notna().sum())} "
                f"labeled_after={int(df_feat.loc[regime_mask, target_col].notna().sum())}"
            )
            df_feat = df_feat.loc[regime_mask].copy()
        num_classes = 5
        objective = "multi:softprob"
        eval_metric = "mlogloss"
        print_target_distribution(df_feat, target_col)
    else:
        print("\n[TARGET] Creating thresholded binary target...")
        target_override = INTRADAY_TARGET_OVERRIDES.get((symbol.upper(), mode))

        min_move_override = None
        if target_override:
            min_move_override = target_override.get("min_move")

        if spy_intraday_mr_force_two_stage:
            print(
                f"[TARGET CONFIG] {symbol}/{mode} "
                f"force_two_stage=True horizon_bars=4 target_horizon=60min"
            )

        if target_override:
            print(
                f"[TARGET CONFIG] {symbol}/{mode} "
                f"min_move_override={min_move_override}"
            )

        df_feat = _build_thresholded_binary_target(
            df_feat,
            mode=mode,
            use_two_stage=use_two_stage,
            min_move_override=min_move_override,
        )
        target_col = "target"
        if regime_selected_index is not None:
            regime_mask = df_feat.index.isin(regime_selected_index)
            selected_regime_rows = df_feat.loc[regime_mask]
            selected_forward_valid = selected_regime_rows["forward_return"].notna()
            selected_target = selected_regime_rows[target_col]
            print(
                f"[REGIME TARGET] {symbol}/{mode}: "
                f"frame_rows={len(df_feat)} "
                f"regime_index_rows={len(regime_selected_index)} "
                f"index_overlap={int(regime_mask.sum())} "
                f"labeled_before={int(df_feat[target_col].notna().sum())} "
                f"labeled_after={int(selected_target.notna().sum())}"
            )
            print(
                f"[REGIME TARGET DETAIL] {symbol}/{mode}: "
                f"selected_rows={len(selected_regime_rows)} "
                f"valid_forward_returns={int(selected_forward_valid.sum())} "
                f"no_same_session_forward_return={int((~selected_forward_valid).sum())} "
                f"neutral={int((selected_forward_valid & selected_target.isna()).sum())} "
                f"class_0={int(selected_target.eq(0).sum())} "
                f"class_1={int(selected_target.eq(1).sum())}"
            )
            df_feat = df_feat.loc[regime_mask].copy()
        binary_evaluation_frame = df_feat.copy()
        num_classes = 2
        objective = "binary:logistic"
        eval_metric = "logloss"

    if mode in ("intraday_mr", "intraday_mom"):
        labeled_rows = df_feat.loc[df_feat[target_col].notna()]
        _log_intraday_session_coverage(
            symbol, mode, "regime_labeled_targets", labeled_rows.index
        )
        missing_counts = labeled_rows.isna().sum()
        missing_counts = missing_counts[missing_counts > 0].sort_values(
            ascending=False
        )
        complete_labeled_rows = int(labeled_rows.notna().all(axis=1).sum())
        print(
            f"[ROW VALIDATION] {symbol}/{mode}: "
            f"labeled_rows={len(labeled_rows)} "
            f"complete_labeled_rows={complete_labeled_rows} "
            f"rows_removed_by_dropna={len(labeled_rows) - complete_labeled_rows}"
        )
        if not missing_counts.empty:
            missing_summary = ", ".join(
                f"{column}={int(count)}/{len(labeled_rows)}"
                for column, count in missing_counts.items()
            )
            print(
                f"[ROW VALIDATION] NaNs among labeled {symbol}/{mode} rows: "
                f"{missing_summary}"
            )

    # XGBoost supports missing feature values natively. Keep labeled intraday
    # rows with unavailable session-specific indicators (for example, VWAP
    # when extended-hours volume is zero) instead of excluding the whole row.
    # Daily training retains its existing complete-row requirement.
    if mode in ("intraday", "intraday_mr", "intraday_mom"):
        df_feat = df_feat.dropna(subset=[target_col]).copy()
    else:
        df_feat = df_feat.dropna().copy()
    if df_feat.empty:
        raise ValueError(f"No rows left after feature engineering for {symbol}/{mode}")
    if mode in ("intraday_mr", "intraday_mom"):
        _log_intraday_session_coverage(
            symbol, mode, "complete_labeled_rows", df_feat.index
        )

    class_counts = df_feat[target_col].value_counts()
    unique_classes = sorted(class_counts.index.tolist())
    print(
        f"\n[CLASS CHECK] Found {len(unique_classes)} unique classes: {unique_classes}"
    )
    for cls in unique_classes:
        count = int(class_counts[cls])
        print(f" Class {cls}: {count} samples ({count / len(df_feat) * 100:.1f}%)")

    min_samples_per_class = 3 if mode in ("intraday_mr", "intraday_mom") else 5
    valid_classes = [
        cls for cls in unique_classes if class_counts[cls] >= min_samples_per_class
    ]
    if len(valid_classes) < 2:
        raise ValueError(
            f"Insufficient class diversity for {symbol}/{mode}. "
            f"Class counts: {class_counts.to_dict()}"
        )

    if use_multiclass and len(valid_classes) < num_classes:
        print(
            f"\n[CLASS REMAP] Only {len(valid_classes)}/{num_classes} classes have enough samples"
        )
        df_feat = df_feat[df_feat[target_col].isin(valid_classes)].copy()
        class_mapping = {old: new for new, old in enumerate(sorted(valid_classes))}
        df_feat[target_col] = df_feat[target_col].map(class_mapping)
        num_classes = len(valid_classes)
        print(f"[CLASS REMAP] Mapping: {class_mapping}")

    print("\n[FEATURES] Preparing feature matrix...")
    exclude_cols = [
        target_col,
        "forward_return",
        "evaluation_target",
        "movement_target",
        "Close",
        "Open",
        "High",
        "Low",
        "Volume",
    ]
    feature_cols = [col for col in df_feat.columns if col not in exclude_cols]

    X = df_feat[feature_cols].copy()
    y = pd.Series(df_feat[target_col], dtype=int)

    unnamed_cols = [
        col for col in X.columns if col == "" or pd.isna(col) or str(col).strip() == ""
    ]
    if unnamed_cols:
        raise ValueError(f"Unnamed feature columns present: {unnamed_cols[:10]}")

    X = _clean_feature_dataframe(X)

    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(
            f"Non-numeric features present after cleaning: {non_numeric[:10]}"
        )

    all_nan_cols = X.columns[X.isna().all()].tolist()
    if all_nan_cols:
        print(f"[VALIDATION] Dropping {len(all_nan_cols)} all-NaN columns")
        X = X.drop(columns=all_nan_cols)

    if X.isna().any().any():
        nan_cols = X.columns[X.isna().any()].tolist()
        extended_session_optional = {
            "vwap_distance",
            "vwap_slope_2",
            "vwap_slope_4",
            "relative_vwap_strength",
        }
        unexpected_nan_cols = [
            column for column in nan_cols if column not in extended_session_optional
        ]
        if mode == "daily" or unexpected_nan_cols:
            columns = unexpected_nan_cols if unexpected_nan_cols else nan_cols
            raise ValueError(
                f"Unexpected NaNs remain in training matrix. Columns: {columns[:10]}"
            )
        missing_counts = X[nan_cols].isna().sum().to_dict()
        print(
            "[VALIDATION] XGBoost will handle missing extended-session features: "
            f"{missing_counts}"
        )

    feature_list = list(X.columns)
    if not feature_list:
        raise ValueError(f"No valid features remaining for {symbol}/{mode}")

    print(f"[FEATURES] Final feature count: {len(feature_list)}")
    print(f"[SANITY] Feature matrix shape: {X.shape} | Target shape: {y.shape}")

    full_feature_schema = {
        "features": feature_list,
        "feature_count": len(feature_list),
    }

    evaluation_X = None
    evaluation_training_target = None
    evaluation_target = None
    evaluation_returns = None
    evaluation_movement_target = None
    if num_classes == 2:
        evaluation_columns = list(X.columns)
        evaluation_X = _clean_feature_dataframe(
            binary_evaluation_frame.loc[:, evaluation_columns]
        )
        evaluation_bundle = pd.concat(
            [
                evaluation_X,
                pd.to_numeric(
                    binary_evaluation_frame["target"], errors="coerce"
                ).rename("training_target"),
                pd.to_numeric(
                    binary_evaluation_frame["evaluation_target"], errors="coerce"
                ).rename("evaluation_target"),
                pd.to_numeric(
                    binary_evaluation_frame["forward_return"], errors="coerce"
                ).rename("forward_return"),
                pd.to_numeric(
                    binary_evaluation_frame["movement_target"], errors="coerce"
                ).rename("movement_target"),
            ],
            axis=1,
        ).replace([np.inf, -np.inf], np.nan)
        # XGBoost accepts missing feature values. Require valid outcome labels
        # and realized returns, while retaining rows with unavailable features.
        evaluation_bundle = evaluation_bundle.dropna(
            subset=["evaluation_target", "forward_return", "movement_target"]
        )
        evaluation_X = evaluation_bundle.loc[:, evaluation_columns].copy()
        evaluation_training_target = evaluation_bundle["training_target"].copy()
        evaluation_target = evaluation_bundle["evaluation_target"].astype(int)
        evaluation_returns = evaluation_bundle["forward_return"].copy()
        evaluation_movement_target = evaluation_bundle["movement_target"].astype(int)
        if mode in ("intraday_mr", "intraday_mom"):
            _log_intraday_session_coverage(
                symbol, mode, "evaluation_universe", evaluation_X.index
            )

    X_train, y_train, X_cal, y_cal, X_test, y_test = _time_ordered_train_cal_test_split(
        X, y
    )
    if mode in ("intraday_mr", "intraday_mom"):
        _log_intraday_session_coverage(
            symbol, mode, "final_test_labeled_rows", X_test.index
        )

    print(f"\n[SPLIT] Train={len(X_train)} Cal={len(X_cal)} Test={len(X_test)}")
    print(f"[SPLIT] Train classes: {y_train.value_counts().to_dict()}")
    print(f"[SPLIT] Cal classes: {y_cal.value_counts().to_dict()}")
    print(f"[SPLIT] Test classes: {y_test.value_counts().to_dict()}")

    _print_chronological_class_counts(
        f"{symbol}/{mode} training_target",
        y,
    )

    if len(sorted(y_train.unique())) < 2:
        raise ValueError("Train set has fewer than 2 classes.")
    if num_classes == 2 and evaluation_target is not None:
        _print_chronological_class_counts(
            f"{symbol}/{mode} evaluation_target",
            evaluation_target,
        )

    if num_classes == 2 and evaluation_training_target is not None:
        eval_labeled = evaluation_training_target.dropna().astype(int)
        _print_chronological_class_counts(
            f"{symbol}/{mode} wf_labeled_training_target",
            eval_labeled,
        )

    print("\n[CLASS BALANCE] Analyzing train-set class distribution...")
    analyze_class_balance(y_train, name=f"{symbol}/{mode} Train Set")

    scale_pos_weight = None
    if num_classes == 2:
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        if n_pos > 0 and n_neg > 0:
            scale_pos_weight = calculate_scale_pos_weight(y_train)
            if 0.83 <= scale_pos_weight <= 1.2:
                print("[CLASS WEIGHT] Classes balanced, no weighting needed")
                scale_pos_weight = None
            else:
                print(
                    f"[CLASS WEIGHT] Applying scale_pos_weight={scale_pos_weight:.3f}"
                )
        else:
            print("[CLASS WEIGHT] Insufficient class samples for weighting")
    else:
        print("[CLASS WEIGHT] Multi-class mode - using default class weights")

    params = {
        "objective": objective,
        "num_class": num_classes if num_classes > 2 else None,
        "eval_metric": eval_metric,
        "tree_method": "hist",
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "max_depth": 6,
        "min_child_weight": 3,
        "gamma": 0.1,
        "reg_alpha": 0.05,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": MODEL_TRAIN_N_JOBS,
    }

    if mode == "daily":
        params.update({"n_estimators": 400, "learning_rate": 0.04})
    else:
        params.update({"n_estimators": 500, "learning_rate": 0.035})

    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight

    params = {k: v for k, v in params.items() if v is not None}
    model = XGBClassifier(**params)

    print(f"\n[TRAINING] Fitting {symbol}/{mode} model...")
    model.fit(X_train, y_train, eval_set=[(X_cal, y_cal)], verbose=False)
    print(f"[TRAINING] Model trained with {model.n_estimators} trees")

    selected_feature_metadata = None
    selected_features = feature_list
    walk_forward_evaluation = None
    extended_hours_oof_metrics = None
    walk_forward_threshold = None
    artifact_decision_threshold = None

    print(
        "[FEATURE SELECTION] Eligibility: "
        f"num_classes={num_classes}, "
        f"use_two_stage={use_two_stage}, "
        f"train_features={len(X_train.columns)}, "
        f"cal_rows={len(X_cal)}"
    )

    if num_classes == 2 and len(X_train.columns) >= 25 and len(X_cal) >= 50:
        try:
            print(f"\n[FEATURE SELECTION] Running calibration-window SHAP selection...")
            top_features, _shap_values, X_train_sel, X_cal_sel, fs_meta = (
                select_features_with_shap(
                    model=model,
                    X_train=X_train,
                    X_cal=X_cal,
                    top_n=None,
                    plot=False,
                    symbol=symbol,
                    mode=mode,
                )
            )
            if len(top_features) >= 10:
                print(
                    f"[FEATURE SELECTION] Retraining on {len(top_features)} selected features"
                )
                X_test = X_test[top_features].copy()
                X_train = X_train_sel
                X_cal = X_cal_sel
                selected_features = top_features
                selected_feature_metadata = fs_meta
                feature_list = top_features

                if len(feature_list) != len(set(feature_list)):
                    raise ValueError("Selected feature list contains duplicates.")

                missing_selected = [
                    col for col in feature_list if col not in X_train.columns
                ]
                if missing_selected:
                    raise ValueError(
                        f"Selected features missing from training schema: {missing_selected[:10]}"
                    )

                print(
                    f"[FEATURE SELECTION] Final schema: "
                    f"{len(feature_list)} features | "
                    f"X_train={X_train.shape} | "
                    f"X_cal={X_cal.shape} | "
                    f"X_test={X_test.shape}"
                )

                model = XGBClassifier(**params)
                model.fit(X_train, y_train, eval_set=[(X_cal, y_cal)], verbose=False)
                print(f"[FEATURE SELECTION] Retrained selected-feature model")
            else:
                print(
                    f"[FEATURE SELECTION] Too few features selected; keeping full schema"
                )
        except Exception as e:
            print(
                f"[FEATURE SELECTION] Skipping SHAP feature selection due to error: {type(e).__name__}: {e}"
            )

    if num_classes == 2:
        print("\n[WALK-FORWARD] Running expanding-fold evaluation with embargo gaps...")
        # Walk-forward receives the complete evaluation schema.
        # Each fold performs feature selection using only that fold's training window,
        # capped at the final model's selected-feature count.
        walk_forward_bundle = evaluation_bundle.loc[
            evaluation_bundle.index < X_test.index.min()
        ].sort_index()
        # Forward labels near the final holdout boundary can include prices from
        # inside that holdout. Remove a full target horizon before creating OOF
        # predictions so the extended-hours gate remains independent of holdout.
        oof_label_horizon = 4 if use_two_stage and mode != "daily" else 1
        if len(walk_forward_bundle) > oof_label_horizon:
            walk_forward_bundle = walk_forward_bundle.iloc[:-oof_label_horizon]
        else:
            walk_forward_bundle = walk_forward_bundle.iloc[0:0]
        walk_forward_X = walk_forward_bundle.loc[:, evaluation_columns].copy()
        walk_forward_target = walk_forward_bundle["evaluation_target"].astype(int)
        walk_forward_training_target = walk_forward_bundle["training_target"].copy()
        walk_forward_returns = walk_forward_bundle["forward_return"].copy()
        walk_forward_movement_target = walk_forward_bundle["movement_target"].astype(int)

        eval_counts = _print_chronological_class_counts(
            f"{symbol}/{mode} walk_forward_eval_target",
            walk_forward_target,
        )

        min_eval_classes = min(
            len(eval_counts["train"]),
            len(eval_counts["cal"]),
            len(eval_counts["test"]),
        )
        if min_eval_classes < 2:
            raise ValueError(
                f"walk-forward precheck failed for {symbol}/{mode}: "
                f"evaluation target lacks class diversity across chronological windows "
                f"(train={eval_counts['train']}, cal={eval_counts['cal']}, test={eval_counts['test']})"
            )

        evaluation_feature_columns = list(feature_list)

        evaluation_X_for_walk_forward = walk_forward_X.loc[
            :, evaluation_feature_columns
        ].copy()

        print(
            f"[WALK-FORWARD SCHEMA] "
            f"evaluation_features={walk_forward_X.shape[1]} "
            f"max_features={len(feature_list)} "
            f"selected_features={len(selected_features)}"
        )

        walk_forward_evaluation = evaluate_walk_forward(
            X=evaluation_X_for_walk_forward,
            y=walk_forward_target,
            forward_returns=walk_forward_returns,
            training_target=walk_forward_training_target,
            model_params=params,
            mode=mode,
            n_splits=MODEL_EVAL_WALK_FORWARD_FOLDS,
            gap_bars=MODEL_EVAL_GAP_BARS,
            cost_bps=MODEL_EVAL_TRANSACTION_COST_BPS,
            max_features=len(feature_list),
            movement_target=walk_forward_movement_target if use_two_stage else None,
            holding_period_bars=4 if use_two_stage and mode != "daily" else 1,
        )
        if mode in ("intraday_mr", "intraday_mom"):
            oof_frame = pd.DataFrame(walk_forward_evaluation.get("predictions", []))
            if not oof_frame.empty:
                oof_frame["timestamp"] = pd.to_datetime(
                    oof_frame["timestamp"], errors="coerce", utc=True
                )
                oof_frame["actual"] = pd.to_numeric(
                    oof_frame["actual"], errors="coerce"
                )
                oof_frame["calibration_probability"] = pd.to_numeric(
                    oof_frame["calibration_probability"], errors="coerce"
                )
                oof_frame = oof_frame.dropna(
                    subset=["timestamp", "actual", "calibration_probability"]
                )
                oof_index = pd.DatetimeIndex(oof_frame["timestamp"])
                oof_session_types = intraday_session_types(oof_index)
                extended_hours_oof_metrics = {}
                for session_name in ("premarket", "after_hours"):
                    session_mask = oof_session_types.eq(session_name).to_numpy()
                    session_rows = oof_frame.loc[session_mask]
                    session_labels = session_rows["actual"].astype(int).to_numpy()
                    session_probabilities = session_rows[
                        "calibration_probability"
                    ].to_numpy(dtype=float)
                    session_brier = (
                        float(brier_score_loss(session_labels, session_probabilities))
                        if len(session_rows)
                        and len(np.unique(session_labels)) >= 2
                        else None
                    )
                    extended_hours_oof_metrics[session_name] = {
                        "samples": int(len(session_rows)),
                        "brier_score": session_brier,
                    }
                print(
                    f"[EXTENDED-HOURS OOF] {symbol}/{mode}: "
                    f"premarket={extended_hours_oof_metrics['premarket']} "
                    f"after_hours={extended_hours_oof_metrics['after_hours']}"
                )
            else:
                oof_index = pd.DatetimeIndex([], tz="UTC")
                extended_hours_oof_metrics = {
                    "premarket": {"samples": 0, "brier_score": None},
                    "after_hours": {"samples": 0, "brier_score": None},
                }
            _log_intraday_session_coverage(
                symbol, mode, "walk_forward_oof_predictions", oof_index
            )
        wf_metrics = walk_forward_evaluation["aggregate"]
        walk_forward_threshold = float(
            walk_forward_evaluation.get("decision_threshold", 0.5)
        )

        print(
            "[WALK-FORWARD] "
            f"folds={walk_forward_evaluation['n_splits']} "
            f"trades={wf_metrics['trade_count']} "
            f"threshold={walk_forward_evaluation['decision_threshold']:.3f} "
            f"net_return={wf_metrics['net_return']:.2%} "
            f"profit_factor={wf_metrics['profit_factor']:.3f} "
            f"max_drawdown={wf_metrics['max_drawdown']:.2%} "
            f"brier={wf_metrics.get('brier_score')}"
        )

        print(
            "[WALK-FORWARD DIAGNOSTICS] "
            f"{symbol}/{mode} "
            f"evaluation_rows={len(walk_forward_X)} "
            f"evaluation_positive_rate={float(np.mean(walk_forward_target)):.2%} "
            f"wf_threshold={walk_forward_evaluation.get('decision_threshold')} "
            f"wf_trade_count={wf_metrics.get('trade_count')}"
        )

    base_model = model
    final_model = model
    calibrated = False

    calibration_method = "sigmoid" if num_classes == 2 else "isotonic"
    print(
        f"\n[CALIBRATION] Applying {calibration_method} calibration on calibration window only"
    )
    try:
        calibrated_model = CalibratedClassifierCV(
            model,
            method=calibration_method,
            cv="prefit",
            n_jobs=MODEL_TRAIN_N_JOBS,
        )
        calibrated_model.fit(X_cal, y_cal)
        final_model = calibrated_model
        calibrated = True
        print(f"[CALIBRATION] Successfully calibrated using {calibration_method}")
    except Exception as e:
        print(f"[CALIBRATION] Failed to calibrate: {e}")
        print("[CALIBRATION] Using uncalibrated model as fallback")
        final_model = model

    movement_model = None
    movement_threshold = None
    movement_calibrated = False
    if use_two_stage and num_classes == 2:
        print("\n[TWO-STAGE] Training meaningful-move gate...")
        movement_X = evaluation_X.loc[:, feature_list].copy()
        print(
            f"[TWO-STAGE SCHEMA] "
            f"movement_X={movement_X.shape} "
            f"feature_list={len(feature_list)}"
        )
        movement_y = evaluation_movement_target.loc[movement_X.index].astype(int)
        (
            move_X_train,
            move_y_train,
            move_X_cal,
            move_y_cal,
            _move_X_test,
            _move_y_test,
        ) = _time_ordered_train_cal_test_split(movement_X, movement_y)
        if move_y_train.nunique() < 2 or move_y_cal.nunique() < 2:
            raise ValueError(
                "Two-stage movement model requires two train/calibration classes"
            )
        movement_params = dict(params)
        movement_params.pop("scale_pos_weight", None)
        move_neg = int((move_y_train == 0).sum())
        move_pos = int((move_y_train == 1).sum())
        if move_neg and move_pos:
            move_weight = move_neg / move_pos
            if not 0.83 <= move_weight <= 1.2:
                movement_params["scale_pos_weight"] = move_weight
        movement_base = XGBClassifier(**movement_params)
        movement_base.fit(
            move_X_train,
            move_y_train,
            eval_set=[(move_X_cal, move_y_cal)],
            verbose=False,
        )
        movement_model = CalibratedClassifierCV(
            movement_base,
            method="sigmoid",
            cv="prefit",
            n_jobs=MODEL_TRAIN_N_JOBS,
        )
        movement_model.fit(move_X_cal, move_y_cal)
        movement_cal_probability = movement_model.predict_proba(move_X_cal)[:, 1]
        movement_threshold = float(
            optimize_binary_decision_threshold(
                move_y_cal,
                movement_cal_probability,
                metric="f1",
                min_threshold=0.35,
                max_threshold=0.65,
                step=0.01,
            )["best_threshold"]
        )

        print(
            f"[TWO-STAGE THRESHOLD OPTIMIZATION] {symbol}/{mode} "
            f"cal_min={movement_cal_probability.min():.4f} "
            f"cal_median={np.median(movement_cal_probability):.4f} "
            f"cal_max={movement_cal_probability.max():.4f} "
            f"movement_threshold={movement_threshold:.4f} "
            f"positive_rate={float(np.mean(move_y_cal)):.2%} "
            f"positives={int((move_y_cal == 1).sum())} "
            f"negatives={int((move_y_cal == 0).sum())}"
        )

        movement_calibrated = True
        print(
            f"[TWO-STAGE] Meaningful-move threshold={movement_threshold:.3f} "
            f"horizon_bars={1 if mode == 'daily' else 4}"
        )

    print("\n[EVALUATION] Computing metrics on final test set...")
    metrics = {}

    try:
        if num_classes == 2:
            raw_cal_proba = base_model.predict_proba(X_cal)[:, 1]
            calibrated_cal_proba = final_model.predict_proba(X_cal)[:, 1]

            threshold_proba_cal = calibrated_cal_proba

            threshold_opt = optimize_binary_decision_threshold(
                y_cal,
                threshold_proba_cal,
                metric="balanced_accuracy",
                min_threshold=0.45,
                max_threshold=0.70,
                step=0.01,
            )

            best_threshold = float(threshold_opt["best_threshold"])
            artifact_decision_threshold = best_threshold

            wf_threshold = None
            wf_trade_count = None
            if walk_forward_evaluation:
                wf_threshold = walk_forward_evaluation.get("decision_threshold")
                wf_trade_count = (walk_forward_evaluation.get("aggregate") or {}).get(
                    "trade_count"
                )

            print(
                f"[THRESHOLD CONSISTENCY] {symbol}/{mode} "
                f"artifact_threshold={best_threshold:.4f} "
                f"walk_forward_threshold="
                f"{'None' if wf_threshold is None else f'{float(wf_threshold):.4f}'} "
                f"artifact_test_rows={len(X_test)} "
                f"walk_forward_rows={len(evaluation_X) if evaluation_X is not None else 0} "
                f"walk_forward_trades={wf_trade_count}"
            )

            if best_threshold <= 0.3500001:
                print(
                    f"[THRESHOLD WARNING] {symbol}/{mode} "
                    f"best threshold is at lower bound; "
                    f"F1 may favor excessive positive predictions"
                )

            print(
                f"[THRESHOLD OPTIMIZATION KEYS] {symbol}/{mode} "
                f"keys={list(threshold_opt.keys())} "
                f"result={threshold_opt}"
            )

            print(
                f"[THRESHOLD OPTIMIZATION] {symbol}/{mode} "
                f"source=calibrated "
                f"cal_min={threshold_proba_cal.min():.4f} "
                f"cal_median={np.median(threshold_proba_cal):.4f} "
                f"cal_max={threshold_proba_cal.max():.4f} "
                f"best_threshold={best_threshold:.4f} "
                f"positive_rate={float(np.mean(y_cal)):.2%} "
                f"positives={int((y_cal == 1).sum())} "
                f"negatives={int((y_cal == 0).sum())}"
            )

            print(
                f"[CALIBRATION CLASS BALANCE] {symbol}/{mode} "
                f"positive_rate={float(np.mean(y_cal)):.2%} "
                f"positives={int(np.sum(y_cal == 1))} "
                f"negatives={int(np.sum(y_cal == 0))}"
            )

            raw_test_proba = base_model.predict_proba(X_test)[:, 1]
            calibrated_test_proba = final_model.predict_proba(X_test)[:, 1]

            y_pred_proba = calibrated_test_proba

            positive_count = int((y_pred_proba >= best_threshold).sum())
            total_count = len(y_pred_proba)
            positive_rate = positive_count / total_count if total_count else 0.0

            print(
                f"[TRADE FUNNEL] {symbol}/{mode} "
                f"evaluation_rows={len(evaluation_X) if evaluation_X is not None else 0} "
                f"test_rows={len(X_test)} "
                f"test_positive_predictions={positive_count} "
                f"test_positive_rate={positive_rate:.2%} "
                f"walk_forward_trades={wf_trade_count}"
            )

            print(
                f"[PROBABILITY DIAGNOSTICS] {symbol}/{mode} "
                f"raw_min={raw_test_proba.min():.4f} "
                f"raw_median={np.median(raw_test_proba):.4f} "
                f"raw_max={raw_test_proba.max():.4f} "
                f"calibrated_min={calibrated_test_proba.min():.4f} "
                f"calibrated_median={np.median(calibrated_test_proba):.4f} "
                f"calibrated_max={calibrated_test_proba.max():.4f} "
                f"decision_source=calibrated "
                f"positive_predictions={positive_count}/{total_count} "
                f"positive_rate={positive_rate:.2%} "
                f"threshold={best_threshold:.4f}"
            )

            y_pred = (y_pred_proba >= best_threshold).astype(int)

            metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
            metrics["logloss"] = (
                float(log_loss(y_test, y_pred_proba, labels=[0, 1]))
                if len(np.unique(y_test)) >= 2
                else None
            )
            metrics["roc_auc"] = (
                float(roc_auc_score(y_test, y_pred_proba))
                if len(np.unique(y_test)) >= 2
                else None
            )
            metrics["precision"] = float(
                precision_score(y_test, y_pred, zero_division=0)
            )
            metrics["recall"] = float(recall_score(y_test, y_pred, zero_division=0))
            metrics["f1"] = float(f1_score(y_test, y_pred, zero_division=0))
            metrics["confusion_matrix"] = confusion_matrix(
                y_test, y_pred, labels=[0, 1]
            ).tolist()
            metrics["decision_threshold"] = float(best_threshold)
            metrics["threshold_metric"] = threshold_opt["metric"]
            metrics["threshold_score"] = float(threshold_opt["best_score"])
            metrics["threshold_grid"] = threshold_opt["grid"]
            metrics["threshold_eligible"] = threshold_opt.get("eligible")
            metrics["threshold_selected_metrics"] = threshold_opt.get(
                "selected_metrics"
            )

            if len(np.unique(y_test)) >= 2:
                metrics["brier_score"] = float(brier_score_loss(y_test, y_pred_proba))
                avg_pred = float(np.mean(y_pred_proba))
                actual_rate = float(np.mean(y_test))
                metrics["calibration_error"] = abs(avg_pred - actual_rate)
            else:
                metrics["brier_score"] = None
                metrics["calibration_error"] = None
        else:
            y_pred_proba = final_model.predict_proba(X_test)
            y_pred = np.argmax(y_pred_proba, axis=1)
            all_classes = list(range(num_classes))

            metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
            metrics["logloss"] = float(
                log_loss(y_test, y_pred_proba, labels=all_classes)
            )
            metrics["precision_weighted"] = float(
                precision_score(y_test, y_pred, average="weighted", zero_division=0)
            )
            metrics["recall_weighted"] = float(
                recall_score(y_test, y_pred, average="weighted", zero_division=0)
            )
            metrics["f1_weighted"] = float(
                f1_score(y_test, y_pred, average="weighted", zero_division=0)
            )
            metrics["precision_per_class"] = precision_score(
                y_test, y_pred, average=None, zero_division=0, labels=all_classes
            ).tolist()
            metrics["recall_per_class"] = recall_score(
                y_test, y_pred, average=None, zero_division=0, labels=all_classes
            ).tolist()
            metrics["confusion_matrix"] = confusion_matrix(
                y_test, y_pred, labels=all_classes
            ).tolist()

            from sklearn.preprocessing import label_binarize

            y_test_bin = label_binarize(y_test, classes=range(num_classes))
            metrics["brier_score"] = float(
                np.mean(np.sum((y_test_bin - y_pred_proba) ** 2, axis=1))
            )

        if mode != "daily" and isinstance(X_test.index, pd.DatetimeIndex):
            test_session_types = intraday_session_types(X_test.index)
            extended_test_metrics = {}
            for session_name in ("premarket", "after_hours"):
                session_mask = test_session_types.eq(session_name).to_numpy()
                session_labels = np.asarray(y_test)[session_mask]
                session_probabilities = np.asarray(y_pred_proba)[session_mask]
                session_rows = int(session_mask.sum())
                if num_classes == 2:
                    session_brier = (
                        float(
                            brier_score_loss(
                                session_labels.astype(int),
                                session_probabilities,
                            )
                        )
                        if session_rows and len(np.unique(session_labels)) >= 2
                        else None
                    )
                else:
                    session_brier = (
                        float(
                            np.mean(
                                np.sum(
                                    (
                                        np.eye(num_classes)[session_labels.astype(int)]
                                        - session_probabilities
                                    )
                                    ** 2,
                                    axis=1,
                                )
                            )
                        )
                        if session_rows
                        else None
                    )
                extended_test_metrics[session_name] = {
                    "samples": session_rows,
                    "brier_score": session_brier,
                }
            metrics["extended_hours_test"] = extended_test_metrics
            print(
                f"[EXTENDED-HOURS TEST] {symbol}/{mode}: "
                f"premarket={extended_test_metrics['premarket']} "
                f"after_hours={extended_test_metrics['after_hours']}"
            )

        metrics["train_samples"] = len(X_train)
        metrics["calibration_samples"] = len(X_cal)
        metrics["test_samples"] = len(X_test)
        metrics["num_features"] = len(feature_list)
        metrics["num_classes"] = num_classes
        metrics["train_class_counts"] = y_train.value_counts().to_dict()
        metrics["calibration_class_counts"] = y_cal.value_counts().to_dict()
        metrics["test_class_counts"] = y_test.value_counts().to_dict()
    except Exception as e:
        print(f"[EVALUATION] Critical metrics error: {e}")
        traceback.print_exc()
        metrics = {
            "train_samples": len(X_train),
            "calibration_samples": len(X_cal),
            "test_samples": len(X_test),
            "num_features": len(feature_list),
            "num_classes": num_classes,
            "error": str(e),
        }

    if extended_hours_oof_metrics is not None:
        metrics["extended_hours_oof"] = extended_hours_oof_metrics

    print(f"\n{'=' * 60}")
    print(f"MODEL VALIDATION: {symbol}/{mode.upper()}")
    print(f"{'=' * 60}")
    print(f"Training samples: {metrics.get('train_samples', 'N/A')}")
    print(f"Calibration samples: {metrics.get('calibration_samples', 'N/A')}")
    print(f"Test samples: {metrics.get('test_samples', 'N/A')}")
    print(f"Features: {metrics.get('num_features', 'N/A')}")
    print(f"Target classes: {metrics.get('num_classes', 'N/A')}")
    print(f"Calibrated: {'Yes' if calibrated else 'No'}")
    print("-" * 60)

    key_metrics = [
        "accuracy",
        "logloss",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "brier_score",
        "calibration_error",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
    ]
    for k in key_metrics:
        if k in metrics and metrics[k] is not None:
            v = metrics[k]
            try:
                print(f"{k:20} {v:.4f}")
            except Exception:
                print(f"{k:20} {v}")

    if "confusion_matrix" in metrics:
        print("-" * 60)
        print("Confusion Matrix:")
        cm = np.array(metrics["confusion_matrix"])
        for row in cm:
            print(" ", row)
    print("=" * 60)

    feature_importance = extract_feature_importance(base_model, feature_list)

    selected_features = list(feature_list)
    full_features = list(full_feature_schema["features"])

    if selected_features != list(
        full_feature_schema.get("selected_features", selected_features)
    ):
        pass

    if len(selected_features) != len(feature_list):
        raise RuntimeError(
            "Selected feature list and feature_list length diverged"
        )

    if set(selected_features) - set(full_features):
        raise RuntimeError(
            "Selected features contain columns absent from full feature schema"
        )

    full_feature_schema["selected_features"] = selected_features
    full_feature_schema["selected_feature_count"] = len(selected_features)

    return {
        "model": final_model,
        "movement_model": movement_model,
        "movement_threshold": movement_threshold,
        "base_model": base_model,
        # Features actually expected by the trained final model.
        "features": list(selected_features),
        # Canonical inference schema for the final model.
        "feature_schema": {
            "features": list(selected_features),
            "feature_count": len(selected_features),
        },
        # Original schema before SHAP selection.
        "full_feature_schema": {
            "features": list(full_feature_schema["features"]),
            "feature_count": int(full_feature_schema["feature_count"]),
        },
        "selected_feature_metadata": selected_feature_metadata,
        "split_metadata": {
            "train_samples": len(X_train),
            "calibration_samples": len(X_cal),
            "test_samples": len(X_test),
            "windows_overlap": False,
        },
        "regime_config": regime_config,
        "metrics": metrics,
        "walk_forward_evaluation": walk_forward_evaluation,
        "decision_threshold": metrics.get("decision_threshold", 0.5),
        "artifact_decision_threshold": artifact_decision_threshold,
        "walk_forward_decision_threshold": (
            float(walk_forward_evaluation.get("decision_threshold"))
            if walk_forward_evaluation is not None
            else None
        ),
        "threshold_optimization": {
            "best_threshold": metrics.get("decision_threshold", 0.5),
            "metric": metrics.get("threshold_metric"),
            "best_score": metrics.get("threshold_score"),
            "grid": metrics.get("threshold_grid", []),
            "eligible": metrics.get("threshold_eligible"),
            "selected_metrics": metrics.get("threshold_selected_metrics"),
        },
        "feature_importance": feature_importance,
        "extended_hours_trained": extended_hours_trained,
        "extended_hours_training_rows": extended_training_rows,
        "trained_at": datetime.now().isoformat(),
        "symbol": symbol,
        "mode": mode,
        "calibrated": calibrated and (movement_calibrated if use_two_stage else True),
        "num_classes": num_classes,
        "target_type": (
            "multiclass"
            if use_multiclass
            else ("two_stage" if use_two_stage else "binary")
        ),
        "target_horizon_bars": (1 if mode == "daily" else (4 if use_two_stage else 1)),
        "target_horizon": (
            "next_trading_session"
            if mode == "daily"
            else ("60min" if use_two_stage else "15min")
        ),
        "class_weighted": scale_pos_weight is not None,
    }


# ---------------------------------------------------------
# LOAD / SAVE MODEL ARTIFACT
# ---------------------------------------------------------
def save_model(artifact: dict, symbol: str, mode: str):
    path = os.path.join(MODEL_DIR, f"{symbol}_{mode}_xgb.pkl")
    joblib.dump(artifact, path)
    print(f"[SAVE] Model saved to {path}")


def load_model(symbol: str, mode: str):
    """Load saved model artifact. Returns None if missing."""
    path = os.path.join(MODEL_DIR, f"{symbol}_{mode}_xgb.pkl")
    if not os.path.exists(path):
        print(f"[WARN] Missing model artifact: {path}")
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"[ERROR] load_model({symbol}, {mode}): {e}")
        return None


# ---------------------------------------------------------
# PREDICT FROM MODEL ARTIFACT
# ---------------------------------------------------------
def predict_from_model(model_dict, df_features: pd.DataFrame):
    """
    Return prediction dict with support for multi-class models.
    Enforces exact feature schema; no fuzzy matching, no silent zero-fill.
    """
    if model_dict is None:
        return None

    if df_features is None or df_features.empty:
        print("[ERROR] predict_from_model: df_features empty.")
        return None

    try:
        model = model_dict["model"]

        schema = model_dict.get("feature_schema")
        if isinstance(schema, dict):
            featcols = schema.get("features", [])
        else:
            featcols = schema or model_dict.get("features", [])

        if not featcols:
            raise ValueError("Model artifact has no selected feature schema")

        num_classes = model_dict.get("num_classes", 2)
        target_type = model_dict.get("target_type", "binary")

        latest = df_features.tail(1).copy()
        latest = _validate_exact_feature_schema(
            latest,
            featcols,
            allow_extra=True,
        )
        X = latest.loc[:, featcols].copy()
        proba = model.predict_proba(X)[0]

        if target_type == "two_stage":
            movement_model = model_dict.get("movement_model")
            if movement_model is None:
                raise ValueError("Two-stage artifact is missing movement_model")
            direction_probability = float(proba[1])
            movement_probability = float(movement_model.predict_proba(X)[0][1])
            movement_threshold = float(model_dict.get("movement_threshold", 0.5))
            movement_expected = movement_probability >= movement_threshold
            bullish_probability = movement_probability * direction_probability
            bearish_probability = movement_probability * (1.0 - direction_probability)
            flat_probability = 1.0 - movement_probability
            final_probability = direction_probability if movement_expected else 0.5
            decision_threshold = float(model_dict.get("decision_threshold", 0.5))
            threshold_metric = (model_dict.get("threshold_optimization") or {}).get(
                "metric"
            )
            return {
                "final_prob": final_probability,
                "direction_prob": direction_probability,
                "movement_prob": movement_probability,
                "movement_threshold": movement_threshold,
                "movement_expected": movement_expected,
                "bullish_prob": bullish_probability,
                "bearish_prob": bearish_probability,
                "flat_prob": flat_probability,
                "decision_threshold": decision_threshold,
                "threshold_metric": threshold_metric,
                "position_size": probability_to_position_size(
                    final_probability, decision_threshold=decision_threshold
                ),
                "top_features": model_dict.get("feature_importance", {}).get(
                    "top_features", []
                )[:5],
                "model_type": "two_stage",
                "target_horizon": model_dict.get("target_horizon"),
            }

        if num_classes == 2 or target_type == "binary":
            prob = float(proba[1])
            decision_threshold = float(model_dict.get("decision_threshold", 0.5))
            threshold_metric = (model_dict.get("threshold_optimization") or {}).get(
                "metric"
            )

            # The saved artifact owns calibration. Applying a second external
            # calibrator here would distort an already calibrated probability.
            return {
                "final_prob": prob,
                "decision_threshold": decision_threshold,
                "threshold_metric": threshold_metric,
                "position_size": probability_to_position_size(
                    prob, decision_threshold=decision_threshold
                ),
                "top_features": model_dict.get("feature_importance", {}).get(
                    "top_features", []
                )[:5],
                "model_type": "binary",
            }

        actual_num_classes = len(proba)

        if actual_num_classes == 5:
            class_names = ["Strong Down", "Weak Down", "Flat", "Weak Up", "Strong Up"]
            predicted_class = int(np.argmax(proba))
            confidence = float(np.max(proba))

            bullish_simple = float(proba[3] + proba[4])
            bearish_simple = float(proba[0] + proba[1])
            flat_prob = float(proba[2])

            bullish_weighted = float(proba[3] * 1.0 + proba[4] * 2.0)
            bearish_weighted = float(proba[0] * 2.0 + proba[1] * 1.0)
            total_weighted = bullish_weighted + bearish_weighted + float(proba[2] * 0.5)

            if total_weighted > 0:
                final_prob = bullish_weighted / (bullish_weighted + bearish_weighted)
            else:
                final_prob = 0.5

            return {
                "final_prob": float(final_prob),
                "probabilities": proba.tolist(),
                "predicted_class": predicted_class,
                "predicted_class_name": class_names[predicted_class],
                "confidence": confidence,
                "bullish_prob": bullish_simple,
                "bearish_prob": bearish_simple,
                "flat_prob": flat_prob,
                "class_breakdown": {
                    "strong_down": float(proba[0]),
                    "weak_down": float(proba[1]),
                    "flat": float(proba[2]),
                    "weak_up": float(proba[3]),
                    "strong_up": float(proba[4]),
                },
                "num_classes": num_classes,
                "model_type": target_type,
            }

        predicted_class = int(np.argmax(proba))
        confidence = float(np.max(proba))
        mid_point = actual_num_classes / 2.0
        bullish_prob = float(
            sum(proba[i] for i in range(actual_num_classes) if i >= mid_point)
        )
        bearish_prob = float(
            sum(proba[i] for i in range(actual_num_classes) if i < mid_point)
        )
        if bullish_prob + bearish_prob > 0:
            final_prob = bullish_prob / (bullish_prob + bearish_prob)
        else:
            final_prob = 0.5

        return {
            "final_prob": float(final_prob),
            "probabilities": proba.tolist(),
            "predicted_class": predicted_class,
            "predicted_class_name": f"Class_{predicted_class}",
            "confidence": confidence,
            "bullish_prob": bullish_prob,
            "bearish_prob": bearish_prob,
            "flat_prob": 0.0,
            "class_breakdown": {
                f"class_{i}": float(proba[i]) for i in range(actual_num_classes)
            },
            "num_classes": actual_num_classes,
            "model_type": target_type,
        }

    except ValueError as e:
        msg = str(e)
        if "Feature schema mismatch" in msg:
            print(f"[WARN] predict_from_model skipped incompatible artifact: {msg}")
            return None
        print(f"[ERROR] predict_from_model: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] predict_from_model: {e}")
        traceback.print_exc()
        return None


# ---------------------------------------------------------
# HELPER: Extract final probability
# ---------------------------------------------------------
def get_final_prob(prediction):
    """Extract final probability from prediction (handles binary and multi-class)."""
    if prediction is None:
        return None
    if isinstance(prediction, (float, int)):
        return float(prediction)
    if isinstance(prediction, dict):
        return prediction.get("final_prob")
    print(f"[WARN] Unknown prediction type: {type(prediction)}")
    return None


# ---------------------------------------------------------
# HELPER: Human-readable signal details
# ---------------------------------------------------------
def get_signal_details(prediction) -> str:
    """Get human-readable signal description for logging/debugging."""
    if prediction is None:
        return "No signal"

    if isinstance(prediction, (float, int)):
        prob = float(prediction)
        if prob >= 0.65:
            return f"STRONG BUY ({prob:.1%})"
        elif prob >= 0.55:
            return f"BUY ({prob:.1%})"
        elif prob <= 0.35:
            return f"STRONG SELL ({prob:.1%})"
        elif prob <= 0.45:
            return f"SELL ({prob:.1%})"
        else:
            return f"NEUTRAL ({prob:.1%})"

    if isinstance(prediction, dict):
        final_prob = prediction.get("final_prob", 0.5)
        pred_class = prediction.get("predicted_class_name", "Unknown")
        confidence = prediction.get("confidence", 0.0)
        breakdown = prediction.get("class_breakdown", {})

        if final_prob >= 0.65:
            signal = "STRONG BUY"
        elif final_prob >= 0.55:
            signal = "BUY"
        elif final_prob <= 0.35:
            signal = "STRONG SELL"
        elif final_prob <= 0.45:
            signal = "SELL"
        else:
            signal = "NEUTRAL"

        details = f"{signal} ({final_prob:.1%})"
        details += f" | Prediction: {pred_class} (conf: {confidence:.1%})"
        details += f" | [â†“â†“:{breakdown.get('strong_down', 0):.0%} "
        details += f"â†“:{breakdown.get('weak_down', 0):.0%} "
        details += f"â†’:{breakdown.get('flat', 0):.0%} "
        details += f"â†‘:{breakdown.get('weak_up', 0):.0%} "
        details += f"â†‘â†‘:{breakdown.get('strong_up', 0):.0%}]"
        position_size = prediction.get("position_size")
        if position_size is not None:
            details += f" | size={position_size:+.2f}"
        return details

    return "Unknown signal type"


# ---------------------------------------------------------
# VIX MARKET REGIME FILTER
# ---------------------------------------------------------
def fetch_vix_level() -> float:
    """Fetch latest VIX close. Returns 20.0 as safe default if unavailable."""
    try:
        df_vix = fetch_historical_data("^VIX", period="5d", interval="1d")
        if df_vix is not None and not df_vix.empty:
            vix_close = df_vix["Close"]
            vix_close = (
                vix_close.iloc[:, 0]
                if isinstance(vix_close, pd.DataFrame)
                else vix_close
            )
            vix_close = vix_close.dropna()
            if len(vix_close) > 0:
                return float(vix_close.iloc[-1])
    except Exception as e:
        print(f"[VIX] Could not fetch VIX: {e}")
    return 20.0


def vix_market_regime_filter(prob: float, vix_level: float, symbol: str = "") -> float:
    """Derate prediction confidence when VIX is elevated."""
    if vix_level > 30:
        scale = 0.80
    elif vix_level > 25:
        scale = 0.85
    elif vix_level > 18:
        scale = 0.92
    else:
        scale = 1.0

    if scale < 1.0:
        adjusted = 0.5 + (prob - 0.5) * scale
        print(
            f"[VIX FILTER] {symbol} VIX={vix_level:.1f} -> scale={scale:.2f} | prob {prob:.3f} -> {adjusted:.3f}"
        )
        return float(adjusted)

    return float(prob)


# ---------------------------------------------------------
# CONTRADICTION DETECTION
# ---------------------------------------------------------
def detect_contradiction(
    daily_prob, intraday_prob, momentum=None, symbol="", verbose=True
):
    """Detect contradictions between daily and intraday signals."""
    if daily_prob is None or intraday_prob is None:
        return {"contradiction": False, "reason": "Missing data", "severity": "low"}

    bullish_threshold = 0.55
    bearish_threshold = 0.45

    daily_bullish = daily_prob > bullish_threshold
    daily_bearish = daily_prob < bearish_threshold
    intraday_bullish = intraday_prob > bullish_threshold
    intraday_bearish = intraday_prob < bearish_threshold

    models_disagree = (daily_bullish and intraday_bearish) or (
        daily_bearish and intraday_bullish
    )

    if not models_disagree:
        return {"contradiction": False, "reason": "Models agree", "severity": "low"}

    if momentum is not None and abs(momentum) > 0.01:
        reason = (
            f"{symbol}: Strong momentum ({momentum:+.2%}) but models disagree - "
            f"daily={daily_prob:.2f}, intraday={intraday_prob:.2f}"
        )
        if verbose:
            print(f"[CONTRADICTION] {reason}")
        return {"contradiction": True, "reason": reason, "severity": "high"}

    extreme_disagree = (daily_prob > 0.65 and intraday_prob < 0.35) or (
        daily_prob < 0.35 and intraday_prob > 0.65
    )

    if extreme_disagree:
        reason = f"{symbol}: Extreme disagreement - daily={daily_prob:.2f}, intraday={intraday_prob:.2f}"
        if verbose:
            print(f"[CONTRADICTION] {reason}")
        return {"contradiction": True, "reason": reason, "severity": "medium"}

    return {"contradiction": False, "reason": "Mild disagreement", "severity": "low"}


def _get_intraday_regime_state(symbol: str) -> dict:
    key = str(symbol).upper().strip()
    state = _intraday_regime_state.get(key)
    if state is None:
        state = {
            "active_regime": None,
            "active_since": None,
            "pending_regime": None,
            "pending_count": 0,
            "last_bar_time": None,
        }
        _intraday_regime_state[key] = state
    return state


def _resolve_intraday_regime_with_confirmation(
    symbol: str,
    bar_time,
    raw_regime: str,
    mom_value: float,
    vol_value: float,
    mom_trig: float,
    vol_trig: float,
    confirm_bars: int = 2,
    min_hold_bars: int = 2,
    hysteresis_enter_mult: float = 1.00,
    hysteresis_exit_mult: float = 0.85,
) -> dict:
    state = _get_intraday_regime_state(symbol)

    current_bar_ts = pd.Timestamp(bar_time) if bar_time is not None else None
    if current_bar_ts is not None and state.get("last_bar_time") is not None:
        if current_bar_ts == pd.Timestamp(state["last_bar_time"]):
            bars_since_switch = 0
            if state["active_since"] is not None:
                try:
                    bars_since_switch = max(
                        0,
                        int((current_bar_ts - pd.Timestamp(state["active_since"])) / pd.Timedelta(minutes=15))
                    )
                except Exception:
                    bars_since_switch = 0
            return {
                "active_regime": state["active_regime"],
                "raw_regime": raw_regime,
                "hysteresis_regime": state["active_regime"],
                "pending_count": state["pending_count"],
                "bars_since_switch": bars_since_switch,
                "switched": False,
            }

    enter_mom = (
        abs(mom_value) >= mom_trig * hysteresis_enter_mult
        or vol_value >= vol_trig * hysteresis_enter_mult
    )
    stay_mom = (
        abs(mom_value) >= mom_trig * hysteresis_exit_mult
        or vol_value >= vol_trig * hysteresis_exit_mult
    )

    hysteresis_regime = "mom" if enter_mom else "mr"
    if state["active_regime"] == "mom" and stay_mom:
        hysteresis_regime = "mom"

    if state["active_regime"] is None:
        state["active_regime"] = hysteresis_regime
        state["active_since"] = bar_time
        state["pending_regime"] = None
        state["pending_count"] = 0
        state["last_bar_time"] = bar_time
        return {
            "active_regime": state["active_regime"],
            "raw_regime": raw_regime,
            "hysteresis_regime": hysteresis_regime,
            "pending_count": 0,
            "bars_since_switch": 0,
            "switched": True,
        }

    bars_since_switch = 0
    if state["active_since"] is not None and bar_time is not None:
        try:
            bars_since_switch = max(
                0,
                int((pd.Timestamp(bar_time) - pd.Timestamp(state["active_since"])) / pd.Timedelta(minutes=15))
            )
        except Exception:
            bars_since_switch = 0

    if hysteresis_regime == state["active_regime"]:
        state["pending_regime"] = None
        state["pending_count"] = 0
    else:
        if bars_since_switch < min_hold_bars:
            state["pending_regime"] = hysteresis_regime
            state["pending_count"] = 0
        else:
            if state["pending_regime"] == hysteresis_regime:
                state["pending_count"] += 1
            else:
                state["pending_regime"] = hysteresis_regime
                state["pending_count"] = 1

            if state["pending_count"] >= confirm_bars:
                state["active_regime"] = hysteresis_regime
                state["active_since"] = bar_time
                state["pending_regime"] = None
                state["pending_count"] = 0
                state["last_bar_time"] = bar_time
                return {
                    "active_regime": state["active_regime"],
                    "raw_regime": raw_regime,
                    "hysteresis_regime": hysteresis_regime,
                    "pending_count": 0,
                    "bars_since_switch": bars_since_switch,
                    "switched": True,
                }

    state["last_bar_time"] = bar_time
    return {
        "active_regime": state["active_regime"],
        "raw_regime": raw_regime,
        "hysteresis_regime": hysteresis_regime,
        "pending_count": state["pending_count"],
        "bars_since_switch": bars_since_switch,
        "switched": False,
    }

# ---------------------------------------------------------
# COMPUTE SIGNALS (kept compatible; inference now uses strict schema)
# ---------------------------------------------------------
def compute_signals(
    symbol, lookback_minutes=60, intraday_weight=INTRADAY_WEIGHT, resample_to="15min"
):
    results = {
        "daily_prob": None,
        "intraday_prob": None,
        "final_prob": None,
        "price": None,
        "allow_intraday": True,
        "intraday_rows": 0,
        "intraday_before": 0,
        "daily_rows": 0,
        "used_lookback_minutes": lookback_minutes,
        "intraday_quality_score": 0.0,
        "intraday_model_used": None,
        "intraday_diagnostic_model_used": None,
        "intraday_probability_status": "unavailable",
        "intraday_diagnostic_prediction": None,
        "intraday_diagnostic_signal": None,
        "intraday_vol": None,
        "intraday_mom": None,
        "intraday_latest_bar_time": None,
        "intraday_bar_age_minutes": None,
        "intraday_bar_source": None,
        "intraday_bar_fresh": False,
        "intraday_momentum_fresh": None,
        "intraday_momentum_confirmed": False,
        "intraday_momentum_bars": 0,
        "intraday_momentum_start_bar_time": None,
        "intraday_momentum_freshness_reason": None,
        "intraday_momentum_source": None,
        "intraday_momentum_window_minutes": None,
        "intraday_momentum_current_price": None,
        "intraday_momentum_current_price_time": None,
        "pyramid_yahoo_price_fresh": False,
        "pyramid_yahoo_price_age_seconds": None,
        "intraday_momentum_reference_price": None,
        "intraday_momentum_reference_time": None,
        "relative_return_4_source": None,
        "volume_time_ratio_source": None,
        "market_session": None,
        "extended_hours_model_ready": None,
        "intraday_regime": None,
        "daily_prediction": None,
        "intraday_prediction": None,
        "daily_signal": None,
        "intraday_signal": None,
        "intraday_weight": min(float(intraday_weight), 0.55),
        "intraday_weight_base_capped": min(float(intraday_weight), 0.55),
        "intraday_stale_factor": 1.0,
        "daily_model_age_days": None,
        "intraday_mr_model_age_days": None,
        "intraday_mom_model_age_days": None,
        "intraday_legacy_model_age_days": None,
    }

    symU = str(symbol).upper().strip()
    market_session = get_market_session()
    results["market_session"] = market_session
    is_extended_session = market_session in {"pre_market", "after_hours"}
    results["extended_hours_model_ready"] = not is_extended_session
    if is_extended_session:
        results["intraday_momentum_fresh"] = False
        results["intraday_momentum_freshness_reason"] = (
            "current_session_momentum_not_confirmed"
        )

    model_daily = load_model(symU, mode="daily")
    model_intra_mr = load_model(symU, mode="intraday_mr")
    model_intra_mom = load_model(symU, mode="intraday_mom")
    model_intraday_legacy = load_model(symU, mode="intraday")

    fresh_models = []
    stale_models = []

    for mode_name, model_obj in [
        ("intraday_mr", model_intra_mr),
        ("intraday_mom", model_intra_mom),
        ("intraday", model_intraday_legacy),
    ]:
        if model_obj is None:
            continue

        age = _artifact_age_days(model_obj)

        if age is None:
            print(f"[STALE] {symU} {mode_name} skipped: missing/invalid trained_at")
            continue

        if age <= 30:
            fresh_models.append((mode_name, model_obj, age))
        else:
            print(f"[STALE] {symU} {mode_name} stale: age={age:.1f}d > 30d")
            stale_models.append((mode_name, model_obj, age))

    vix_level = fetch_vix_level()
    results["vix_level"] = vix_level
    print(f"[VIX] {symU} Current VIX: {vix_level:.1f}")

    df_daily = None
    try:
        df_daily = fetch_historical_data(symU, period="6mo", interval="1d")
        if df_daily is not None and not df_daily.empty:
            results["daily_rows"] = len(df_daily)
            df_feat = build_daily_features(df_daily)

            daily_prediction = predict_from_model(model_daily, df_feat)
            results["daily_prediction"] = daily_prediction
            results["daily_prob"] = get_final_prob(daily_prediction)
            results["daily_signal"] = get_signal_details(daily_prediction)

            if results["daily_prob"] is not None:
                try:
                    dp = results["daily_prob"]
                    current_price = results.get("price")
                    if current_price is None:
                        dc = df_daily["Close"]
                        dc = dc.iloc[:, 0] if isinstance(dc, pd.DataFrame) else dc
                        dc = dc.dropna()
                        if len(dc) > 0:
                            current_price = float(dc.iloc[-1])

                    if current_price and float(current_price) > 0:
                        log_prediction(
                            symbol=symU,
                            mode="daily",
                            predicted_prob=float(dp),
                            price=float(current_price),
                            prediction_details=(
                                daily_prediction
                                if isinstance(daily_prediction, dict)
                                else None
                            ),
                        )
                except Exception as e:
                    print(f"[WARN] Could not log daily prediction: {e}")

                try:
                    dc = df_daily["Close"]
                    dc = dc.iloc[:, 0] if isinstance(dc, pd.DataFrame) else dc
                    dc = dc.dropna()
                    if len(dc) > 0 and results["price"] is None:
                        results["price"] = float(dc.iloc[-1])
                except Exception:
                    pass
    except Exception as e:
        print(f"[ERROR] Daily prediction error: {e}")

    df_intra_resampled = None
    vol = None
    extended_session_vol = None
    mom1h = None
    close = pd.Series(dtype=float)

    try:
        candidate_lookbacks = sorted(
            [
                lookback_minutes,
                max(lookback_minutes, 600),
                max(lookback_minutes, 900),
                max(lookback_minutes, 1200),
            ]
        )

        best_df = None
        best_len = 0
        best_lb = lookback_minutes
        best_before = 0
        best_source = None
        rt = str(resample_to).lower().replace("mins", "min").replace(" ", "")

        for lb in candidate_lookbacks:
            df_raw = fetch_intraday_history(
                symU,
                lookback_minutes=lb,
                interval=resample_to,
                include_extended_hours=is_extended_session,
            )
            if df_raw is None or df_raw.empty:
                continue

            before = len(df_raw)
            data_source = df_raw.attrs.get("data_source", "unknown")
            df_raw = df_raw.sort_index()
            if rt in ["15m", "15min", "15t"]:
                df_res = df_raw.dropna()
            else:
                df_res = df_raw.resample(resample_to).last().dropna()

            n_res = len(df_res)
            if n_res > best_len:
                best_len = n_res
                best_df = df_res
                best_lb = lb
                best_before = before
                best_source = data_source
            if n_res >= 80:
                break

        df_intra_resampled = best_df
        results["intraday_before"] = best_before
        results["used_lookback_minutes"] = best_lb
        results["intraday_bar_source"] = best_source
        results["intraday_rows"] = (
            0 if df_intra_resampled is None else len(df_intra_resampled)
        )

        if df_intra_resampled is not None and not df_intra_resampled.empty:
            latest_bar_time = pd.Timestamp(df_intra_resampled.index[-1])
            if latest_bar_time.tzinfo is None:
                latest_bar_time = latest_bar_time.tz_localize("UTC")
            else:
                latest_bar_time = latest_bar_time.tz_convert("UTC")
            results["intraday_latest_bar_time"] = latest_bar_time.isoformat()
            results["intraday_bar_age_minutes"] = round(
                (pd.Timestamp.now(tz="UTC") - latest_bar_time).total_seconds()
                / 60.0,
                2,
            )
            bar_age_minutes = float(results["intraday_bar_age_minutes"])
            results["intraday_bar_fresh"] = (
                -5.0
                <= bar_age_minutes
                <= float(EXTENDED_HOURS_MAX_INTRADAY_BAR_AGE_MINUTES)
            )
        print(
            f"[INTRADAY MODEL DATA] {symU} "
            f"source={results['intraday_bar_source'] or 'unavailable'} "
            f"latest={results['intraday_latest_bar_time'] or 'NA'} "
            f"age={results['intraday_bar_age_minutes']}m "
            f"fresh={results['intraday_bar_fresh']}"
        )

        if (
            df_intra_resampled is None
            or len(df_intra_resampled) < MIN_INTRADAY_BARS_FOR_FEATURES
        ):
            results["allow_intraday"] = False
            results["intraday_prob"] = None
            results["intraday_quality_score"] = 0.0
            results["intraday_probability_status"] = "insufficient_or_stale_bars"
            if is_extended_session:
                results["intraday_momentum_fresh"] = False
                results["intraday_momentum_freshness_reason"] = (
                    "insufficient_intraday_history"
                )
        else:
            close = df_intra_resampled["Close"]
            close = close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close
            close = close.dropna()

            if len(close) > 0:
                results["price"] = float(close.iloc[-1])
            if len(close) >= 12:
                vol = close.pct_change().rolling(12).std().iloc[-1]
                results["intraday_vol"] = float(vol) if not pd.isna(vol) else None

            session_start = get_current_session_start()
            session_start_utc = (
                pd.Timestamp(session_start).tz_convert("UTC")
                if session_start is not None
                else None
            )
            bars_frame = df_intra_resampled.copy()
            bars_frame.index = pd.to_datetime(
                bars_frame.index, utc=True, errors="coerce"
            )
            bars_frame = bars_frame.loc[~bars_frame.index.isna()].sort_index()
            session_frame = (
                bars_frame.loc[bars_frame.index >= session_start_utc]
                if session_start_utc is not None
                else bars_frame.iloc[0:0]
            )
            momentum_source = "alpaca_iex_bars"
            momentum_frame = session_frame
            yahoo_relative_return_4 = None
            yahoo_relative_return_4_source = None

            # If IEX bars are absent, stale, or too sparse for the one-hour
            # reference, try uncached, timestamped 1-minute extended-session
            # prices. This is diagnostic input only; strategy gates separately
            # require an explicitly confirmed 45-90 minute window.
            bar_latest_time = (
                session_frame.index[-1] if not session_frame.empty else None
            )
            bar_age_minutes = (
                (pd.Timestamp.now(tz="UTC") - bar_latest_time).total_seconds() / 60.0
                if bar_latest_time is not None
                else None
            )
            if (
                session_start_utc is not None
                and (
                    bar_latest_time is None
                    or bar_age_minutes is None
                    or bar_age_minutes
                    > float(EXTENDED_HOURS_MAX_INTRADAY_BAR_AGE_MINUTES)
                    or len(session_frame) < int(EXTENDED_HOURS_MIN_MOMENTUM_BARS)
                )
            ):
                fallback_frame = fetch_intraday_price_fallback(symU)
                if fallback_frame is not None and not fallback_frame.empty:
                    fallback_frame = fallback_frame.copy()
                    fallback_frame.index = pd.to_datetime(
                        fallback_frame.index, utc=True, errors="coerce"
                    )
                    fallback_frame = fallback_frame.loc[
                        ~fallback_frame.index.isna()
                    ].sort_index()
                    fallback_frame = fallback_frame.loc[
                        fallback_frame.index >= session_start_utc
                    ]
                    if not fallback_frame.empty and (
                        bar_latest_time is None
                        or fallback_frame.index[-1] >= bar_latest_time
                    ):
                        momentum_frame = fallback_frame
                        momentum_source = "yfinance_1m_prepost"

            momentum_close = (
                momentum_frame["Close"]
                if not momentum_frame.empty
                else pd.Series(dtype=float)
            )
            momentum_close = (
                momentum_close.iloc[:, 0]
                if isinstance(momentum_close, pd.DataFrame)
                else momentum_close
            )
            momentum_close = pd.to_numeric(momentum_close, errors="coerce").dropna()
            if is_extended_session:
                extended_session_vol = _latest_extended_session_volatility(
                    momentum_frame
                )
                results["intraday_volatility_4bar"] = extended_session_vol
                results["intraday_volatility_4bar_source"] = (
                    momentum_source if extended_session_vol is not None else None
                )
            results["intraday_momentum_bars"] = len(momentum_close)

            # If no price point from this session exists, retain a clearly
            # unconfirmed stale diagnostic from the most recent historical
            # window; it is never eligible to block an ordinary new BUY.
            if momentum_close.empty and not close.empty:
                momentum_close = pd.to_numeric(close, errors="coerce").dropna()
                momentum_source = "stale_historical_bars"

            if len(momentum_close) >= 2:
                momentum_index = pd.to_datetime(
                    momentum_close.index, utc=True, errors="coerce"
                )
                momentum_close.index = momentum_index
                momentum_close = momentum_close.loc[
                    ~momentum_close.index.isna()
                ].sort_index()
                current_time = pd.Timestamp(momentum_close.index[-1])
                current_price = float(momentum_close.iloc[-1])
                now_utc = pd.Timestamp.now(tz="UTC")
                current_age_minutes = (now_utc - current_time).total_seconds() / 60.0
                results["intraday_momentum_current_price"] = current_price
                results["intraday_momentum_current_price_time"] = (
                    current_time.isoformat()
                )
                results["intraday_momentum_source"] = momentum_source
                current_age_seconds = current_age_minutes * 60.0
                yahoo_price_fresh = (
                    momentum_source == "yfinance_1m_prepost"
                    and -5.0
                    <= current_age_seconds
                    <= float(EXTENDED_HOURS_MAX_FALLBACK_PRICE_AGE_SECONDS)
                )
                results["pyramid_yahoo_price_fresh"] = bool(yahoo_price_fresh)
                results["pyramid_yahoo_price_age_seconds"] = round(
                    current_age_seconds, 1
                )
                if yahoo_price_fresh:
                    # Use the same fresh extended-session price for session
                    # return and later position/exposure sizing.
                    results["price"] = current_price
                results["intraday_momentum_fresh"] = (
                    -5.0 <= current_age_minutes
                    <= float(EXTENDED_HOURS_MAX_INTRADAY_BAR_AGE_MINUTES)
                )

                offsets = (current_time - momentum_close.index).total_seconds() / 60.0
                reference_mask = (offsets >= 45.0) & (offsets <= 90.0)
                reference_positions = np.flatnonzero(reference_mask)
                reference_position = None
                if len(reference_positions):
                    reference_position = min(
                        reference_positions,
                        key=lambda pos: abs(float(offsets[pos]) - 60.0),
                    )

                confirmed = (
                    reference_position is not None
                    and results["intraday_momentum_fresh"] is True
                    and session_start_utc is not None
                    and current_time >= session_start_utc
                    and momentum_source != "stale_historical_bars"
                )

                if reference_position is None:
                    prior_positions = np.flatnonzero(offsets > 0.0)
                    if len(prior_positions):
                        reference_position = int(prior_positions[-1])

                if reference_position is not None:
                    reference_time = pd.Timestamp(
                        momentum_close.index[reference_position]
                    )
                    reference_price = float(momentum_close.iloc[reference_position])
                    window_minutes = (
                        current_time - reference_time
                    ).total_seconds() / 60.0
                    if reference_price > 0 and window_minutes > 0:
                        momentum_value = current_price / reference_price - 1.0
                        results["intraday_mom"] = float(momentum_value)
                        results["intraday_momentum_reference_price"] = reference_price
                        results["intraday_momentum_reference_time"] = (
                            reference_time.isoformat()
                        )
                        results["intraday_momentum_window_minutes"] = round(
                            window_minutes, 2
                        )
                        results["intraday_momentum_confirmed"] = bool(confirmed)
                        results["intraday_momentum_start_bar_time"] = (
                            reference_time.isoformat()
                        )
                        if confirmed:
                            mom1h = float(momentum_value)
                            results["intraday_momentum_freshness_reason"] = (
                                "confirmed_45_90m"
                            )
                        elif momentum_source == "stale_historical_bars":
                            results["intraday_momentum_freshness_reason"] = "stale_historical_only"
                        elif not results["intraday_momentum_fresh"]:
                            results["intraday_momentum_freshness_reason"] = "latest_price_stale"
                        else:
                            results["intraday_momentum_freshness_reason"] = "short_window_unconfirmed"

                        if momentum_source == "yfinance_1m_prepost":
                            yahoo_relative_return_4_source = (
                                "yfinance_1m_prepost_unavailable"
                            )
                            if confirmed:
                                try:
                                    spy_frame = fetch_intraday_price_fallback("SPY")
                                    if spy_frame is not None and not spy_frame.empty:
                                        spy_close = spy_frame["Close"]
                                        if isinstance(spy_close, pd.DataFrame):
                                            spy_close = spy_close.iloc[:, 0]
                                        spy_close = pd.to_numeric(
                                            spy_close, errors="coerce"
                                        ).dropna()
                                        spy_close.index = pd.to_datetime(
                                            spy_close.index,
                                            utc=True,
                                            errors="coerce",
                                        )
                                        spy_close = spy_close.loc[
                                            ~spy_close.index.isna()
                                        ].sort_index()
                                        if session_start_utc is not None:
                                            spy_close = spy_close.loc[
                                                spy_close.index >= session_start_utc
                                            ]

                                        def _spy_asof(target_time):
                                            available = spy_close.loc[
                                                spy_close.index <= target_time
                                            ]
                                            if available.empty:
                                                return None
                                            spy_time = available.index[-1]
                                            if (
                                                target_time - spy_time
                                            ).total_seconds() > 300:
                                                return None
                                            return float(available.iloc[-1])

                                        spy_current = _spy_asof(current_time)
                                        spy_reference = _spy_asof(reference_time)
                                        if (
                                            spy_current is not None
                                            and spy_reference is not None
                                            and spy_reference > 0
                                            and reference_price > 0
                                        ):
                                            asset_return = (
                                                current_price / reference_price - 1.0
                                            )
                                            spy_return = (
                                                spy_current / spy_reference - 1.0
                                            )
                                            yahoo_relative_return_4 = (
                                                asset_return - spy_return
                                            )
                                            yahoo_relative_return_4_source = (
                                                "yfinance_1m_prepost"
                                            )
                                except Exception as exc:
                                    print(
                                        f"[RELATIVE YAHOO FALLBACK] {symU}: "
                                        f"SPY alignment unavailable: {exc}"
                                    )
                elif momentum_source == "stale_historical_bars":
                    results["intraday_momentum_freshness_reason"] = "no_historical_reference"
                else:
                    results["intraday_momentum_freshness_reason"] = "no_reference_price"
            else:
                results["intraday_momentum_fresh"] = False
                results["intraday_momentum_freshness_reason"] = "insufficient_price_points"

            # The active-regime model uses only confirmed one-hour momentum.
            # Short-window or stale numeric readings remain diagnostics only.

            regime_config = None
            for artifact in (model_intra_mom, model_intra_mr):
                candidate_config = (artifact or {}).get("regime_config")
                if candidate_config:
                    regime_config = candidate_config
                    break

            if regime_config:
                MOMTRIG = float(regime_config["momentum_threshold"])
                VOLTRIG = float(regime_config["volatility_threshold"])
                if is_extended_session and "extended_volatility_threshold" in regime_config:
                    MOMTRIG = float(
                        regime_config.get("extended_momentum_threshold", MOMTRIG)
                    )
                    VOLTRIG = float(regime_config["extended_volatility_threshold"])
                    vol = extended_session_vol
                    results["intraday_vol"] = vol
                    results["intraday_volatility_window_bars"] = 4
                    results["intraday_volatility_source"] = (
                        results.get("intraday_volatility_4bar_source")
                    )
                results["intraday_regime_source"] = "model_artifact"
                print(
                    f"[REGIME] {symU} artifact thresholds: "
                    f"mom={MOMTRIG:.6f} vol={VOLTRIG:.6f}"
                )
            else:
                # Backward compatibility for artifacts trained before regime
                # metadata was persisted. New models always use the branch above.
                try:
                    adaptive = get_adaptive_regime_thresholds(
                        symU, lookback_days=30, percentile=0.70
                    )
                    MOMTRIG = float(adaptive["mom_trig"])
                    VOLTRIG = float(adaptive["vol_trig"])
                    results["intraday_regime_source"] = "legacy_adaptive"
                except Exception:
                    MOMTRIG = float(INTRADAY_MOM_TRIG)
                    VOLTRIG = float(INTRADAY_VOL_TRIG)
                    ovr = (INTRADAY_REGIME_OVERRIDES or {}).get(symU)
                    if ovr:
                        MOMTRIG = float(ovr.get("mom_trig", MOMTRIG))
                        VOLTRIG = float(ovr.get("vol_trig", VOLTRIG))
                    results["intraday_regime_source"] = "legacy_config"

            fresh_models = []
            stale_models = []

            for mode_name, model_obj in [
                ("intraday_mr", model_intra_mr),
                ("intraday_mom", model_intra_mom),
                ("intraday", model_intraday_legacy),
            ]:
                if model_obj is None:
                    continue

                age = _artifact_age_days(model_obj)

                if age is None:
                    print(f"[STALE] {symU} {mode_name} skipped: missing/invalid trained_at")
                    continue
                elif age <= 30:
                    fresh_models.append((mode_name, model_obj, age))
                else:
                    print(f"[STALE] {symU} {mode_name} stale: age={age:.1f}d > 30d")
                    stale_models.append((mode_name, model_obj, age))
                    print(
                        f"stale_now={[name for name, _, _ in stale_models]}"
                    )

            if vol is not None and mom1h is not None:
                raw_regime = "mom" if (abs(mom1h) >= MOMTRIG or vol >= VOLTRIG) else "mr"
                latest_bar_time = df_intra_resampled.index[-1] if len(df_intra_resampled) else None

                regime_decision = _resolve_intraday_regime_with_confirmation(
                    symbol=symU,
                    bar_time=latest_bar_time,
                    raw_regime=raw_regime,
                    mom_value=float(mom1h),
                    vol_value=float(vol),
                    mom_trig=float(MOMTRIG),
                    vol_trig=float(VOLTRIG),
                    confirm_bars=2,
                    min_hold_bars=2,
                    hysteresis_enter_mult=1.00,
                    hysteresis_exit_mult=0.85,
                )

                results["intraday_regime_raw"] = raw_regime
                results["intraday_regime_hysteresis"] = regime_decision["hysteresis_regime"]
                results["intraday_regime"] = regime_decision["active_regime"]
                results["intraday_regime_pending_count"] = regime_decision["pending_count"]
                results["intraday_regime_bars_since_switch"] = regime_decision["bars_since_switch"]
                results["intraday_regime_switched"] = regime_decision["switched"]
                results["intraday_regime_confirm_bars"] = 2
                results["intraday_regime_min_hold_bars"] = 2
                results["intraday_regime_hysteresis_enter_mult"] = 1.00
                results["intraday_regime_hysteresis_exit_mult"] = 0.85

                print(
                    f"[REGIME ROUTER] {symU} "
                    f"mom1h={float(mom1h):+.4%} vol12={float(vol):.6f} "
                    f"raw={raw_regime} hysteresis={regime_decision['hysteresis_regime']} "
                    f"active={regime_decision['active_regime']} "
                    f"pending={regime_decision['pending_count']} "
                    f"hold_bars={regime_decision['bars_since_switch']}"
                )

                if results["intraday_regime"] == "mom":
                    preferred_order = ["intraday_mom", "intraday_mr", "intraday"]
                else:
                    preferred_order = ["intraday_mr", "intraday_mom", "intraday"]
            else:
                preferred_order = ["intraday_mom", "intraday_mr", "intraday"]

            ordered_candidates = _order_intraday_model_candidates(
                preferred_order,
                fresh_models,
                stale_models,
            )
            candidate_models = [(name, obj) for name, obj, _, _ in ordered_candidates]
            diagnostic_only_models = []
            if is_extended_session:
                rejected_extended = [
                    name
                    for name, artifact in candidate_models
                    if not bool((artifact or {}).get("extended_hours_trained"))
                ]
                candidate_models = [
                    (name, artifact)
                    for name, artifact in candidate_models
                    if bool((artifact or {}).get("extended_hours_trained"))
                ]
                fresh_names = {name for name, _, _ in fresh_models}
                diagnostic_only_models = [
                    (name, artifact)
                    for name, artifact, _, _ in ordered_candidates
                    if name in fresh_names
                    and not bool((artifact or {}).get("extended_hours_trained"))
                ]
                results["extended_hours_model_ready"] = bool(candidate_models)
                if rejected_extended:
                    print(
                        f"[EXTENDED-HOURS MODEL GATE] {symU}: skipped "
                        f"unvalidated artifacts={rejected_extended}"
                    )

            diagnostic_only = bool(
                is_extended_session
                and not candidate_models
                and diagnostic_only_models
                and results.get("intraday_bar_fresh") is True
            )
            prediction_models = (
                diagnostic_only_models if diagnostic_only else candidate_models
            )
            if diagnostic_only:
                results["intraday_probability_status"] = (
                    "unvalidated_extended_hours_blend"
                )
                print(
                    f"[EXTENDED-HOURS UNVALIDATED BLEND] {symU}: "
                    f"using fresh unvalidated artifacts in the daily/intraday "
                    "blend; existing BUY safeguards and sizing limits remain active"
                )
            elif candidate_models:
                results["intraday_probability_status"] = "validated_model"

            print(
                f"fresh={[name for name, _, _ in fresh_models]} "
                f"stale={[name for name, _, _ in stale_models]} "
                f"candidate_pre={[name for name, _ in candidate_models]} "
                f"diagnostic_pre={[name for name, _ in diagnostic_only_models]}"
            )

            if (
                candidate_models
                and ordered_candidates
                and candidate_models[0][0] == ordered_candidates[0][0]
                and ordered_candidates[0][3]
            ):
                selected_name, _, selected_age, _ = ordered_candidates[0]
                print(
                    f"[STALE-REGIME-MATCH] {symU}: preferred {selected_name} artifact "
                    f"is {selected_age:.1f}d old; keeping regime match ahead of "
                    "opposite-regime fallbacks"
                )

            results["intraday_candidate_order"] = [name for name, _ in candidate_models]

            if not prediction_models:
                results["allow_intraday"] = False
                results["intraday_prob"] = None
                results["intraday_quality_score"] = 0.0
                if is_extended_session:
                    results["intraday_probability_status"] = (
                        "no_validated_or_fresh_diagnostic_model"
                    )
            else:
                if symU == "SPY":
                    benchmark_intra = df_intra_resampled
                else:
                    benchmark_intra = fetch_intraday_history(
                        "SPY",
                        lookback_minutes=best_lb,
                        interval=resample_to,
                        include_extended_hours=is_extended_session,
                    )
                df_feat_intra = build_intraday_features(
                    df_intra_resampled,
                    benchmark_df=benchmark_intra,
                )
                if df_feat_intra is None or df_feat_intra.empty:
                    results["allow_intraday"] = False
                    results["intraday_prob"] = None
                    results["intraday_quality_score"] = 0.0
                    results["intraday_probability_status"] = (
                        "diagnostic_feature_build_failed"
                        if diagnostic_only
                        else "feature_build_failed"
                    )
                else:
                    latest_context = df_feat_intra.iloc[-1]
                    for context_name in (
                        "vwap_distance",
                        "vwap_slope_2",
                        "vwap_slope_4",
                        "above_vwap",
                        "volume_time_ratio",
                        "session_volume_pace",
                        "relative_return_1",
                        "relative_return_2",
                        "relative_return_4",
                        "relative_strength_accel",
                        "relative_session_return",
                        "relative_vwap_strength",
                        "relative_market_available",
                    ):
                        if context_name in latest_context.index:
                            value = latest_context.get(context_name)
                            results[context_name] = (
                                float(value) if pd.notna(value) else None
                            )
                    if results.get("intraday_momentum_source") == "yfinance_1m_prepost":
                        # Never carry a stale Alpaca relative-return value into
                        # a Yahoo-based pyramid confirmation.
                        results["relative_return_4"] = yahoo_relative_return_4
                        results["relative_market_available"] = int(
                            yahoo_relative_return_4 is not None
                        )
                        results["relative_return_4_source"] = (
                            yahoo_relative_return_4_source
                            or "yfinance_1m_prepost_unavailable"
                        )
                    else:
                        results["relative_return_4_source"] = "intraday_features"
                    if is_extended_session:
                        # The regular-session volume baseline is undefined
                        # pre/post market. Treat it as neutral, never as a
                        # measured Yahoo/Alpaca volume ratio.
                        results["volume_time_ratio"] = 1.0
                        results["volume_time_ratio_source"] = (
                            "extended_session_neutral"
                        )
                    else:
                        results["volume_time_ratio_source"] = "intraday_features"
                    intraday_prediction = None
                    model_used = None

                    for cand_name, cand_model in prediction_models:
                        try:
                            pred = predict_from_model(cand_model, df_feat_intra)
                            if pred is not None:
                                intraday_prediction = pred
                                model_used = cand_name
                                break
                        except Exception as e:
                            print(
                                f"[WARN] Intraday fallback skipped {symU}/{cand_name}: {e}"
                            )
                            continue

                    if intraday_prediction is None:
                        results["allow_intraday"] = False
                        results["intraday_prob"] = None
                        results["intraday_quality_score"] = 0.0
                        if diagnostic_only:
                            results["intraday_probability_status"] = (
                                "diagnostic_prediction_failed"
                            )
                    elif diagnostic_only:
                        results["intraday_diagnostic_model_used"] = model_used
                        results["intraday_diagnostic_prediction"] = intraday_prediction
                        results["intraday_diagnostic_signal"] = get_signal_details(
                            intraday_prediction
                        )
                        # Use only its probability in the blend. Keep this
                        # unvalidated artifact out of active-model metadata
                        # (thresholds, movement gates, and promotion monitoring).
                        results["intraday_probability_status"] = (
                            "unvalidated_extended_hours_blend"
                        )
                        results["allow_intraday"] = True
                        results["intraday_quality_score"] = min(
                            1.0, len(df_intra_resampled) / 120.0
                        )
                    else:
                        results["intraday_model_used"] = model_used
                        results["intraday_prediction"] = intraday_prediction
                        results["intraday_signal"] = get_signal_details(
                            intraday_prediction
                        )
                        results["intraday_probability_status"] = "validated_model"
                    results["intraday_candidate_order"] = [
                        name for name, _ in candidate_models
                    ]
                    results["intraday_diagnostic_candidate_order"] = [
                        name for name, _ in diagnostic_only_models
                    ]
                    ip = get_final_prob(intraday_prediction)
                    if ip is not None:
                        results["intraday_prob"] = float(ip)
                        if diagnostic_only:
                            print(
                                f"[EXTENDED-HOURS UNVALIDATED BLEND] {symU}: "
                                f"model={model_used} probability={float(ip):.4f}; "
                                "included in the normal adaptive daily/intraday blend"
                            )
                        else:
                            results["intraday_quality_score"] = min(
                                1.0, len(df_intra_resampled) / 120.0
                            )
                            try:
                                current_price = results.get("price") or fetch_latest_price(
                                    symU
                                )
                                if (
                                    current_price
                                    and float(current_price) > 0
                                    and results.get("intraday_bar_fresh", False)
                                ):
                                    log_prediction(
                                        symbol=symU,
                                        mode=model_used,
                                        predicted_prob=float(ip),
                                        price=float(current_price),
                                        prediction_details=(
                                            intraday_prediction
                                            if isinstance(intraday_prediction, dict)
                                            else None
                                        ),
                                    )
                            except Exception as e:
                                print(f"[WARN] Could not log prediction: {e}")
    except Exception as e:
        print(f"[ERROR] Intraday prediction error: {e}")
        traceback.print_exc()
        results["allow_intraday"] = False
        results["intraday_prob"] = None
        results["intraday_quality_score"] = 0.0
        results["intraday_probability_status"] = "inference_error"

    # Intraday probability is trained on 15-minute features. A fresh 1-minute
    # price used by momentum/limit-price checks does not make stale 15-minute
    # model features current. Keep the old probability visible for diagnosis,
    # but exclude it from the blended signal when its source bar is stale.
    if results.get("intraday_prob") is not None and not results.get(
        "intraday_bar_fresh", False
    ):
        results["allow_intraday"] = False
        results["intraday_quality_score"] = 0.0
        if results.get("intraday_probability_status") == "validated_model":
            results["intraday_probability_status"] = "stale_bar_excluded"
        print(
            f"[INTRADAY MODEL DATA] {symU}: stale 15-minute features; "
            "intraday probability excluded from blend"
        )

    if results.get("price") is None:
        try:
            p = fetch_latest_price(symU)
            if p is not None and float(p) > 0:
                results["price"] = float(p)
        except Exception:
            pass

    # ===============================
    # FINAL PROBABILITY + WEIGHT LOGIC
    # ===============================
    dp = results["daily_prob"]
    ip = results["intraday_prob"]

    # --- 1) Start with bar-count quality (what you already had) ---
    # At this point, results["intraday_quality_score"] is q_bars = min(1.0, len(df_intra_resampled) / 120.0)
    q_bars = float(results.get("intraday_quality_score", 0.0) or 0.0)

    # --- 2) Compute alignment-based factor q_align ---
    q_align = 1.0
    if dp is not None and ip is not None:
        try:
            dp_f = float(dp)
            ip_f = float(ip)
            delta = ip_f - dp_f

            # Max disagreement you are willing to tolerate before alignment goes to zero
            max_delta = 0.20  # 20 percentage points; tweak if needed

            # Alignment factor: 1 when delta=0, 0 when |delta|>=max_delta (before flooring)
            align = 1.0 - min(1.0, abs(delta) / max_delta)

            # Optional floor: don't fully kill intraday unless disagreement is huge
            # e.g., allow at least 0.3 alignment when |delta| < max_delta
            q_align = max(0.3, align)
        except Exception:
            q_align = 1.0

    # --- 3) Combine bar-count and alignment into overall quality q ---
    q = q_bars * q_align
    results["intraday_quality_score"] = q  # now combined quality

    # --- 4) Compute base weight using combined quality ---
    weight = float(intraday_weight)
    if not results["allow_intraday"] or ip is None:
        weight = 0.0
    else:
        weight = float(intraday_weight) * q

    # --- 5) Contradiction logic (unchanged, but now based on dp/ip) ---
    if dp is not None and ip is not None:
        contradiction = detect_contradiction(
            dp, ip, momentum=mom1h, symbol=symU, verbose=True
        )
        results["contradiction_detected"] = contradiction["contradiction"]
        results["contradiction_reason"] = contradiction["reason"]
        results["contradiction_severity"] = contradiction["severity"]
        if contradiction["contradiction"]:
            weight = 0.0
    else:
        results["contradiction_detected"] = False

    results["intraday_weight_used"] = weight

    # --- 6) Final mixing (unchanged, just uses new weight) ---
    intraday_probability_usable = ip is not None and results.get(
        "allow_intraday", False
    )
    if dp is None and not intraday_probability_usable:
        results["final_prob"] = None
    elif dp is None:
        results["final_prob"] = float(ip)
    elif not intraday_probability_usable or weight <= 0.0:
        results["final_prob"] = float(dp)
    else:
        raw_final = float(weight * ip + (1 - weight) * dp)
        results["final_prob_raw"] = raw_final
        results["final_prob"] = vix_market_regime_filter(
            raw_final, vix_level=vix_level, symbol=symU
        )

    results["final_signal"] = get_signal_details(results["final_prob"])
    print(
        f"[DEBUG] compute_signals({symU}) -> "
        f"dp={results['daily_prob']} ip={results['intraday_prob']} "
        f"fp={results['final_prob']} w={results.get('intraday_weight_used', results.get('intraday_weight'))}"
    )
    return results


# ---------------------------------------------------------
# COMPATIBILITY WRAPPER
# ---------------------------------------------------------
def predict_next(
    symbol, lookback_minutes=60, intraday_weight=INTRADAY_WEIGHT, resample_to="15min"
):
    sig = compute_signals(symbol, lookback_minutes, intraday_weight, resample_to)
    return sig.get("final_prob")


# ---------------------------------------------------------
# WALK-FORWARD THRESHOLD OPTIMIZATION
# ---------------------------------------------------------
def optimize_binary_decision_threshold(
    y_true,
    y_prob,
    metric: str = "f1",
    min_threshold: float = 0.35,
    max_threshold: float = 0.65,
    step: float = 0.01,
):
    """
    Optimize classification threshold on a calibration window only.
    Supported metrics: f1, precision, recall, balanced_accuracy.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length")
    if len(np.unique(y_true)) < 2:
        raise ValueError("Threshold optimization requires at least 2 classes")

    best_threshold = 0.5
    best_score = -np.inf
    results = []

    thresholds = np.arange(min_threshold, max_threshold + 1e-9, step)
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)

        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, y_pred, zero_division=0)
        elif metric == "balanced_accuracy":
            tp = np.sum((y_true == 1) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            score = 0.5 * (tpr + tnr)
        else:
            raise ValueError(f"Unsupported threshold metric: {metric}")

        results.append({"threshold": float(thr), "score": float(score)})
        if score > best_score:
            best_score = score
            best_threshold = float(thr)

    return {
        "best_threshold": best_threshold,
        "best_score": float(best_score),
        "metric": metric,
        "grid": results,
    }


# ---------------------------------------------------------
# POSITION SIZING FROM PROBABILITY EDGE
# ---------------------------------------------------------
def probability_to_position_size(
    prob: float,
    decision_threshold: float = 0.5,
    max_position: float = 1.0,
    dead_zone: float = 0.03,
):
    """
    Map probability edge into a bounded position size in [-max_position, +max_position].
    Small edges inside the dead zone map to 0.
    """
    if prob is None or not np.isfinite(prob):
        return 0.0

    prob = float(prob)
    edge = prob - float(decision_threshold)

    if abs(edge) <= float(dead_zone):
        return 0.0

    signed_edge = np.sign(edge)
    effective = max(0.0, abs(edge) - float(dead_zone))
    scale = max(1e-9, 0.5 - float(dead_zone))
    size = signed_edge * min(
        float(max_position), effective / scale * float(max_position)
    )
    return float(size)


def extract_feature_importance(base_model, feature_names):
    """
    Extract compact feature importance report from underlying XGBoost model.
    Supports calibrated artifacts by using base_model when available.
    """
    if base_model is None or not hasattr(base_model, "feature_importances_"):
        return {
            "top_features": [],
            "importance_type": "feature_importances_",
        }

    importances = np.asarray(base_model.feature_importances_, dtype=float)
    if len(importances) != len(feature_names):
        return {
            "top_features": [],
            "importance_type": "feature_importances_",
        }

    pairs = []
    for feat, imp in zip(feature_names, importances):
        pairs.append(
            {
                "feature": str(feat),
                "importance": float(imp),
            }
        )

    pairs = sorted(pairs, key=lambda x: x["importance"], reverse=True)
    return {
        "top_features": pairs[:25],
        "importance_type": "feature_importances_",
    }
