# model_xgb.py
import os
import traceback
import warnings
from datetime import datetime
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
)
from predictive_model.features import build_daily_features, build_intraday_features
from predictive_model.model_monitor import evaluate_predictions, log_prediction
from predictive_model.target_labels import backtest_threshold, create_target_label
from predictive_model.trading_metrics import (
    calculate_financial_metrics,
    print_trading_report,
)

from config.config import (
    INTRADAY_MOM_TRIG,
    INTRADAY_REGIME_OVERRIDES,
    INTRADAY_VOL_TRIG,
    INTRADAY_WEIGHT,
    MIN_INTRADAY_BARS_FOR_FEATURES,
)

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
CALIBRATOR_DIR = Path("models") / "calibrators"
_calibrator_cache = {}

warnings.filterwarnings("ignore", message=".*cv='prefit'.*")


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


def _clean_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean DataFrame to ensure all columns are proper 1D Series.
    This fixes issues with XGBoost complaining about DataFrame columns.
    """
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(x) for x in col if str(x) != ""]).strip("_") for col in df.columns]

    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]

    for col in list(df.columns):
        if isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]

        if hasattr(df[col], "values") and getattr(df[col].values, "ndim", 1) > 1:
            vals = df[col].values
            if vals.shape[0] == len(df):
                df[col] = pd.Series(vals[:, 0], index=df.index)

        df[col] = pd.to_numeric(df[col], errors="coerce")

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
    aligned = _clean_feature_dataframe(aligned)

    unnamed_cols = [
        col for col in aligned.columns if col == "" or pd.isna(col) or str(col).strip() == ""
    ]
    if unnamed_cols:
        raise ValueError(f"Unnamed columns present in inference features: {unnamed_cols[:10]}")

    non_numeric = aligned.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(f"Non-numeric features present: {non_numeric[:10]}")

    if aligned.isna().any().any():
        nan_cols = aligned.columns[aligned.isna().any()].tolist()
        raise ValueError(f"NaNs present in inference features: {nan_cols[:10]}")

    return aligned


def _time_ordered_train_cal_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_frac: float = 0.70,
    cal_frac: float = 0.10,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")

    n = len(X)
    # Tier 1: very small datasets → still refuse
    if n < 40:
        raise ValueError(f"Not enough rows for any reasonable split: n={n}")

    # Tier 2: small but usable (e.g., 40–99 rows) → simple 80/20 split, no separate cal window
    if n < 100:
        # Use 80% train, 20% test; reuse test as "cal" to keep API stable
        split_idx = int(n * 0.8)
        if split_idx <= 0 or split_idx >= n:
            raise ValueError(f"Not enough data to split train/test for n={n}")

        X_train = X.iloc[:split_idx].copy()
        y_train = y.iloc[:split_idx].copy()
        X_test = X.iloc[split_idx:].copy()
        y_test = y.iloc[split_idx:].copy()

        # For small-n case, treat test as calibration too
        X_cal = X_test.copy()
        y_cal = y_test.copy()

        return X_train, y_train, X_cal, y_cal, X_test, y_test

    # Tier 3: normal path (n >= 100) → proper train/cal/test
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


def _build_thresholded_binary_target(df_feat: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    Binary target with a no-trade band to reduce micro-noise labels.
    """
    df_feat = df_feat.copy()
    if "Close" not in df_feat.columns:
        raise ValueError("Close column is required for target creation")

    next_ret = df_feat["Close"].shift(-1) / df_feat["Close"] - 1.0
    min_move = 0.002 if mode == "daily" else 0.0008

    df_feat["target"] = np.where(
        next_ret > min_move,
        1,
        np.where(next_ret < -min_move, 0, np.nan),
    )
    df_feat = df_feat.dropna(subset=["target"]).copy()
    df_feat["target"] = df_feat["target"].astype(int)
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


def _add_intraday_regime_cols(dffeat: pd.DataFrame) -> pd.DataFrame:
    """
    Add regime columns with CONSISTENT names: mom_12_abs, vol_12.
    """
    dffeat = dffeat.copy()
    try:
        close = dffeat["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = pd.to_numeric(close, errors="coerce")

        ret12 = close.pct_change(12)
        dffeat["ret_12"] = ret12
        dffeat["mom_12_abs"] = ret12.abs()
        dffeat["vol_12"] = close.pct_change().rolling(12).std()
    except Exception as e:
        print(f"[REGIME COLS] Error adding regime features: {e}")
    return dffeat


def _filter_intraday_rows_by_mode(df_feat: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    Filters intraday feature rows by regime.
    Uses percentile-based thresholds computed only from the current training frame.
    """
    df_feat = _add_intraday_regime_cols(df_feat)

    df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
    df_feat = df_feat.dropna(subset=["mom_12_abs", "vol_12"])

    if df_feat.empty:
        print(f"[WARN] Empty dataframe after regime column addition for mode={mode}")
        return df_feat

    mom_p60 = df_feat["mom_12_abs"].quantile(0.60)
    vol_p60 = df_feat["vol_12"].quantile(0.60)
    mom_p30 = df_feat["mom_12_abs"].quantile(0.30)
    vol_p30 = df_feat["vol_12"].quantile(0.30)

    print(
        f"[REGIME] {mode}: mom_p30={mom_p30:.4f} mom_p60={mom_p60:.4f} "
        f"vol_p30={vol_p30:.5f} vol_p60={vol_p60:.5f}"
    )

    if mode == "intraday_mom":
        filtered = df_feat[
            (df_feat["mom_12_abs"] >= mom_p60) |
            (df_feat["vol_12"] >= vol_p60)
        ]
        print(
            f"[REGIME] intraday_mom: {len(df_feat)} -> {len(filtered)} rows "
            f"({len(filtered) / len(df_feat) * 100:.1f}%)"
        )
        return filtered

    if mode == "intraday_mr":
        filtered = df_feat[
            (df_feat["mom_12_abs"] < mom_p30) &
            (df_feat["vol_12"] < vol_p30)
        ]
        print(
            f"[REGIME] intraday_mr: {len(df_feat)} -> {len(filtered)} rows "
            f"({len(filtered) / len(df_feat) * 100:.1f}%)"
        )
        return filtered

    return df_feat


# ---------------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------------
def train_model(df: pd.DataFrame, symbol: str, mode: str = "daily", use_multiclass: bool = False):
    """
    Train XGB model with chronological train/calibration/test split.
    Calibration is performed only on the calibration window, never on the final test window.
    """
    from class_balancing import analyze_class_balance, calculate_scale_pos_weight
    from target_engineering import create_multiclass_target, print_target_distribution
    from predictive_model.feature_selection import select_features_with_shap

    df = df.copy()

    df = df.copy()

    print(f"\n[FEATURES] Building engineered features for {symbol}/{mode}...")
    if mode == "daily":
        df_feat = build_daily_features(df)
    elif mode in ("intraday", "intraday_mr", "intraday_mom"):
        df_feat = build_intraday_features(df)
        if mode in ("intraday_mr", "intraday_mom"):
            df_feat = _filter_intraday_rows_by_mode(df_feat, mode=mode)
            target_col = "target"
            num_classes = 2
        for c in ["ret_12", "mom_12_abs", "vol_12"]:
            if c in df_feat.columns:
                df_feat = df_feat.drop(columns=[c])
    else:
        raise ValueError("mode must be 'daily' or one of 'intraday', 'intraday_mr', 'intraday_mom'")

    print(f"[FEATURES] Built {len(df_feat.columns)} total columns after feature engineering")

    print("\n[CLEANUP] Cleaning DataFrame columns...")
    df_feat = _clean_feature_dataframe(df_feat)
    print(f"[CLEANUP] Rows after clean: {len(df_feat)} | Cols: {len(df_feat.columns)}")

    if use_multiclass:
        print("\n[TARGET] Creating multi-class target...")
        thresholds = (-0.015, -0.003, 0.003, 0.015) if mode == "daily" else (-0.008, -0.002, 0.002, 0.008)
        df_feat = create_multiclass_target(df_feat, forward_periods=1, thresholds=thresholds)
        target_col = "Target_multiclass"
        num_classes = 5
        objective = "multi:softprob"
        eval_metric = "mlogloss"
        print_target_distribution(df_feat, target_col)
    else:
        print("\n[TARGET] Creating thresholded binary target...")
        df_feat = _build_thresholded_binary_target(df_feat, mode=mode)
        target_col = "target"
        num_classes = 2
        objective = "binary:logistic"
        eval_metric = "logloss"

    df_feat = df_feat.dropna().copy()
    if df_feat.empty:
        raise ValueError(f"No rows left after feature engineering for {symbol}/{mode}")

    class_counts = df_feat[target_col].value_counts()
    unique_classes = sorted(class_counts.index.tolist())
    print(f"\n[CLASS CHECK] Found {len(unique_classes)} unique classes: {unique_classes}")
    for cls in unique_classes:
        count = int(class_counts[cls])
        print(f" Class {cls}: {count} samples ({count / len(df_feat) * 100:.1f}%)")

    min_samples_per_class = 3 if mode in ("intraday_mr", "intraday_mom") else 5
    valid_classes = [cls for cls in unique_classes if class_counts[cls] >= min_samples_per_class]
    if len(valid_classes) < 2:
        raise ValueError(
            f"Insufficient class diversity for {symbol}/{mode}. "
            f"Class counts: {class_counts.to_dict()}"
        )

    if use_multiclass and len(valid_classes) < num_classes:
        print(f"\n[CLASS REMAP] Only {len(valid_classes)}/{num_classes} classes have enough samples")
        df_feat = df_feat[df_feat[target_col].isin(valid_classes)].copy()
        class_mapping = {old: new for new, old in enumerate(sorted(valid_classes))}
        df_feat[target_col] = df_feat[target_col].map(class_mapping)
        num_classes = len(valid_classes)
        print(f"[CLASS REMAP] Mapping: {class_mapping}")

    print("\n[FEATURES] Preparing feature matrix...")
    exclude_cols = [target_col, "Close", "Open", "High", "Low", "Volume"]
    feature_cols = [col for col in df_feat.columns if col not in exclude_cols]

    X = df_feat[feature_cols].copy()
    y = pd.Series(df_feat[target_col], dtype=int)

    unnamed_cols = [col for col in X.columns if col == "" or pd.isna(col) or str(col).strip() == ""]
    if unnamed_cols:
        raise ValueError(f"Unnamed feature columns present: {unnamed_cols[:10]}")

    X = _clean_feature_dataframe(X)

    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(f"Non-numeric features present after cleaning: {non_numeric[:10]}")

    all_nan_cols = X.columns[X.isna().all()].tolist()
    if all_nan_cols:
        print(f"[VALIDATION] Dropping {len(all_nan_cols)} all-NaN columns")
        X = X.drop(columns=all_nan_cols)

    if X.isna().any().any():
        nan_cols = X.columns[X.isna().any()].tolist()
        raise ValueError(f"NaNs remain in training matrix. Columns: {nan_cols[:10]}")

    feature_list = list(X.columns)
    if not feature_list:
        raise ValueError(f"No valid features remaining for {symbol}/{mode}")

    print(f"[FEATURES] Final feature count: {len(feature_list)}")
    print(f"[SANITY] Feature matrix shape: {X.shape} | Target shape: {y.shape}")

    full_feature_schema = {
        "features": feature_list,
        "feature_count": len(feature_list),
    }

    X_train, y_train, X_cal, y_cal, X_test, y_test = _time_ordered_train_cal_test_split(X, y)

    print(f"\n[SPLIT] Train={len(X_train)} Cal={len(X_cal)} Test={len(X_test)}")
    print(f"[SPLIT] Train classes: {y_train.value_counts().to_dict()}")
    print(f"[SPLIT] Cal classes: {y_cal.value_counts().to_dict()}")
    print(f"[SPLIT] Test classes: {y_test.value_counts().to_dict()}")

    if len(sorted(y_train.unique())) < 2:
        raise ValueError("Train set has fewer than 2 classes.")
    if num_classes == 2 and len(sorted(y_cal.unique())) < 2:
        raise ValueError("Calibration set has fewer than 2 classes.")
    if len(sorted(y_test.unique())) < 2:
        print("[WARN] Final test set has one class; ROC AUC and some calibration metrics may be unavailable.")

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
                print(f"[CLASS WEIGHT] Applying scale_pos_weight={scale_pos_weight:.3f}")
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
        "n_jobs": -1,
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

    if num_classes == 2 and len(X_train.columns) >= 25 and len(X_cal) >= 50:
        try:
            print(f"\n[FEATURE SELECTION] Running calibration-window SHAP selection...")
            top_features, _shap_values, X_train_sel, X_cal_sel, fs_meta = select_features_with_shap(
                model=model,
                X_train=X_train,
                X_cal=X_cal,
                top_n=None,
                plot=False,
                symbol=symbol,
                mode=mode,
            )
            if len(top_features) >= 10:
                print(f"[FEATURE SELECTION] Retraining on {len(top_features)} selected features")
                X_test = X_test[top_features].copy()
                X_train = X_train_sel
                X_cal = X_cal_sel
                selected_features = top_features
                selected_feature_metadata = fs_meta
                feature_list = top_features

                model = XGBClassifier(**params)
                model.fit(X_train, y_train, eval_set=[(X_cal, y_cal)], verbose=False)
                print(f"[FEATURE SELECTION] Retrained selected-feature model")
            else:
                print(f"[FEATURE SELECTION] Too few features selected; keeping full schema")
        except Exception as e:
            # print(f"[FEATURE SELECTION] Skipping due to error: {e}")
            print("[FEATURE SELECTION] Skipping SHAP feature selection due to XGBoost shape mismatch.")

    base_model = model
    final_model = model
    calibrated = False

    calibration_method = "sigmoid" if num_classes == 2 else "isotonic"
    print(f"\n[CALIBRATION] Applying {calibration_method} calibration on calibration window only")
    try:
        calibrated_model = CalibratedClassifierCV(
            model,
            method=calibration_method,
            cv="prefit",
            n_jobs=-1,
        )
        calibrated_model.fit(X_cal, y_cal)
        final_model = calibrated_model
        calibrated = True
        print(f"[CALIBRATION] Successfully calibrated using {calibration_method}")
    except Exception as e:
        print(f"[CALIBRATION] Failed to calibrate: {e}")
        print("[CALIBRATION] Using uncalibrated model as fallback")
        final_model = model

    print("\n[EVALUATION] Computing metrics on final test set...")
    metrics = {}

    try:
        if num_classes == 2:
            cal_pred_proba = final_model.predict_proba(X_cal)[:, 1]
            threshold_opt = optimize_binary_decision_threshold(
                y_cal,
                cal_pred_proba,
                metric="f1",
                min_threshold=0.35,
                max_threshold=0.65,
                step=0.01,
            )
            best_threshold = float(threshold_opt["best_threshold"])
            y_pred_proba = final_model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba >= best_threshold).astype(int)

            metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
            metrics["logloss"] = float(log_loss(y_test, y_pred_proba, labels=[0, 1])) if len(np.unique(y_test)) >= 2 else None
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_pred_proba)) if len(np.unique(y_test)) >= 2 else None
            metrics["precision"] = float(precision_score(y_test, y_pred, zero_division=0))
            metrics["recall"] = float(recall_score(y_test, y_pred, zero_division=0))
            metrics["f1"] = float(f1_score(y_test, y_pred, zero_division=0))
            metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()
            metrics["decision_threshold"] = float(best_threshold)
            metrics["threshold_metric"] = threshold_opt["metric"]
            metrics["threshold_score"] = float(threshold_opt["best_score"])
            metrics["threshold_grid"] = threshold_opt["grid"]

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
            metrics["logloss"] = float(log_loss(y_test, y_pred_proba, labels=all_classes))
            metrics["precision_weighted"] = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
            metrics["recall_weighted"] = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
            metrics["f1_weighted"] = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
            metrics["precision_per_class"] = precision_score(
                y_test, y_pred, average=None, zero_division=0, labels=all_classes
            ).tolist()
            metrics["recall_per_class"] = recall_score(
                y_test, y_pred, average=None, zero_division=0, labels=all_classes
            ).tolist()
            metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred, labels=all_classes).tolist()

            from sklearn.preprocessing import label_binarize
            y_test_bin = label_binarize(y_test, classes=range(num_classes))
            metrics["brier_score"] = float(np.mean(np.sum((y_test_bin - y_pred_proba) ** 2, axis=1)))

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

    return {
        "model": final_model,
        "base_model": base_model,
        "features": feature_list,
        "feature_schema": {
            "features": feature_list,
            "feature_count": len(feature_list),
            "target_type": "multiclass" if use_multiclass else "binary",
        },
        "full_feature_schema": full_feature_schema,
        "selected_feature_metadata": selected_feature_metadata,
        "split_metadata": {
            "train_samples": len(X_train),
            "calibration_samples": len(X_cal),
            "test_samples": len(X_test),
        },
        "metrics": metrics,
        "decision_threshold": metrics.get("decision_threshold", 0.5),
        "threshold_optimization": {
            "metric": metrics.get("threshold_metric"),
            "score": metrics.get("threshold_score"),
            "grid": metrics.get("threshold_grid", []),
        },
        "feature_importance": feature_importance,
        "trained_at": datetime.now().isoformat(),
        "symbol": symbol,
        "mode": mode,
        "calibrated": calibrated,
        "num_classes": num_classes,
        "target_type": "multiclass" if use_multiclass else "binary",
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
        feat_cols = model_dict.get("feature_schema", {}).get("features", model_dict["features"])
        num_classes = model_dict.get("num_classes", 2)
        target_type = model_dict.get("target_type", "binary")

        latest = df_features.tail(1).copy()
        latest = _validate_exact_feature_schema(latest, feat_cols, allow_extra=True)
        X = latest.loc[:, feat_cols].copy()

        proba = model.predict_proba(X)[0]

        if num_classes == 2 or target_type == "binary":
            prob = float(proba[1])
            decision_threshold = float(model_dict.get("decision_threshold", 0.5))

            # NEW: Platt calibration if available
            try:
                sym = str(model_dict.get("symbol") or "").upper()
                mode = str(model_dict.get("mode") or "")
                platt = load_platt_calibrator(sym, mode) if sym and mode else None
            except Exception:
                platt = None

            if platt is not None:
                try:
                    prob = float(platt.predict_proba([[prob]])[0, 1])
                except Exception:
                    # Fail safe: if calibration breaks, keep raw prob
                    pass
                
            return {
                "final_prob": prob,
                "decision_threshold": decision_threshold,
                "position_size": probability_to_position_size(prob, decision_threshold=decision_threshold),
                "top_features": model_dict.get("feature_importance", {}).get("top_features", [])[:5],
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
        bullish_prob = float(sum(proba[i] for i in range(actual_num_classes) if i >= mid_point))
        bearish_prob = float(sum(proba[i] for i in range(actual_num_classes) if i < mid_point))
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
            "class_breakdown": {f"class_{i}": float(proba[i]) for i in range(actual_num_classes)},
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
        details += f" | [↓↓:{breakdown.get('strong_down', 0):.0%} "
        details += f"↓:{breakdown.get('weak_down', 0):.0%} "
        details += f"→:{breakdown.get('flat', 0):.0%} "
        details += f"↑:{breakdown.get('weak_up', 0):.0%} "
        details += f"↑↑:{breakdown.get('strong_up', 0):.0%}]"
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
            vix_close = vix_close.iloc[:, 0] if isinstance(vix_close, pd.DataFrame) else vix_close
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
        print(f"[VIX FILTER] {symbol} VIX={vix_level:.1f} -> scale={scale:.2f} | prob {prob:.3f} -> {adjusted:.3f}")
        return float(adjusted)

    return float(prob)


# ---------------------------------------------------------
# CONTRADICTION DETECTION
# ---------------------------------------------------------
def detect_contradiction(daily_prob, intraday_prob, momentum=None, symbol="", verbose=True):
    """Detect contradictions between daily and intraday signals."""
    if daily_prob is None or intraday_prob is None:
        return {"contradiction": False, "reason": "Missing data", "severity": "low"}

    bullish_threshold = 0.55
    bearish_threshold = 0.45

    daily_bullish = daily_prob > bullish_threshold
    daily_bearish = daily_prob < bearish_threshold
    intraday_bullish = intraday_prob > bullish_threshold
    intraday_bearish = intraday_prob < bearish_threshold

    models_disagree = (
        (daily_bullish and intraday_bearish) or
        (daily_bearish and intraday_bullish)
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

    extreme_disagree = (
        (daily_prob > 0.65 and intraday_prob < 0.35) or
        (daily_prob < 0.35 and intraday_prob > 0.65)
    )

    if extreme_disagree:
        reason = f"{symbol}: Extreme disagreement - daily={daily_prob:.2f}, intraday={intraday_prob:.2f}"
        if verbose:
            print(f"[CONTRADICTION] {reason}")
        return {"contradiction": True, "reason": reason, "severity": "medium"}

    return {"contradiction": False, "reason": "Mild disagreement", "severity": "low"}


# ---------------------------------------------------------
# COMPUTE SIGNALS (kept compatible; inference now uses strict schema)
# ---------------------------------------------------------
def compute_signals(symbol, lookback_minutes=60, intraday_weight=INTRADAY_WEIGHT, resample_to="15min"):
    results = {
        "daily_prob": None,
        "intraday_prob": None,
        "final_prob": None,
        "price": None,
        "intraday_weight": float(intraday_weight),
        "allow_intraday": True,
        "intraday_rows": 0,
        "intraday_before": 0,
        "daily_rows": 0,
        "used_lookback_minutes": lookback_minutes,
        "intraday_quality_score": 0.0,
        "intraday_model_used": None,
        "intraday_vol": None,
        "intraday_mom": None,
        "intraday_regime": None,
        "daily_prediction": None,
        "intraday_prediction": None,
        "daily_signal": None,
        "intraday_signal": None,
    }

    symU = str(symbol).upper().strip()

    model_daily = load_model(symU, mode="daily")
    model_intra_mr = load_model(symU, mode="intraday_mr")
    model_intra_mom = load_model(symU, mode="intraday_mom")
    model_intraday_legacy = load_model(symU, mode="intraday")

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
                            prediction_details=daily_prediction if isinstance(daily_prediction, dict) else None,
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
    mom1h = None
    close = pd.Series(dtype=float)

    try:
        candidate_lookbacks = sorted([
            lookback_minutes,
            max(lookback_minutes, 600),
            max(lookback_minutes, 900),
            max(lookback_minutes, 1200),
        ])

        best_df = None
        best_len = 0
        best_lb = lookback_minutes
        best_before = 0
        rt = str(resample_to).lower().replace("mins", "min").replace(" ", "")

        for lb in candidate_lookbacks:
            df_raw = fetch_intraday_history(symU, lookback_minutes=lb, interval=resample_to)
            if df_raw is None or df_raw.empty:
                continue

            before = len(df_raw)
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
            if n_res >= 80:
                break

        df_intra_resampled = best_df
        results["intraday_before"] = best_before
        results["used_lookback_minutes"] = best_lb
        results["intraday_rows"] = 0 if df_intra_resampled is None else len(df_intra_resampled)

        if df_intra_resampled is None or len(df_intra_resampled) < MIN_INTRADAY_BARS_FOR_FEATURES:
            results["allow_intraday"] = False
            results["intraday_prob"] = None
            results["intraday_quality_score"] = 0.0
        else:
            close = df_intra_resampled["Close"]
            close = close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close
            close = close.dropna()

            if len(close) > 0:
                results["price"] = float(close.iloc[-1])
            if len(close) >= 12:
                vol = close.pct_change().rolling(12).std().iloc[-1]
                results["intraday_vol"] = float(vol) if not pd.isna(vol) else None
            if len(close) >= 4:
                mom1h = close.pct_change(4).iloc[-1]
                results["intraday_mom"] = float(mom1h) if not pd.isna(mom1h) else None

            try:
                adaptive = get_adaptive_regime_thresholds(symU, lookback_days=30, percentile=0.70)
                MOMTRIG = float(adaptive["mom_trig"])
                VOLTRIG = float(adaptive["vol_trig"])
                print(f"[ADAPTIVE] {symU} using adaptive: mom={MOMTRIG:.4f} vol={VOLTRIG:.5f}")
            except Exception:
                MOMTRIG = float(INTRADAY_MOM_TRIG)
                VOLTRIG = float(INTRADAY_VOL_TRIG)
                ovr = (INTRADAY_REGIME_OVERRIDES or {}).get(symU)
                if ovr:
                    MOMTRIG = float(ovr.get("mom_trig", MOMTRIG))
                    VOLTRIG = float(ovr.get("vol_trig", VOLTRIG))
                print(f"[CONFIG] {symU} using config: mom={MOMTRIG:.4f} vol={VOLTRIG:.5f}")

            candidate_models = []

            if vol is not None and mom1h is not None:
                if abs(mom1h) >= MOMTRIG or vol >= VOLTRIG:
                    results["intraday_regime"] = "mom"
                    candidate_models = [
                        ("intraday_mom", model_intra_mom),
                        ("intraday_mr", model_intra_mr),
                        ("intraday", model_intraday_legacy),
                    ]
                else:
                    results["intraday_regime"] = "mr"
                    candidate_models = [
                        ("intraday_mr", model_intra_mr),
                        ("intraday_mom", model_intra_mom),
                        ("intraday", model_intraday_legacy),
                    ]
            else:
                candidate_models = [
                    ("intraday_mom", model_intra_mom),
                    ("intraday_mr", model_intra_mr),
                    ("intraday", model_intraday_legacy),
                ]

            candidate_models = [(name, mdl) for name, mdl in candidate_models if mdl is not None]

            if not candidate_models:
                results["allow_intraday"] = False
                results["intraday_prob"] = None
                results["intraday_quality_score"] = 0.0
            else:
                df_feat_intra = build_intraday_features(df_intra_resampled)
                if df_feat_intra is None or df_feat_intra.empty:
                    results["allow_intraday"] = False
                    results["intraday_prob"] = None
                    results["intraday_quality_score"] = 0.0
                else:
                    intraday_prediction = None
                    model_used = None

                    for cand_name, cand_model in candidate_models:
                        try:
                            pred = predict_from_model(cand_model, df_feat_intra)
                            if pred is not None:
                                intraday_prediction = pred
                                model_used = cand_name
                                break
                        except Exception as e:
                            print(f"[WARN] Intraday fallback skipped {symU}/{cand_name}: {e}")
                            continue

                    if intraday_prediction is None:
                        results["allow_intraday"] = False
                        results["intraday_prob"] = None
                        results["intraday_quality_score"] = 0.0
                    else:
                        results["intraday_model_used"] = model_used
                    results["intraday_prediction"] = intraday_prediction
                    results["intraday_model_used"] = model_used
                    results["intraday_signal"] = get_signal_details(intraday_prediction)
                    ip = get_final_prob(intraday_prediction)
                    if ip is not None:
                        results["intraday_prob"] = float(ip)
                        results["intraday_quality_score"] = min(1.0, len(df_intra_resampled) / 120.0)
                        try:
                            current_price = results.get("price") or fetch_latest_price(symU)
                            if current_price and float(current_price) > 0:
                                log_prediction(
                                    symbol=symU,
                                    mode=model_used,
                                    predicted_prob=float(ip),
                                    price=float(current_price),
                                    prediction_details=intraday_prediction if isinstance(intraday_prediction, dict) else None,
                                )
                        except Exception as e:
                            print(f"[WARN] Could not log prediction: {e}")
    except Exception as e:
        print(f"[ERROR] Intraday prediction error: {e}")
        traceback.print_exc()
        results["allow_intraday"] = False
        results["intraday_prob"] = None
        results["intraday_quality_score"] = 0.0

    if results.get("price") is None:
        try:
            p = fetch_latest_price(symU)
            if p is not None and float(p) > 0:
                results["price"] = float(p)
        except Exception:
            pass

    dp = results["daily_prob"]
    ip = results["intraday_prob"]

    # --- 1) Start from existing bar-count quality (already set earlier) ---
    # Earlier you set:
    #   results["intraday_quality_score"] = min(1.0, len(df_intra_resampled) / 120.0)
    # Here we interpret that as q_bars.
    q_bars = float(results.get("intraday_quality_score", 0.0) or 0.0)

    # --- 2) Alignment factor between daily and intraday ---
    # Measures how much the two probabilities agree.
    q_align = 1.0
    if dp is not None and ip is not None:
        try:
            dp_f = float(dp)
            ip_f = float(ip)
            delta = ip_f - dp_f  # intraday minus daily

            # Max disagreement (in probability points) where alignment decays to zero
            max_delta = 0.20  # 20 percentage points; tweak as you like

            # align = 1 when delta = 0, down toward 0 as |delta| -> max_delta
            align = 1.0 - min(1.0, abs(delta) / max_delta)

            # Optional floor so intraday isn't completely discarded for moderate disagreement
            q_align = max(0.3, align)
        except Exception:
            q_align = 1.0

    # --- 3) Combined intraday "quality" ---
    # This now encodes both bar-count (coverage) and agreement with daily.
    q = q_bars * q_align
    results["intraday_quality_score"] = q

    # --- 4) Base weight using combined quality ---
    weight = float(intraday_weight)
    if not results["allow_intraday"] or ip is None:
        weight = 0.0
    else:
        weight = float(intraday_weight) * q

    # --- 5) Contradiction logic (unchanged) ---
    if dp is not None and ip is not None:
        contradiction = detect_contradiction(dp, ip, momentum=mom1h, symbol=symU, verbose=True)
        results["contradiction_detected"] = contradiction["contradiction"]
        results["contradiction_reason"] = contradiction["reason"]
        results["contradiction_severity"] = contradiction["severity"]
        if contradiction["contradiction"]:
            weight = 0.0
    else:
        results["contradiction_detected"] = False

    # Store the actual intraday weight used (for diagnostics)
    results["intraday_weight_used"] = weight

    # --- 6) Final probability mixing + VIX filter (unchanged) ---
    if dp is None and ip is None:
        results["final_prob"] = None
    elif dp is None:
        results["final_prob"] = float(ip)
    elif ip is None or weight <= 0.0:
        results["final_prob"] = float(dp)
    else:
        raw_final = float(weight * ip + (1.0 - weight) * dp)
        results["final_prob_raw"] = raw_final
        results["final_prob"] = vix_market_regime_filter(raw_final, vix_level=vix_level, symbol=symU)

    results["final_signal"] = get_signal_details(results["final_prob"])
    return results


# ---------------------------------------------------------
# COMPATIBILITY WRAPPER
# ---------------------------------------------------------
def predict_next(symbol, lookback_minutes=60, intraday_weight=INTRADAY_WEIGHT, resample_to="15min"):
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
    size = signed_edge * min(float(max_position), effective / scale * float(max_position))
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
        pairs.append({
            "feature": str(feat),
            "importance": float(imp),
        })

    pairs = sorted(pairs, key=lambda x: x["importance"], reverse=True)
    return {
        "top_features": pairs[:25],
        "importance_type": "feature_importances_",
    }
