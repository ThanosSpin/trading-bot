# model_xgb.py
import os
from datetime import datetime
import warnings

import joblib
import numpy as np
import pandas as pd
import traceback
from xgboost import XGBClassifier

from predictive_model.features import build_daily_features, build_intraday_features
from predictive_model.data_loader import fetch_historical_data, fetch_intraday_history, fetch_latest_price
from predictive_model.target_labels import create_target_label, backtest_threshold
from predictive_model.trading_metrics import calculate_financial_metrics, print_trading_report
from predictive_model.adaptive_thresholds import get_adaptive_regime_thresholds
from predictive_model.model_monitor import log_prediction, evaluate_predictions
from sklearn.metrics import (
    accuracy_score, log_loss, roc_auc_score,
    confusion_matrix, precision_score, recall_score, f1_score,
    brier_score_loss,
)
from sklearn.calibration import CalibratedClassifierCV

from config.config import (
    INTRADAY_WEIGHT,
    MIN_INTRADAY_BARS_FOR_FEATURES,
    INTRADAY_MOM_TRIG,
    INTRADAY_VOL_TRIG,
    INTRADAY_REGIME_OVERRIDES,
)

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

warnings.filterwarnings("ignore", message=".*cv='prefit'.*")

# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _clean_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean DataFrame to ensure all columns are proper 1D Series.
    This fixes issues with XGBoost complaining about DataFrame columns.
    """
    df = df.copy()
    
    for col in df.columns:
        if isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]
        if hasattr(df[col], 'values') and df[col].values.ndim > 1:
            if df[col].values.shape[0] == len(df):
                df[col] = pd.Series(df[col].values[:, 0], index=df.index)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def normalize_feature_name(name: str) -> str:
    return name.lower().replace('_', '').replace(' ', '')


def find_matching_feature(model_feat: str, available_features: list) -> str:
    """
    Find matching feature with fuzzy logic (case-insensitive, normalized).
    Returns matched feature name or None.
    """
    if not model_feat or not available_features:
        return None

    feature_lower = str(model_feat).lower().strip()

    # STAGE 1: Exact match
    if model_feat in available_features:
        return model_feat

    # STAGE 2: Case-insensitive exact
    for feat in available_features:
        if str(feat).lower().strip() == feature_lower:
            return feat

    # STAGE 3: Normalized (no underscores/spaces)
    model_normalized = feature_lower.replace("_", "").replace(" ", "")
    for feat in available_features:
        if str(feat).lower().strip().replace("_", "").replace(" ", "") == model_normalized:
            return feat

    # STAGE 4: Partial substring match
    for feat in available_features:
        feat_lower = str(feat).lower().strip()
        if feature_lower in feat_lower or feat_lower in feature_lower:
            return feat

    # STAGE 5: Token-based (>=2 shared tokens)
    feature_tokens = set(feature_lower.split("_"))
    best_match = None
    best_score = 0
    for feat in available_features:
        feat_tokens = set(str(feat).lower().strip().split("_"))
        score = len(feature_tokens & feat_tokens)
        if score > best_score and score >= 2:
            best_score = score
            best_match = feat

    return best_match


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
        # Use consistent underscore names throughout
        dffeat["ret_12"]     = ret12
        dffeat["mom_12_abs"] = ret12.abs()
        dffeat["vol_12"]     = close.pct_change().rolling(12).std()
    except Exception as e:
        print(f"[SPY-FEATURES] Error adding regime features: {e}")
    return dffeat


def _filter_intraday_rows_by_mode(df_feat: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    Filters intraday feature rows by regime.
    Uses adaptive percentile-based thresholds.
    """
    df_feat = _add_intraday_regime_cols(df_feat)

    df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
    # FIX: column names now match what _add_intraday_regime_cols creates
    df_feat = df_feat.dropna(subset=["mom_12_abs", "vol_12"])

    if df_feat.empty:
        print(f"[WARN] Empty dataframe after regime column addition for mode={mode}")
        return df_feat

    mom_p60 = df_feat["mom_12_abs"].quantile(0.60)
    vol_p60 = df_feat["vol_12"].quantile(0.60)
    mom_p30 = df_feat["mom_12_abs"].quantile(0.30)
    vol_p30 = df_feat["vol_12"].quantile(0.30)

    print(f"[REGIME] {mode}: mom_p30={mom_p30:.4f} mom_p60={mom_p60:.4f} vol_p30={vol_p30:.5f} vol_p60={vol_p60:.5f}")

    if mode == "intraday_mom":
        filtered = df_feat[
            (df_feat["mom_12_abs"] >= mom_p60) |
            (df_feat["vol_12"] >= vol_p60)
        ]
        print(f"[REGIME] intraday_mom: {len(df_feat)} → {len(filtered)} rows ({len(filtered)/len(df_feat)*100:.1f}%)")
        return filtered

    if mode == "intraday_mr":
        filtered = df_feat[
            (df_feat["mom_12_abs"] < mom_p30) &
            (df_feat["vol_12"] < vol_p30)
        ]
        print(f"[REGIME] intraday_mr: {len(df_feat)} → {len(filtered)} rows ({len(filtered)/len(df_feat)*100:.1f}%)")
        return filtered

    return df_feat


def train_model(df: pd.DataFrame, symbol: str, mode: str = "daily", use_multiclass: bool = False):
    """
    Train XGB model.

    Args:
        df: DataFrame with OHLCV data
        symbol: Stock symbol
        mode: Training mode (daily, intraday, intraday_mr, intraday_mom)
        use_multiclass: If True, train 5-class model; if False, binary model
    """
    from indicators import add_indicators
    from target_engineering import create_multiclass_target, print_target_distribution
    from class_balancing import calculate_scale_pos_weight, analyze_class_balance
    from cross_validation import time_series_cv
    from sklearn.model_selection import train_test_split

    df = df.copy()

    # Enhanced indicators
    print(f"\n[FEATURES] Adding enhanced indicators for {symbol}/{mode}...")
    try:
        df = add_indicators(df)
        print(f"[FEATURES] ✅ Added {len(df.columns)} total features")
    except Exception as e:
        print(f"[FEATURES] ⚠️ Enhanced indicators failed: {e}")
        print(f"[FEATURES] Falling back to basic feature engineering")

    # Feature generation
    if mode == "daily":
        df_feat = build_daily_features(df)
    elif mode in ("intraday", "intraday_mr", "intraday_mom"):
        df_feat = build_intraday_features(df)
        if mode in ("intraday_mr", "intraday_mom"):
            df_feat = _filter_intraday_rows_by_mode(df_feat, mode=mode)
        # Drop regime helper columns to avoid leakage
        for c in ["ret_12", "mom_12_abs", "vol_12"]:
            if c in df_feat.columns:
                df_feat.drop(columns=c, inplace=True)
    else:
        raise ValueError("mode must be 'daily' or one of 'intraday', 'intraday_mr', 'intraday_mom'")

    # Clean DataFrame
    print(f"\n[CLEANUP] Cleaning DataFrame columns...")
    cleaned_cols = []
    for col in list(df_feat.columns):
        if isinstance(df_feat[col], pd.DataFrame):
            print(f"[CLEANUP] Converting DataFrame column '{col}' to Series")
            df_feat[col] = df_feat[col].iloc[:, 0]
            cleaned_cols.append(col)
        elif hasattr(df_feat[col], 'values') and df_feat[col].values.ndim > 1:
            print(f"[CLEANUP] Column '{col}' has shape {df_feat[col].values.shape}")
            if df_feat[col].values.shape[1] > 0:
                df_feat[col] = pd.Series(df_feat[col].values[:, 0], index=df_feat.index)
                cleaned_cols.append(col)
            else:
                df_feat.drop(columns=[col], inplace=True)

    if cleaned_cols:
        print(f"[CLEANUP] ✅ Cleaned {len(cleaned_cols)} columns: {cleaned_cols[:5]}{'...' if len(cleaned_cols) > 5 else ''}")
    else:
        print(f"[CLEANUP] ✅ All columns already clean")

    # Target
    if use_multiclass:
        print(f"\n[TARGET] Creating multi-class target (5 classes)...")
        thresholds = (-0.015, -0.003, 0.003, 0.015) if mode == "daily" else (-0.008, -0.002, 0.002, 0.008)
        df_feat = create_multiclass_target(df_feat, forward_periods=1, thresholds=thresholds)
        target_col = "Target_multiclass"
        num_classes = 5
        objective = "multi:softprob"
        eval_metric = "mlogloss"
        print_target_distribution(df_feat, target_col)
    else:
        print(f"\n[TARGET] Creating binary target...")
        df_feat["target"] = (df_feat["Close"].shift(-1) > df_feat["Close"]).astype(int)
        target_col = "target"
        num_classes = 2
        objective = "binary:logistic"
        eval_metric = "logloss"

    df_feat.dropna(inplace=True)

    if df_feat.empty:
        raise ValueError(f"No rows left after feature engineering for {symbol}/{mode}")

    # Class balance check
    class_counts = df_feat[target_col].value_counts()
    unique_classes = sorted(class_counts.index.tolist())

    print(f"\n[CLASS CHECK] Found {len(unique_classes)} unique classes: {unique_classes}")
    for cls in unique_classes:
        count = class_counts[cls]
        print(f"  Class {cls}: {count} samples ({count/len(df_feat)*100:.1f}%)")

    if mode in ("intraday_mr", "intraday_mom"):
        MIN_SAMPLES_PER_CLASS = 3
        MIN_UNIQUE_CLASSES = 2
        print(f"[CLASS CHECK] Using relaxed thresholds for {mode}: min_samples={MIN_SAMPLES_PER_CLASS}")
    else:
        MIN_SAMPLES_PER_CLASS = 5
        MIN_UNIQUE_CLASSES = 2

    valid_classes = [cls for cls in unique_classes if class_counts[cls] >= MIN_SAMPLES_PER_CLASS]

    if len(valid_classes) < MIN_UNIQUE_CLASSES:
        if mode in ("intraday_mr", "intraday_mom"):
            print(f"\n[CLASS CHECK] Trying fallback: min_samples=2")
            valid_classes = [cls for cls in unique_classes if class_counts[cls] >= 2]
            if len(valid_classes) >= MIN_UNIQUE_CLASSES:
                print(f"[CLASS CHECK] ✅ Fallback successful: {len(valid_classes)} valid classes")
            else:
                raise ValueError(
                    f"Insufficient class diversity for {symbol}/{mode} even with min_samples=2: "
                    f"Found {len(valid_classes)} valid classes (need {MIN_UNIQUE_CLASSES}). "
                    f"Class counts: {class_counts.to_dict()}."
                )
        else:
            raise ValueError(
                f"Insufficient class diversity for {symbol}/{mode}: "
                f"Found {len(valid_classes)} valid classes (need {MIN_UNIQUE_CLASSES}). "
                f"Class counts: {class_counts.to_dict()}."
            )

    # Remap classes to consecutive integers if needed
    if use_multiclass and len(valid_classes) < num_classes:
        print(f"\n[CLASS REMAP] Only {len(valid_classes)}/{num_classes} classes have enough samples")
        df_feat = df_feat[df_feat[target_col].isin(valid_classes)]
        class_mapping = {old: new for new, old in enumerate(sorted(valid_classes))}
        df_feat[target_col] = df_feat[target_col].map(class_mapping)
        num_classes = len(valid_classes)
        print(f"[CLASS REMAP] Mapping: {class_mapping}")
        print(f"[CLASS REMAP] ✅ Remapped to {num_classes} consecutive classes")
        print(df_feat[target_col].value_counts().sort_index())

    # Prepare feature matrix
    print(f"\n[FEATURES] Preparing feature matrix...")
    exclude_cols = [target_col, "Close", "Open", "High", "Low", "Volume"]
    feature_cols = [col for col in df_feat.columns if col not in exclude_cols]

    X = df_feat[feature_cols].copy()
    y = df_feat[target_col].copy()

    # Validation
    print(f"[VALIDATION] Validating feature matrix...")

    unnamed_cols = [col for col in X.columns if col == '' or pd.isna(col) or str(col).strip() == '']
    if unnamed_cols:
        print(f"[VALIDATION] Dropping {len(unnamed_cols)} unnamed columns")
        X = X.drop(columns=unnamed_cols, errors='ignore')

    non_numeric = X.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) > 0:
        print(f"[VALIDATION] Dropping {len(non_numeric)} non-numeric columns")
        X = X.drop(columns=non_numeric)

    problematic_cols = []
    for col in X.columns:
        try:
            if isinstance(X[col], pd.DataFrame):
                X[col] = X[col].iloc[:, 0]
            col_values = X[col].values
            if col_values.ndim > 1:
                if col_values.shape[0] == len(X) and col_values.shape[1] == 1:
                    X[col] = col_values.ravel()
                elif col_values.shape[0] == len(X) and col_values.shape[1] > 1:
                    X[col] = col_values[:, 0]
                else:
                    problematic_cols.append(col)
                    continue
            if X[col].values.ndim == 1:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            else:
                problematic_cols.append(col)
        except Exception as e:
            print(f"[VALIDATION] Error processing '{col}': {e}")
            problematic_cols.append(col)

    if problematic_cols:
        print(f"[VALIDATION] Dropping {len(problematic_cols)} problematic columns")
        X = X.drop(columns=problematic_cols, errors='ignore')

    all_nan_cols = X.columns[X.isna().all()].tolist()
    if all_nan_cols:
        print(f"[VALIDATION] Dropping {len(all_nan_cols)} all-NaN columns")
        X = X.drop(columns=all_nan_cols)

    if X.isna().any().any():
        nan_count = X.isna().sum().sum()
        print(f"[VALIDATION] Filling {nan_count} NaN values with 0")
        X = X.fillna(0)

    feature_list = list(X.columns)
    print(f"[FEATURES] Final feature count: {len(feature_list)}")

    if len(feature_list) == 0:
        raise ValueError(f"No valid features remaining for {symbol}/{mode}")

    y = pd.Series(y, dtype=int)

    print(f"\n[SANITY CHECK] Feature matrix shape: {X.shape}")
    print(f"[SANITY CHECK] Target shape: {y.shape}")
    print(f"[SANITY CHECK] All columns are 1D: {all(X[col].values.ndim == 1 for col in X.columns)}")

    for col in X.columns:
        if isinstance(X[col], pd.DataFrame):
            raise ValueError(f"Column '{col}' is still a DataFrame!")
        if X[col].values.ndim != 1:
            raise ValueError(f"Column '{col}' is not 1D! Shape: {X[col].values.shape}")

    # Train/test split
    USE_TIME_SERIES_CV = True
    if USE_TIME_SERIES_CV and len(X) >= 200:
        print(f"\n[CV] Using time-series cross-validation...")
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]
        print(f"[CV] Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    else:
        split_idx = int(len(X) * 0.8)
        if split_idx <= 0 or split_idx >= len(X):
            raise ValueError(f"Not enough data to split train/test for {symbol}/{mode}")
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    train_classes = sorted(y_train.unique())
    test_classes = sorted(y_test.unique())

    print(f"\n[SPLIT CHECK] Train classes: {train_classes} (counts: {y_train.value_counts().to_dict()})")
    print(f"[SPLIT CHECK] Test classes: {test_classes} (counts: {y_test.value_counts().to_dict()})")

    if len(train_classes) < 2:
        raise ValueError(
            f"Train set has only {len(train_classes)} class(es) for {symbol}/{mode}. "
            f"Cannot train a classifier."
        )

    if len(test_classes) < 2:
        print(f"\n[WARN] Test set has only {len(test_classes)} class(es)!")
        print(f"[FIX] Attempting stratified split...")
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42, shuffle=True
            )
            test_classes = sorted(y_test.unique())
            print(f"[FIX] ✅ Stratified split successful")
            print(f"[FIX] Train classes: {sorted(y_train.unique())}")
            print(f"[FIX] Test classes: {test_classes}")
        except Exception as e:
            print(f"[FIX] ⚠️ Stratified split failed: {e}")
            print(f"[FIX] Proceeding with original split")

    # Class balancing
    print(f"\n[CLASS BALANCE] Analyzing class distribution...")
    analyze_class_balance(y_train, name=f"{symbol}/{mode} Train Set")

    scale_pos_weight = None
    if num_classes == 2:
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        if n_pos > 0 and n_neg > 0:
            scale_pos_weight = calculate_scale_pos_weight(y_train)
            if scale_pos_weight > 1.2 or scale_pos_weight < 0.83:
                print(f"[CLASS WEIGHT] Applying scale_pos_weight={scale_pos_weight:.3f}")
            else:
                print(f"[CLASS WEIGHT] Classes balanced, no weighting needed")
                scale_pos_weight = None
        else:
            print(f"[CLASS WEIGHT] ⚠️ Insufficient class samples")
    else:
        print(f"[CLASS WEIGHT] Multi-class mode - using default class weights")

    # XGB params
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

    # Fit
    print(f"\n[TRAINING] Fitting {symbol}/{mode} model...")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    print(f"[TRAINING] ✅ Model trained with {model.n_estimators} trees")

    base_model = model

    # Calibration
    final_model = model
    calibrated = False

    if num_classes == 2:
        print(f"\n[CALIBRATION] Applying Platt scaling (sigmoid) for {symbol}/{mode}")
        calibration_method = 'sigmoid'
    else:
        print(f"\n[CALIBRATION] Applying isotonic calibration for {symbol}/{mode}")
        calibration_method = 'isotonic'

    try:
        calibrated_model = CalibratedClassifierCV(
            model, method=calibration_method, cv='prefit', n_jobs=-1
        )
        calibrated_model.fit(X_test, y_test)
        final_model = calibrated_model
        calibrated = True
        print(f"[CALIBRATION] ✅ Successfully calibrated using {calibration_method}")
    except Exception as e:
        print(f"[CALIBRATION] ⚠️ Failed to calibrate: {e}")
        print(f"[CALIBRATION] Using uncalibrated model as fallback")
        final_model = model

    # Metrics
    print(f"\n[EVALUATION] Computing metrics on test set...")
    metrics = {}

    try:
        if num_classes == 2:
            y_pred_proba = final_model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)

            try:
                metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
            except Exception as e:
                print(f"[METRICS] Accuracy failed: {e}")
                metrics["accuracy"] = 0.0

            try:
                if len(np.unique(y_test)) >= 2:
                    metrics["logloss"] = float(log_loss(y_test, y_pred_proba, labels=[0, 1]))
                else:
                    metrics["logloss"] = None
            except Exception as e:
                print(f"[METRICS] LogLoss failed: {e}")
                metrics["logloss"] = None

            try:
                if len(np.unique(y_test)) >= 2:
                    metrics["roc_auc"] = float(roc_auc_score(y_test, y_pred_proba))
                else:
                    metrics["roc_auc"] = None
            except Exception as e:
                print(f"[METRICS] ROC AUC failed: {e}")
                metrics["roc_auc"] = None

            try:
                metrics["precision"] = float(precision_score(y_test, y_pred, zero_division=0))
                metrics["recall"] = float(recall_score(y_test, y_pred, zero_division=0))
                metrics["f1"] = float(f1_score(y_test, y_pred, zero_division=0))
            except Exception as e:
                print(f"[METRICS] Precision/Recall/F1 failed: {e}")
                metrics["precision"] = 0.0
                metrics["recall"] = 0.0
                metrics["f1"] = 0.0

            try:
                metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()
            except Exception as e:
                print(f"[METRICS] Confusion matrix failed: {e}")
                metrics["confusion_matrix"] = [[0, 0], [0, 0]]

            try:
                if len(np.unique(y_test)) >= 2:
                    metrics["brier_score"] = float(brier_score_loss(y_test, y_pred_proba))
                    avg_pred = float(np.mean(y_pred_proba))
                    actual_rate = float(np.mean(y_test))
                    metrics["calibration_error"] = abs(avg_pred - actual_rate)
                else:
                    metrics["brier_score"] = None
                    metrics["calibration_error"] = None
            except Exception as e:
                print(f"[METRICS] Calibration metrics failed: {e}")
                metrics["brier_score"] = None
                metrics["calibration_error"] = None

        else:
            y_pred_proba = final_model.predict_proba(X_test)
            y_pred = np.argmax(y_pred_proba, axis=1)

            try:
                metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
            except Exception as e:
                print(f"[METRICS] Accuracy failed: {e}")
                metrics["accuracy"] = 0.0

            try:
                all_classes = list(range(num_classes))
                metrics["logloss"] = float(log_loss(y_test, y_pred_proba, labels=all_classes))
            except Exception as e:
                print(f"[METRICS] LogLoss failed: {e}")
                metrics["logloss"] = None

            try:
                metrics["precision_weighted"] = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
                metrics["recall_weighted"] = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
                metrics["f1_weighted"] = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
            except Exception as e:
                print(f"[METRICS] Weighted metrics failed: {e}")
                metrics["precision_weighted"] = 0.0
                metrics["recall_weighted"] = 0.0
                metrics["f1_weighted"] = 0.0

            try:
                metrics["precision_per_class"] = precision_score(y_test, y_pred, average=None, zero_division=0, labels=list(range(num_classes))).tolist()
                metrics["recall_per_class"] = recall_score(y_test, y_pred, average=None, zero_division=0, labels=list(range(num_classes))).tolist()
            except Exception as e:
                print(f"[METRICS] Per-class metrics failed: {e}")
                metrics["precision_per_class"] = [0.0] * num_classes
                metrics["recall_per_class"] = [0.0] * num_classes

            try:
                metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred, labels=list(range(num_classes))).tolist()
            except Exception as e:
                print(f"[METRICS] Confusion matrix failed: {e}")
                metrics["confusion_matrix"] = [[0] * num_classes for _ in range(num_classes)]

            try:
                from sklearn.preprocessing import label_binarize
                y_test_bin = label_binarize(y_test, classes=range(num_classes))
                metrics["brier_score"] = float(np.mean(np.sum((y_test_bin - y_pred_proba) ** 2, axis=1)))
            except Exception as e:
                print(f"[METRICS] Brier score failed: {e}")
                metrics["brier_score"] = None

        metrics["train_samples"] = len(X_train)
        metrics["test_samples"] = len(X_test)
        metrics["num_features"] = len(feature_list)
        metrics["num_classes"] = num_classes
        metrics["train_class_counts"] = y_train.value_counts().to_dict()
        metrics["test_class_counts"] = y_test.value_counts().to_dict()

    except Exception as e:
        print(f"[EVALUATION] ⚠️ Critical metrics error: {e}")
        traceback.print_exc()
        metrics = {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "num_features": len(feature_list),
            "num_classes": num_classes,
            "error": str(e)
        }

    # Print summary
    print(f"\n{'='*60}")
    print(f"MODEL VALIDATION: {symbol}/{mode.upper()}")
    print(f"{'='*60}")
    print(f"Training samples:     {metrics.get('train_samples', 'N/A')}")
    print(f"Test samples:         {metrics.get('test_samples', 'N/A')}")
    print(f"Features:             {metrics.get('num_features', 'N/A')}")
    print(f"Target classes:       {metrics.get('num_classes', 'N/A')}")
    print(f"Calibrated:           {'Yes' if calibrated else 'No'}")
    print("-" * 60)

    key_metrics = ["accuracy", "logloss", "roc_auc", "precision", "recall", "f1",
                   "brier_score", "calibration_error",
                   "precision_weighted", "recall_weighted", "f1_weighted"]

    for k in key_metrics:
        if k in metrics and metrics[k] is not None:
            v = metrics[k]
            try:
                print(f"{k:20} {v:.4f}")
            except:
                print(f"{k:20} {v}")

    if "confusion_matrix" in metrics:
        print("-" * 60)
        print("Confusion Matrix:")
        cm = np.array(metrics["confusion_matrix"])
        for row in cm:
            print("  ", row)

    print("=" * 60)

    return {
        "model": final_model,
        "base_model": base_model,
        "features": feature_list,
        "metrics": metrics,
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
    Handles both binary (returns float) and multi-class (returns dict).
    """
    if model_dict is None:
        return None

    if df_features is None or df_features.empty:
        print("[ERROR] predict_from_model: df_features empty.")
        return None

    try:
        model = model_dict["model"]
        feat_cols = model_dict["features"]
        num_classes = model_dict.get("num_classes", 2)
        target_type = model_dict.get("target_type", "binary")

        # Case-insensitive feature matching
        feature_mapping = {}
        missing_features = []
        match_stats = {'exact': 0, 'case': 0, 'normalized': 0}

        for model_feat in feat_cols:
            matched_feat = find_matching_feature(model_feat, df_features.columns.tolist())
            if matched_feat:
                feature_mapping[model_feat] = matched_feat
                if matched_feat == model_feat:
                    match_stats['exact'] += 1
                elif matched_feat.lower() == model_feat.lower():
                    match_stats['case'] += 1
                else:
                    match_stats['normalized'] += 1
            else:
                feature_mapping[model_feat] = None
                missing_features.append(model_feat)

        # Build feature vector
        X_data = {}
        for model_feat, data_feat in feature_mapping.items():
            if data_feat is not None:
                try:
                    val = df_features[data_feat].iloc[-1]
                    if not isinstance(val, (int, float, np.number)):
                        val = float(val) if pd.notna(val) else 0.0
                    if not np.isfinite(val):
                        val = 0.0
                    X_data[model_feat] = val
                except Exception:
                    X_data[model_feat] = 0.0
            else:
                X_data[model_feat] = 0.0

        X = pd.DataFrame([X_data], columns=feat_cols)

        # Report missing features
        total_features = len(feat_cols)
        if missing_features:
            missing_pct = len(missing_features) / total_features * 100
            if missing_pct > 20:
                print(f"[ERROR] {len(missing_features)} features missing ({missing_pct:.1f}%)")
                print(f"[ERROR] Missing: {missing_features[:10]}")
            elif missing_pct > 5:
                print(f"[WARN] {len(missing_features)} features missing ({missing_pct:.1f}%)")
                print(f"[WARN] Top missing: {missing_features[:5]}")
        else:
            if match_stats['case'] + match_stats['normalized'] > 0:
                print(f"[INFO] Matched {total_features} features: "
                      f"{match_stats['exact']} exact, "
                      f"{match_stats['case']} case-insensitive, "
                      f"{match_stats['normalized']} normalized")

        # Final cleanup
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                X[col] = pd.to_numeric(X[col], errors='coerce')
        X = X.fillna(0)
        for col in X.columns:
            if X[col].values.ndim != 1:
                X[col] = X[col].values.ravel()

        proba = model.predict_proba(X)[0]

        # Binary
        if num_classes == 2 or target_type == "binary":
            return float(proba[1])

        # Multi-class
        actual_num_classes = len(proba)

        if actual_num_classes == 5:
            class_names = ['Strong Down', 'Weak Down', 'Flat', 'Weak Up', 'Strong Up']
            predicted_class = int(np.argmax(proba))
            confidence = float(np.max(proba))

            bullish_simple = float(proba[3] + proba[4])
            bearish_simple = float(proba[0] + proba[1])
            flat_prob = float(proba[2])

            bullish_weighted = float(proba[3] * 1.0 + proba[4] * 2.0)
            bearish_weighted = float(proba[0] * 2.0 + proba[1] * 1.0)
            flat_weighted = float(proba[2] * 0.5)
            total_weighted = bullish_weighted + bearish_weighted + flat_weighted

            if total_weighted > 0:
                final_prob = bullish_weighted / (bullish_weighted + bearish_weighted)
            else:
                final_prob = 0.5

            return {
                'final_prob': float(final_prob),
                'probabilities': proba.tolist(),
                'predicted_class': predicted_class,
                'predicted_class_name': class_names[predicted_class],
                'confidence': confidence,
                'bullish_prob': bullish_simple,
                'bearish_prob': bearish_simple,
                'flat_prob': flat_prob,
                'class_breakdown': {
                    'strong_down': float(proba[0]),
                    'weak_down': float(proba[1]),
                    'flat': float(proba[2]),
                    'weak_up': float(proba[3]),
                    'strong_up': float(proba[4]),
                },
                'num_classes': num_classes,
                'model_type': target_type,
            }

        else:
            # Reduced class model (3 or 4 classes)
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
                'final_prob': float(final_prob),
                'probabilities': proba.tolist(),
                'predicted_class': predicted_class,
                'predicted_class_name': f'Class_{predicted_class}',
                'confidence': confidence,
                'bullish_prob': bullish_prob,
                'bearish_prob': bearish_prob,
                'flat_prob': 0.0,
                'class_breakdown': {f'class_{i}': float(proba[i]) for i in range(actual_num_classes)},
                'num_classes': actual_num_classes,
                'model_type': target_type,
            }

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
        return prediction.get('final_prob')
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
        final_prob = prediction.get('final_prob', 0.5)
        pred_class = prediction.get('predicted_class_name', 'Unknown')
        confidence = prediction.get('confidence', 0.0)
        breakdown = prediction.get('class_breakdown', {})

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
        print(f"[VIX FILTER] {symbol} VIX={vix_level:.1f} → scale={scale:.2f} | prob {prob:.3f} → {adjusted:.3f}")
        return float(adjusted)

    return float(prob)


# ---------------------------------------------------------
# CONTRADICTION DETECTION
# ---------------------------------------------------------
def detect_contradiction(daily_prob, intraday_prob, momentum=None, symbol="", verbose=True):
    """Detect contradictions between daily and intraday signals."""
    if daily_prob is None or intraday_prob is None:
        return {'contradiction': False, 'reason': 'Missing data', 'severity': 'low'}

    BULLISH_THRESHOLD = 0.55
    BEARISH_THRESHOLD = 0.45

    daily_bullish = daily_prob > BULLISH_THRESHOLD
    daily_bearish = daily_prob < BEARISH_THRESHOLD
    intraday_bullish = intraday_prob > BULLISH_THRESHOLD
    intraday_bearish = intraday_prob < BEARISH_THRESHOLD

    models_disagree = (
        (daily_bullish and intraday_bearish) or
        (daily_bearish and intraday_bullish)
    )

    if not models_disagree:
        return {'contradiction': False, 'reason': 'Models agree', 'severity': 'low'}

    if momentum is not None and abs(momentum) > 0.01:
        reason = (
            f"{symbol}: Strong momentum ({momentum:+.2%}) but models disagree - "
            f"daily={daily_prob:.2f}, intraday={intraday_prob:.2f}"
        )
        if verbose:
            print(f"[CONTRADICTION] {reason}")
        return {'contradiction': True, 'reason': reason, 'severity': 'high'}

    extreme_disagree = (
        (daily_prob > 0.65 and intraday_prob < 0.35) or
        (daily_prob < 0.35 and intraday_prob > 0.65)
    )

    if extreme_disagree:
        reason = f"{symbol}: Extreme disagreement - daily={daily_prob:.2f}, intraday={intraday_prob:.2f}"
        if verbose:
            print(f"[CONTRADICTION] {reason}")
        return {'contradiction': True, 'reason': reason, 'severity': 'medium'}

    return {'contradiction': False, 'reason': 'Mild disagreement', 'severity': 'low'}


# ---------------------------------------------------------
# COMPUTE SIGNALS (MAIN ENTRY POINT)
# ---------------------------------------------------------
def compute_signals(
    symbol,
    lookback_minutes=60,
    intraday_weight=INTRADAY_WEIGHT,
    resample_to="15min",
):
    """
    Compute trading signals combining daily and intraday models.
    Returns both simple probabilities (backward compatible) and detailed
    multi-class predictions for advanced analysis.
    """
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

    # Load models
    model_daily = load_model(symU, mode="daily")
    model_intra_mr = load_model(symU, mode="intraday_mr")
    model_intra_mom = load_model(symU, mode="intraday_mom")
    model_intraday_legacy = load_model(symU, mode="intraday")

    # Fetch VIX
    vix_level = fetch_vix_level()
    results["vix_level"] = vix_level
    print(f"[VIX] {symU} Current VIX: {vix_level:.1f}")

    # Daily prediction
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
                        try:
                            dc = df_daily["Close"]
                            dc = dc.iloc[:, 0] if isinstance(dc, pd.DataFrame) else dc
                            dc = dc.dropna()
                            if len(dc) > 0:
                                current_price = float(dc.iloc[-1])
                        except:
                            pass

                    if current_price and float(current_price) > 0:
                        log_prediction(
                            symbol=symU,
                            mode="daily",
                            predicted_prob=float(dp),
                            price=float(current_price),
                            prediction_details=daily_prediction if isinstance(daily_prediction, dict) else None
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

    # Intraday prediction
    df_intra_resampled = None
    df_feat_intra = None
    vol = None
    mom1h = None
    close = pd.Series(dtype=float)  # initialise so vol_regime block is safe

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

            try:
                if len(close) > 0:
                    results["price"] = float(close.iloc[-1])
            except Exception:
                pass

            if len(close) >= 12:
                vol = close.pct_change().rolling(12).std().iloc[-1]
                results["intraday_vol"] = float(vol) if not pd.isna(vol) else None

            if len(close) >= 4:
                mom1h = close.pct_change(4).iloc[-1]
                results["intraday_mom"] = float(mom1h) if not pd.isna(mom1h) else None

            # Volume ratio
            try:
                vol_series = df_intra_resampled["Volume"]
                if isinstance(vol_series, pd.DataFrame):
                    vol_series = vol_series.iloc[:, 0]
                vol_series = pd.to_numeric(vol_series, errors="coerce").dropna()

                if len(vol_series) >= 10:
                    current_vol = float(vol_series.iloc[-1])
                    baseline_vol = float(vol_series.iloc[:-1].tail(20).median())
                    results["intraday_volume_ratio"] = (
                        current_vol / baseline_vol if baseline_vol > 0 else None
                    )
                    print(f"[VOL_RATIO] {symU} current={current_vol:.0f}  baseline(median)={baseline_vol:.0f}  ratio={results['intraday_volume_ratio']:.2f}")
            except Exception as _e:
                print(f"[VOL_RATIO] {symU} failed: {_e}")
                results["intraday_volume_ratio"] = None

            # Adaptive thresholds
            try:
                adaptive = get_adaptive_regime_thresholds(symU, lookback_days=30, percentile=0.70)
                MOMTRIG = float(adaptive['mom_trig'])
                VOLTRIG = float(adaptive['vol_trig'])
                print(f"[ADAPTIVE] {symU} using adaptive: mom={MOMTRIG:.4f} vol={VOLTRIG:.5f}")
            except Exception:
                MOMTRIG = float(INTRADAY_MOM_TRIG)
                VOLTRIG = float(INTRADAY_VOL_TRIG)
                ovr = (INTRADAY_REGIME_OVERRIDES or {}).get(symU)
                if ovr:
                    MOMTRIG = float(ovr.get("mom_trig", MOMTRIG))
                    VOLTRIG = float(ovr.get("vol_trig", VOLTRIG))
                print(f"[CONFIG] {symU} using config: mom={MOMTRIG:.4f} vol={VOLTRIG:.5f}")

            # Select model based on regime
            active_model = None
            model_used = None
            if vol is not None and mom1h is not None:
                if abs(mom1h) >= MOMTRIG or vol >= VOLTRIG:
                    active_model = model_intra_mom
                    results["intraday_regime"] = "mom"
                    model_used = "intraday_mom"
                    print(f"[REGIME] {symU}: MOMENTUM (mom={mom1h:.4f}, vol={vol:.5f})")
                else:
                    active_model = model_intra_mr
                    results["intraday_regime"] = "mr"
                    model_used = "intraday_mr"
                    print(f"[REGIME] {symU}: MEAN-REVERSION (mom={mom1h:.4f}, vol={vol:.5f})")

            if active_model is None:
                if model_intra_mom is not None:
                    active_model = model_intra_mom
                    model_used = "intraday_mom"
                elif model_intra_mr is not None:
                    active_model = model_intra_mr
                    model_used = "intraday_mr"
                elif model_intraday_legacy is not None:
                    active_model = model_intraday_legacy
                    model_used = "intraday"
                print(f"[FALLBACK] {symU}: Using {model_used}")

            if active_model is None:
                print(f"[FAIL] {symU}: NO MODELS LOADED AT ALL")
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
                    print(f"[TRY] {symU} using {model_used}")
                    intraday_prediction = predict_from_model(active_model, df_feat_intra)

                    results["intraday_prediction"] = intraday_prediction
                    results["intraday_model_used"] = model_used
                    results["intraday_signal"] = get_signal_details(intraday_prediction)

                    ip = get_final_prob(intraday_prediction)

                    if ip is not None:
                        ip_original = float(ip)
                        momentum_boost_applied = False

                        # Momentum probability boost
                        if results.get("intraday_regime") == "mom" and mom1h is not None:
                            mom1h_val = float(mom1h)
                            if abs(mom1h_val) > 0.005 and 0.30 < ip_original < 0.70:
                                if abs(mom1h_val) > 0.01:
                                    boost_magnitude = min(0.20, abs(mom1h_val) * 7.0)
                                else:
                                    boost_magnitude = min(0.15, abs(mom1h_val) * 5.0)

                                if mom1h_val > 0:
                                    ip = min(0.85, ip_original + boost_magnitude)
                                    print(f"[MOMENTUM BOOST] {symU} mom={mom1h_val:.2%} (UP) -> prob {ip_original:.3f} → {ip:.3f}")
                                else:
                                    ip = max(0.15, ip_original - boost_magnitude)
                                    print(f"[MOMENTUM BOOST] {symU} mom={mom1h_val:.2%} (DOWN) -> prob {ip_original:.3f} → {ip:.3f}")
                                momentum_boost_applied = True

                        # SPY-specific cap
                        if symU == "SPY" and model_used == "intraday_mom":
                            if ip > 0.70:
                                print(f"[SPY CAP] Capping {ip:.3f} → 0.70")
                                ip = 0.70
                            elif ip < 0.30:
                                print(f"[SPY CAP] Flooring {ip:.3f} → 0.30")
                                ip = 0.30

                        results["intraday_prob"] = float(ip)
                        results["intraday_prob_original"] = float(ip_original)
                        results["intraday_quality_score"] = min(1.0, len(df_intra_resampled) / 120.0)
                        boost_str = " (BOOSTED)" if momentum_boost_applied else ""
                        print(f"[SUCCESS] {symU} {model_used} ip={ip:.3f}{boost_str}")

                        try:
                            current_price = results.get("price") or fetch_latest_price(symU)
                            if current_price and float(current_price) > 0:
                                log_prediction(
                                    symbol=symU,
                                    mode=model_used,
                                    predicted_prob=float(ip),
                                    price=float(current_price),
                                    prediction_details=intraday_prediction if isinstance(intraday_prediction, dict) else None
                                )
                        except Exception as e:
                            print(f"[WARN] Could not log prediction: {e}")
                    else:
                        print(f"[PREDICT_FAIL] {symU} {model_used} returned None")

    except Exception as e:
        print(f"[ERROR] Intraday prediction error: {e}")
        traceback.print_exc()
        results["allow_intraday"] = False
        results["intraday_prob"] = None
        results["intraday_quality_score"] = 0.0

    # Price fallback
    if results.get("price") is None:
        try:
            p = fetch_latest_price(symU)
            if p is not None and float(p) > 0:
                results["price"] = float(p)
        except Exception:
            pass

    # Combine daily + intraday
    dp = results["daily_prob"]
    ip = results["intraday_prob"]

    if vol is None:
        vol = float(results.get("intraday_vol") or 0.0)
    if mom1h is None:
        mom1h = float(results.get("intraday_mom") or 0.0)

    weight = float(intraday_weight)
    if not results["allow_intraday"] or ip is None:
        weight = 0.0
    else:
        q = float(results.get("intraday_quality_score", 1.0) or 0.0)
        weight = float(intraday_weight) * q

    original_weight = weight

    # Cap weight in first 30 min after market open
    import pytz
    from datetime import datetime as _dt
    _ny = pytz.timezone("America/New_York")
    _now_ny = _dt.now(_ny)
    _market_open_min = (_now_ny.hour * 60 + _now_ny.minute) - (9 * 60 + 30)
    if 0 <= _market_open_min <= 30:
        weight = min(weight, 0.30)
        print(f"[WEIGHT] {symU}: first 30min cap → intraday_weight={weight:.2f} (was {original_weight:.2f})")

    # Contradiction check before blending
    if dp is not None and ip is not None:
        contradiction = detect_contradiction(
            daily_prob=dp,
            intraday_prob=ip,
            momentum=mom1h,
            symbol=symU,
            verbose=True
        )

        if contradiction['contradiction']:
            print(f"[WARN] {symU}: Contradiction detected - zeroing intraday weight (was {weight:.2f})")
            print(f"       Reason: {contradiction['reason']}")
            weight = 0.0
            results["contradiction_detected"] = True
            results["contradiction_reason"] = contradiction['reason']
            results["contradiction_severity"] = contradiction['severity']
        else:
            results["contradiction_detected"] = False
    else:
        results["contradiction_detected"] = False

    results["intraday_weight_original"] = original_weight
    results["intraday_weight_used"] = weight

    # Blend
    if dp is None and ip is None:
        results["final_prob"] = None
    elif dp is None:
        results["final_prob"] = float(ip)
    elif ip is None or weight <= 0.0:
        results["final_prob"] = float(dp)
    else:
        raw_final = float(weight * ip + (1 - weight) * dp)
        final_prob_vix = vix_market_regime_filter(raw_final, vix_level=vix_level, symbol=symU)
        results["final_prob"] = final_prob_vix
        results["final_prob_raw"] = raw_final

    # Additional weight adjustments (re-blend if weight changes)
    if mom1h is not None and mom1h >= 0.005:
        weight = max(weight, 0.75)
        print(f"[MOMENTUM BOOST] {symU}: mom={mom1h:.2%} -> weight={weight:.2f}")

    if mom1h is not None and mom1h >= 0.010:
        weight = min(0.90, weight + 0.15)
        print(f"[MOMENTUM SURGE] {symU}: mom={mom1h:.2%} -> weight={weight:.2f}")

    vol_val = results.get("intraday_vol")
    if vol_val is not None:
        if vol_val >= 0.025:
            weight = min(0.90, weight + 0.20)
        elif vol_val <= 0.008 and symU == "SPY":
            weight = max(0.30, weight - 0.20)

    try:
        if dp is not None and ip is not None and dp > 0.78 and ip < 0.25 and mom1h and mom1h < -0.005:
            weight = max(weight, 0.70)
    except:
        pass

    # Volatility regime scaling
    vol_regime = 1.0
    if len(close) >= 20:
        vol20_pct = close.pct_change().tail(20).std()
        vol_regime = vol20_pct / 0.02
        if vol_regime > 1.5:
            weight = min(0.90, weight + 0.15)
        elif vol_regime < 0.7:
            weight = max(0.25, weight - 0.15)

    results["vol_regime"] = float(vol_regime)

    weight = float(min(max(weight, 0.0), 0.90))

    if weight != results["intraday_weight_used"]:
        print(f"[WEIGHT ADJUSTED] {symU}: {results['intraday_weight_used']:.2f} -> {weight:.2f}")
        results["intraday_weight_used"] = weight
        if dp is not None and ip is not None and weight > 0.0:
            results["final_prob"] = float(weight * ip + (1 - weight) * dp)

    results["intraday_weight"] = weight

    # Debug prints
    def fmt(x, n=3):
        return "NA" if x is None else f"{float(x):.{n}f}"

    def fmtpct(x):
        return "NA" if x is None else f"{float(x)*100:.2f}%"

    print(f"[DEBUG] {symU} dp={fmt(dp)} ip={fmt(ip)} q={results.get('intraday_quality_score',0):.2f} weight={weight:.2f} model={results.get('intraday_model_used')} price={fmt(results.get('price'), 2)}")
    print(f"[DEBUG] {symU} vol={fmt(results.get('intraday_vol'), 5)} mom={fmtpct(results.get('intraday_mom'))} regime={results.get('intraday_regime')}")

    daily_pred = results.get("daily_prediction")
    intraday_pred = results.get("intraday_prediction")

    if isinstance(daily_pred, dict):
        bullish = daily_pred.get('bullish_prob', 0)
        bearish = daily_pred.get('bearish_prob', 0)
        pred_class = daily_pred.get('predicted_class_name', 'N/A')
        print(f"[MULTI-CLASS] {symU} DAILY: {pred_class} | Bullish={bullish:.2%} Bearish={bearish:.2%}")

    if isinstance(intraday_pred, dict):
        bullish = intraday_pred.get('bullish_prob', 0)
        bearish = intraday_pred.get('bearish_prob', 0)
        pred_class = intraday_pred.get('predicted_class_name', 'N/A')
        print(f"[MULTI-CLASS] {symU} INTRADAY: {pred_class} | Bullish={bullish:.2%} Bearish={bearish:.2%}")

    results["final_signal"] = get_signal_details(results["final_prob"])

    print(f"[FINAL] {symU} final_prob={fmt(results['final_prob'])} | {results.get('final_signal', '')}")

    if results.get("contradiction_detected"):
        print(f"\n⚠️ [{symU}] CONTRADICTION SUMMARY")
        print(f"   Daily prob: {dp:.2f}")
        print(f"   Intraday prob: {ip:.2f}")
        print(f"   Momentum: {results.get('intraday_mom', 'N/A')}")
        print(f"   Weight: {original_weight:.2f} -> {weight:.2f} (zeroed)")
        print(f"   Final prob: {results['final_prob']:.2f} (using daily only)\n")

    return results


# ---------------------------------------------------------
# COMPATIBILITY WRAPPER
# ---------------------------------------------------------
def predict_next(symbol, lookback_minutes=60, intraday_weight=INTRADAY_WEIGHT, resample_to="15min"):
    """Compatibility wrapper: returns only the final probability."""
    sig = compute_signals(symbol, lookback_minutes, intraday_weight, resample_to)
    return sig.get("final_prob")
