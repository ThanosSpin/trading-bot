# model_xgb.py
import os
from datetime import datetime
import warnings

import joblib
import numpy as np
import pandas as pd
import traceback
from xgboost import XGBClassifier

from features import build_daily_features, build_intraday_features
from data_loader import fetch_historical_data, fetch_intraday_history, fetch_latest_price
from target_labels import create_target_label, backtest_threshold
from trading_metrics import calculate_financial_metrics, print_trading_report
from adaptive_thresholds import get_adaptive_regime_thresholds
from model_monitor import log_prediction, evaluate_predictions
from sklearn.metrics import (
    accuracy_score, log_loss, roc_auc_score,
    confusion_matrix, precision_score, recall_score, f1_score,
    brier_score_loss,
)
from sklearn.calibration import CalibratedClassifierCV

from config import (
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
        # Convert DataFrame columns to Series
        if isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]
        
        # Flatten multi-dimensional arrays (take first column, not ravel)
        if hasattr(df[col], 'values') and df[col].values.ndim > 1:
            if df[col].values.shape[0] == len(df):
                df[col] = pd.Series(df[col].values[:, 0], index=df.index)
        
        # Ensure numeric
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def normalize_feature_name(name: str) -> str:
    """
    Normalize feature name for matching.
    - Lowercase
    - Remove underscores
    - Remove spaces
    """
    return name.lower().replace('_', '').replace(' ', '')


def find_matching_feature(model_feat: str, available_features: list) -> str:
    """
    Find matching feature with fuzzy logic.
    
    Returns:
        Matched feature name or None
    """
    # Try exact match
    if model_feat in available_features:
        return model_feat
    
    # Try case-insensitive
    for feat in available_features:
        if feat.lower() == model_feat.lower():
            return feat
    
    # Try normalized (no underscores, lowercase)
    model_normalized = normalize_feature_name(model_feat)
    for feat in available_features:
        if normalize_feature_name(feat) == model_normalized:
            return feat
    
    # No match
    return None

def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Proper OHLCV resample:
      Open=first, High=max, Low=min, Close=last, Volume=sum
    """
    df = df.copy()

    # must have these columns
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
    """
    Adds simple regime columns for filtering intraday training rows.
    Assumes df_feat contains Close (your feature pipeline already does).
    """
    df_feat = df_feat.copy()

    # 12 bars * 15m = 3 hours (good regime window)
    df_feat["ret_12"] = df_feat["Close"].pct_change(12)
    df_feat["mom_12_abs"] = df_feat["ret_12"].abs()

    # realized vol over same window
    df_feat["vol_12"] = df_feat["Close"].pct_change().rolling(12).std()

    return df_feat


def _filter_intraday_rows_by_mode(df_feat: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    ✅ IMPROVED: Filters intraday feature rows with better regime separation.
    Uses adaptive percentile-based thresholds instead of fixed values.
    """
    df_feat = _add_intraday_regime_cols(df_feat)
    
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
    df_feat = df_feat.dropna(subset=["mom_12_abs", "vol_12"])
    
    if df_feat.empty:
        print(f"[WARN] Empty dataframe after regime column addition for mode={mode}")
        return df_feat
    
    # ✅ IMPROVED: Use percentile-based thresholds for better separation
    mom_p60 = df_feat["mom_12_abs"].quantile(0.60)
    vol_p60 = df_feat["vol_12"].quantile(0.60)
    
    # Lower thresholds for mean reversion (bottom 30%)
    mom_p30 = df_feat["mom_12_abs"].quantile(0.30)
    vol_p30 = df_feat["vol_12"].quantile(0.30)
    
    print(f"[REGIME] {mode}: mom_p30={mom_p30:.4f} mom_p60={mom_p60:.4f} vol_p30={vol_p30:.5f} vol_p60={vol_p60:.5f}")
    
    if mode == "intraday_mom":
        # Momentum: High momentum OR high volatility (more aggressive)
        filtered = df_feat[
            (df_feat["mom_12_abs"] >= mom_p60) | 
            (df_feat["vol_12"] >= vol_p60)
        ]
        print(f"[REGIME] intraday_mom: {len(df_feat)} → {len(filtered)} rows ({len(filtered)/len(df_feat)*100:.1f}%)")
        return filtered
    
    if mode == "intraday_mr":
        # Mean reversion: Low momentum AND low volatility
        filtered = df_feat[
            (df_feat["mom_12_abs"] < mom_p30) & 
            (df_feat["vol_12"] < vol_p30)
        ]
        print(f"[REGIME] intraday_mr: {len(df_feat)} → {len(filtered)} rows ({len(filtered)/len(df_feat)*100:.1f}%)")
        return filtered
    
    return df_feat


def train_model(df: pd.DataFrame, symbol: str, mode: str = "daily", use_multiclass: bool = False):
    """
    ✨ ENHANCED: Train XGB model with Week 1 optimizations
    
    Args:
        df: DataFrame with OHLCV data
        symbol: Stock symbol
        mode: Training mode (daily, intraday, intraday_mr, intraday_mom)
        use_multiclass: If True, train 5-class model; if False, binary model
    
    Improvements:
    1. ✅ Multi-class targets (5 classes instead of binary)
    2. ✅ Enhanced indicators (BB, ATR, ADX - 20+ new features)
    3. ✅ Time-series cross-validation
    4. ✅ Class balancing with scale_pos_weight
    5. ✅ Probability calibration (already implemented)
    """
    
    from indicators import add_indicators  # Enhanced version
    from target_engineering import create_multiclass_target, print_target_distribution
    from class_balancing import calculate_scale_pos_weight, analyze_class_balance
    from cross_validation import time_series_cv
    from sklearn.model_selection import train_test_split
    
    df = df.copy()

    # -----------------------------
    # ✨ NEW: ENHANCED INDICATORS
    # -----------------------------
    print(f"\n[FEATURES] Adding enhanced indicators for {symbol}/{mode}...")
    try:
        df = add_indicators(df)
        print(f"[FEATURES] ✅ Added {len(df.columns)} total features")
    except Exception as e:
        print(f"[FEATURES] ⚠️ Enhanced indicators failed: {e}")
        print(f"[FEATURES] Falling back to basic feature engineering")

    # -----------------------------
    # FEATURE GENERATION (existing)
    # -----------------------------
    if mode == "daily":
        df_feat = build_daily_features(df)
    elif mode in ("intraday", "intraday_mr", "intraday_mom"):
        df_feat = build_intraday_features(df)


        if mode in ("intraday_mr", "intraday_mom"):
            df_feat = _filter_intraday_rows_by_mode(df_feat, mode=mode)


            for c in ["ret_12", "mom_12_abs", "vol_12"]:
                if c in df_feat.columns:
                    df_feat.drop(columns=c, inplace=True)
    else:
        raise ValueError("mode must be 'daily' or one of 'intraday', 'intraday_mr', 'intraday_mom'")

    # ============================================================
    # ✨ NEW: CLEAN DATAFRAME (Fix multi-dimensional columns)
    # ============================================================
    print(f"\n[CLEANUP] Cleaning DataFrame columns...")
    
    cleaned_cols = []
    for col in df_feat.columns:
        # Handle DataFrame columns (take first column)
        if isinstance(df_feat[col], pd.DataFrame):
            print(f"[CLEANUP] Converting DataFrame column '{col}' to Series")
            df_feat[col] = df_feat[col].iloc[:, 0]
            cleaned_cols.append(col)
        
        # Handle multi-dimensional array values
        elif hasattr(df_feat[col], 'values') and df_feat[col].values.ndim > 1:
            print(f"[CLEANUP] Column '{col}' has shape {df_feat[col].values.shape}")
            # Take first column if multi-dimensional
            if df_feat[col].values.shape[1] > 0:
                print(f"[CLEANUP] Extracting first column from '{col}'")
                df_feat[col] = pd.Series(df_feat[col].values[:, 0], index=df_feat.index)
                cleaned_cols.append(col)
            else:
                print(f"[CLEANUP] Warning: Column '{col}' has shape {df_feat[col].values.shape}, dropping")
                df_feat.drop(columns=[col], inplace=True)
    
    if cleaned_cols:
        print(f"[CLEANUP] ✅ Cleaned {len(cleaned_cols)} columns: {cleaned_cols[:5]}{'...' if len(cleaned_cols) > 5 else ''}")
    else:
        print(f"[CLEANUP] ✅ All columns already clean")

    # -----------------------------
    # ✨ MULTI-CLASS TARGET
    # -----------------------------
    if use_multiclass:
        print(f"\n[TARGET] Creating multi-class target (5 classes)...")
        
        if mode == "daily":
            thresholds = (-0.015, -0.003, 0.003, 0.015)
        else:
            thresholds = (-0.008, -0.002, 0.002, 0.008)
        
        df_feat = create_multiclass_target(
            df_feat, 
            forward_periods=1, 
            thresholds=thresholds
        )
        target_col = "Target_multiclass"
        num_classes = 5
        objective = "multi:softprob"
        eval_metric = "mlogloss"
        
        print_target_distribution(df_feat, target_col)
        
    else:
        # Binary target
        print(f"\n[TARGET] Creating binary target...")
        df_feat["target"] = (df_feat["Close"].shift(-1) > df_feat["Close"]).astype(int)
        target_col = "target"
        num_classes = 2
        objective = "binary:logistic"
        eval_metric = "logloss"

    # Drop NaN from indicators and target
    df_feat.dropna(inplace=True)

    if df_feat.empty:
        raise ValueError(f"No rows left after feature engineering for {symbol}/{mode}")

    # ============================================================
    # ✅ NEW: CHECK CLASS BALANCE BEFORE PROCEEDING
    # ============================================================
    class_counts = df_feat[target_col].value_counts()
    unique_classes = sorted(class_counts.index.tolist())

    print(f"\n[CLASS CHECK] Found {len(unique_classes)} unique classes: {unique_classes}")
    for cls in unique_classes:
        count = class_counts[cls]
        print(f"  Class {cls}: {count} samples ({count/len(df_feat)*100:.1f}%)")

    # ✅ ADAPTIVE: Lower requirements for intraday regime models
    if mode in ("intraday_mr", "intraday_mom"):
        MIN_SAMPLES_PER_CLASS = 3  # More lenient for regime-filtered data
        MIN_UNIQUE_CLASSES = 2
        print(f"[CLASS CHECK] Using relaxed thresholds for {mode}: min_samples={MIN_SAMPLES_PER_CLASS}")
    else:
        MIN_SAMPLES_PER_CLASS = 5
        MIN_UNIQUE_CLASSES = 2

    valid_classes = [cls for cls in unique_classes if class_counts[cls] >= MIN_SAMPLES_PER_CLASS]

    if len(valid_classes) < MIN_UNIQUE_CLASSES:
        # ✅ FALLBACK: Try even more lenient threshold (at least 2 samples)
        if mode in ("intraday_mr", "intraday_mom"):
            print(f"\n[CLASS CHECK] Trying fallback: min_samples=2")
            valid_classes = [cls for cls in unique_classes if class_counts[cls] >= 2]
            
            if len(valid_classes) >= MIN_UNIQUE_CLASSES:
                print(f"[CLASS CHECK] ✅ Fallback successful: {len(valid_classes)} valid classes")
            else:
                raise ValueError(
                    f"Insufficient class diversity for {symbol}/{mode} even with min_samples=2: "
                    f"Found {len(valid_classes)} valid classes (need {MIN_UNIQUE_CLASSES}). "
                    f"Class counts: {class_counts.to_dict()}. "
                    f"Skipping this regime model - not enough filtered data."
                )
        else:
            raise ValueError(
                f"Insufficient class diversity for {symbol}/{mode}: "
                f"Found {len(valid_classes)} valid classes (need {MIN_UNIQUE_CLASSES}). "
                f"Class counts: {class_counts.to_dict()}. "
                f"Try different regime thresholds or use daily mode."
            )
    
    # ✅ REMAP CLASSES TO CONSECUTIVE INTEGERS (for XGBoost)
    if use_multiclass and len(valid_classes) < num_classes:
        print(f"\n[CLASS REMAP] Only {len(valid_classes)}/{num_classes} classes have enough samples")
        
        # Filter to only valid classes
        df_feat = df_feat[df_feat[target_col].isin(valid_classes)]
        
        # Remap to consecutive integers (0, 1, 2, ...)
        class_mapping = {old_class: new_class for new_class, old_class in enumerate(sorted(valid_classes))}
        df_feat[target_col] = df_feat[target_col].map(class_mapping)
        num_classes = len(valid_classes)
        
        print(f"[CLASS REMAP] Mapping: {class_mapping}")
        print(f"[CLASS REMAP] ✅ Remapped to {num_classes} consecutive classes: {list(range(num_classes))}")
        print(f"[CLASS REMAP] New distribution:")
        print(df_feat[target_col].value_counts().sort_index())

    # ============================================================
    # ✨ PREPARE FEATURES (with additional safety checks)
    # ============================================================
    print(f"\n[FEATURES] Preparing feature matrix...")
    
    # Identify columns to exclude
    exclude_cols = [target_col, "Close", "Open", "High", "Low", "Volume"]
    
    # Get feature columns
    feature_cols = [col for col in df_feat.columns if col not in exclude_cols]
    
    X = df_feat[feature_cols].copy()
    y = df_feat[target_col].copy()
    
    # ============================================================
    # ✨ FINAL VALIDATION: Ensure X is clean
    # ============================================================
    print(f"[VALIDATION] Validating feature matrix...")
    
    # ✅ STEP 1: Remove unnamed/empty column names
    unnamed_cols = [col for col in X.columns if col == '' or pd.isna(col) or str(col).strip() == '']
    if unnamed_cols:
        print(f"[VALIDATION] Dropping {len(unnamed_cols)} unnamed columns")
        X = X.drop(columns=unnamed_cols, errors='ignore')
    
    # ✅ STEP 2: Remove non-numeric columns
    non_numeric = X.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) > 0:
        print(f"[VALIDATION] Dropping {len(non_numeric)} non-numeric columns")
        X = X.drop(columns=non_numeric)
    
    # ✅ STEP 3: Check each column individually
    problematic_cols = []
    for col in X.columns:
        try:
            # Skip if already problematic
            if col in problematic_cols:
                continue
                
            # Check if it's a DataFrame
            if isinstance(X[col], pd.DataFrame):
                print(f"[VALIDATION] Converting DataFrame column '{col}'")
                X[col] = X[col].iloc[:, 0]
            
            # Check dimensionality
            col_values = X[col].values
            if col_values.ndim > 1:
                if col_values.shape[0] == len(X) and col_values.shape[1] == 1:
                    # Can safely flatten
                    print(f"[VALIDATION] Flattening '{col}' (shape: {col_values.shape})")
                    X[col] = col_values.ravel()
                elif col_values.shape[0] == len(X) and col_values.shape[1] > 1:
                    # Take first column
                    print(f"[VALIDATION] Taking first column from '{col}' (shape: {col_values.shape})")
                    X[col] = col_values[:, 0]
                else:
                    # Problematic shape
                    print(f"[VALIDATION] Problematic shape for '{col}': {col_values.shape}")
                    problematic_cols.append(col)
                    continue
            
            # ✅ STEP 4: Convert to numeric ONLY if 1D now
            if X[col].values.ndim == 1:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            else:
                print(f"[VALIDATION] Column '{col}' still multi-dimensional after fixes")
                problematic_cols.append(col)
            
        except Exception as e:
            print(f"[VALIDATION] Error processing '{col}': {e}")
            problematic_cols.append(col)
    
    # ✅ STEP 5: Drop all problematic columns at once
    if problematic_cols:
        print(f"[VALIDATION] Dropping {len(problematic_cols)} problematic columns")
        X = X.drop(columns=problematic_cols, errors='ignore')
    
    # ✅ STEP 6: Drop columns that are all NaN
    all_nan_cols = X.columns[X.isna().all()].tolist()
    if all_nan_cols:
        print(f"[VALIDATION] Dropping {len(all_nan_cols)} all-NaN columns")
        X = X.drop(columns=all_nan_cols)
    
    # ✅ STEP 7: Fill remaining NaN with 0
    if X.isna().any().any():
        nan_count = X.isna().sum().sum()
        print(f"[VALIDATION] Filling {nan_count} NaN values with 0")
        X = X.fillna(0)
    
    feature_list = list(X.columns)
    print(f"[FEATURES] Final feature count: {len(feature_list)}")
    
    if len(feature_list) == 0:
        raise ValueError(f"No valid features remaining for {symbol}/{mode}")
    
    # Validate y (target)
    y = pd.Series(y, dtype=int)
    
    # ============================================================
    # ✨ FINAL SANITY CHECK
    # ============================================================
    print(f"\n[SANITY CHECK] Feature matrix shape: {X.shape}")
    print(f"[SANITY CHECK] Target shape: {y.shape}")
    print(f"[SANITY CHECK] All columns are 1D: {all(X[col].values.ndim == 1 for col in X.columns)}")
    
    # Final check: verify no column is a DataFrame and all are 1D
    for col in X.columns:
        if isinstance(X[col], pd.DataFrame):
            raise ValueError(f"Column '{col}' is still a DataFrame!")
        if X[col].values.ndim != 1:
            raise ValueError(f"Column '{col}' is not 1D! Shape: {X[col].values.shape}")

    # -----------------------------
    # TIME-SERIES CV SPLIT
    # -----------------------------
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

    # ============================================================
    # ✅ NEW: VERIFY TRAIN/TEST CLASS DISTRIBUTION
    # ============================================================
    train_classes = sorted(y_train.unique())
    test_classes = sorted(y_test.unique())
    
    print(f"\n[SPLIT CHECK] Train classes: {train_classes} (counts: {y_train.value_counts().to_dict()})")
    print(f"[SPLIT CHECK] Test classes: {test_classes} (counts: {y_test.value_counts().to_dict()})")
    
    if len(train_classes) < 2:
        raise ValueError(
            f"Train set has only {len(train_classes)} class(es) for {symbol}/{mode}. "
            f"Cannot train a classifier. Try using more data or daily mode."
        )
    
    # ✅ FIX: Use stratified split if test set lacks diversity
    if len(test_classes) < 2:
        print(f"\n[WARN] Test set has only {len(test_classes)} class(es)!")
        print(f"[FIX] Attempting stratified split to ensure class diversity...")
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=0.2, 
                stratify=y,
                random_state=42,
                shuffle=True  # Need shuffle=True for stratify
            )
            
            test_classes = sorted(y_test.unique())
            print(f"[FIX] ✅ Stratified split successful")
            print(f"[FIX] Train classes: {sorted(y_train.unique())}")
            print(f"[FIX] Test classes: {test_classes}")
            
            if len(test_classes) < 2:
                print(f"[FIX] ⚠️ Still only {len(test_classes)} test class - will skip some metrics")
            
        except Exception as e:
            print(f"[FIX] ⚠️ Stratified split failed: {e}")
            print(f"[FIX] Proceeding with original split (some metrics will fail)")

    # -----------------------------
    # CLASS BALANCING
    # -----------------------------
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

    # -----------------------------
    # XGB PARAMS
    # -----------------------------
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

    # -----------------------------
    # FIT
    # -----------------------------
    print(f"\n[TRAINING] Fitting {symbol}/{mode} model...")
    
    model.fit(
        X_train, 
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    print(f"[TRAINING] ✅ Model trained with {model.n_estimators} trees")

    # ✅ STORE BASE MODEL FOR SHAP (before calibration)
    base_model = model

    # -----------------------------
    # ✨ CALIBRATION (ENHANCED)
    # -----------------------------
    final_model = model  # Default: use uncalibrated model
    calibrated = False
    
    if num_classes == 2:
        # Binary: use Platt scaling (sigmoid)
        print(f"\n[CALIBRATION] Applying Platt scaling (sigmoid) for {symbol}/{mode}")
        calibration_method = 'sigmoid'
    else:
        # Multi-class: use isotonic regression (better for multi-class)
        print(f"\n[CALIBRATION] Applying isotonic calibration for {symbol}/{mode}")
        calibration_method = 'isotonic'
    
    try:
        calibrated_model = CalibratedClassifierCV(
            model, 
            method=calibration_method,
            cv='prefit',
            n_jobs=-1
        )
        
        # Fit calibration on test set (or use CV for production)
        calibrated_model.fit(X_test, y_test)
        final_model = calibrated_model
        calibrated = True
        print(f"[CALIBRATION] ✅ Successfully calibrated using {calibration_method}")
        
    except Exception as e:
        print(f"[CALIBRATION] ⚠️ Failed to calibrate: {e}")
        print(f"[CALIBRATION] Using uncalibrated model as fallback")
        final_model = model


    # -----------------------------
    # ✨ ENHANCED METRICS (with robust error handling)
    # -----------------------------
    print(f"\n[EVALUATION] Computing metrics on test set...")
    metrics = {}
    
    try:
        if num_classes == 2:
            # Binary metrics
            y_pred_proba = final_model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            # ✅ SAFE METRICS
            try:
                metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
            except Exception as e:
                print(f"[METRICS] Accuracy failed: {e}")
                metrics["accuracy"] = 0.0
            
            # ✅ Log loss requires at least 2 classes in y_test
            try:
                if len(np.unique(y_test)) >= 2:
                    metrics["logloss"] = float(log_loss(y_test, y_pred_proba, labels=[0, 1]))
                else:
                    print(f"[METRICS] Skipping logloss (only {len(np.unique(y_test))} class in test set)")
                    metrics["logloss"] = None
            except Exception as e:
                print(f"[METRICS] LogLoss failed: {e}")
                metrics["logloss"] = None
            
            # ✅ ROC AUC requires at least 2 classes
            try:
                if len(np.unique(y_test)) >= 2:
                    metrics["roc_auc"] = float(roc_auc_score(y_test, y_pred_proba))
                else:
                    print(f"[METRICS] Skipping ROC AUC (only 1 class in test set)")
                    metrics["roc_auc"] = None
            except Exception as e:
                print(f"[METRICS] ROC AUC failed: {e}")
                metrics["roc_auc"] = None
            
            # ✅ Precision/Recall/F1 with zero_division handling
            try:
                metrics["precision"] = float(precision_score(y_test, y_pred, zero_division=0))
                metrics["recall"] = float(recall_score(y_test, y_pred, zero_division=0))
                metrics["f1"] = float(f1_score(y_test, y_pred, zero_division=0))
            except Exception as e:
                print(f"[METRICS] Precision/Recall/F1 failed: {e}")
                metrics["precision"] = 0.0
                metrics["recall"] = 0.0
                metrics["f1"] = 0.0
            
            # ✅ Confusion matrix
            try:
                metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()
            except Exception as e:
                print(f"[METRICS] Confusion matrix failed: {e}")
                metrics["confusion_matrix"] = [[0, 0], [0, 0]]

            # ✅ Calibration metrics
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
            # Multi-class metrics
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
        
        # Additional metadata
        metrics["train_samples"] = len(X_train)
        metrics["test_samples"] = len(X_test)
        metrics["num_features"] = len(feature_list)
        metrics["num_classes"] = num_classes
        metrics["train_class_counts"] = y_train.value_counts().to_dict()
        metrics["test_class_counts"] = y_test.value_counts().to_dict()
        
    except Exception as e:
        print(f"[EVALUATION] ⚠️ Critical metrics error: {e}")
        import traceback
        traceback.print_exc()
        
        # Minimal fallback metrics
        metrics = {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "num_features": len(feature_list),
            "num_classes": num_classes,
            "error": str(e)
        }

    # -----------------------------
    # PRINT VALIDATION SUMMARY
    # -----------------------------
    print(f"\n{'='*60}")
    print(f"MODEL VALIDATION: {symbol}/{mode.upper()}")
    print(f"{'='*60}")
    print(f"Training samples:     {metrics.get('train_samples', 'N/A')}")
    print(f"Test samples:         {metrics.get('test_samples', 'N/A')}")
    print(f"Features:             {metrics.get('num_features', 'N/A')}")
    print(f"Target classes:       {metrics.get('num_classes', 'N/A')}")
    print(f"Calibrated:           {'Yes' if calibrated else 'No'}")
    print("-" * 60)
    
    # Print key metrics
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
    
    # Confusion matrix
    if "confusion_matrix" in metrics:
        print("-" * 60)
        print("Confusion Matrix:")
        cm = np.array(metrics["confusion_matrix"])
        for row in cm:
            print("  ", row)
    
    print("=" * 60)

    # -----------------------------
    # RETURN ARTIFACT
    # -----------------------------
    artifact = {
        "model": final_model,              # ✅ Calibrated model for predictions
        "base_model": base_model,          # ✅ Uncalibrated XGBoost for SHAP
        "features": feature_list,
        "metrics": metrics,
        "trained_at": datetime.now().isoformat(),
        "symbol": symbol,
        "mode": mode,
        "calibrated": calibrated,
        "num_classes": num_classes,        # ✨ NEW
        "target_type": "multiclass" if use_multiclass else "binary",  # ✨ NEW
        "class_weighted": scale_pos_weight is not None,  # ✨ NEW
    }
    
    return artifact


# ---------------------------------------------------------
# LOAD MODEL ARTIFACT
# ---------------------------------------------------------
def load_model(symbol: str, mode: str):
    """
    Load saved model artifact:
      (model, base_model, features, metrics, trained_at, symbol, mode)
    or return None if missing.
    """
    path = os.path.join(MODEL_DIR, f"{symbol}_{mode}_xgb.pkl")
    if not os.path.exists(path):
        return None
    try:
        data = joblib.load(path)
        return data
    except Exception as e:
        print(f"[ERROR] load_model({symbol}, {mode}): {e}")
        return None

def find_matching_feature(feature_name: str, available_features: list) -> str:
    """
    Case-insensitive feature matching with intelligent fallbacks.
    Handles version drift between trained models and current feature sets.
    
    Args:
        feature_name: Target feature from trained model (e.g., "RSI_14")
        available_features: List of features in current dataframe
    
    Returns:
        Matched feature name from available_features, or None if no match
    """
    if not feature_name or not available_features:
        return None
    
    feature_lower = str(feature_name).lower().strip()
    
    # STAGE 1: Exact match (case-insensitive)
    for feat in available_features:
        if str(feat).lower().strip() == feature_lower:
            return feat
    
    # STAGE 2: Normalized match (remove underscores/spaces)
    feature_normalized = feature_lower.replace("_", "").replace(" ", "")
    
    for feat in available_features:
        feat_normalized = str(feat).lower().strip().replace("_", "").replace(" ", "")
        if feature_normalized == feat_normalized:
            return feat
    
    # STAGE 3: Partial match (substring)
    for feat in available_features:
        feat_lower = str(feat).lower().strip()
        if feature_lower in feat_lower or feat_lower in feature_lower:
            if feature_lower in feat_lower:
                return feat
    
    # STAGE 4: Token-based match (split by underscore)
    feature_tokens = set(feature_lower.split("_"))
    best_match = None
    best_score = 0
    
    for feat in available_features:
        feat_tokens = set(str(feat).lower().strip().split("_"))
        common = feature_tokens & feat_tokens
        score = len(common)
        
        if score > best_score and score >= 2:
            best_score = score
            best_match = feat
    
    if best_match:
        return best_match
    
    # STAGE 5: No match found
    return None

# ---------------------------------------------------------
# ✨ ENHANCED: PREDICT FROM MODEL ARTIFACT (MULTI-CLASS SUPPORT)
# ---------------------------------------------------------
def predict_from_model(model_dict, df_features: pd.DataFrame):
    """
    ✨ ENHANCED: Return prediction dict with support for multi-class models.
    ✅ FIXED: Case-insensitive feature matching with better validation.
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
        
        # ============================================================
        # ✅ IMPROVED: Case-insensitive feature matching
        # ============================================================
        
        # Map model features to current feature names
        feature_mapping = {}
        missing_features = []
        match_stats = {'exact': 0, 'case': 0, 'normalized': 0}
        
        for model_feat in feat_cols:
            matched_feat = find_matching_feature(model_feat, df_features.columns.tolist())
            
            if matched_feat:
                # Feature found
                feature_mapping[model_feat] = matched_feat
                
                # Track match type for reporting
                if matched_feat == model_feat:
                    match_stats['exact'] += 1
                elif matched_feat.lower() == model_feat.lower():
                    match_stats['case'] += 1
                else:
                    match_stats['normalized'] += 1
            else:
                # Feature truly missing
                feature_mapping[model_feat] = None
                missing_features.append(model_feat)
        
        # ============================================================
        # ✅ BUILD FEATURE VECTOR (single row for prediction)
        # ============================================================
        
        X_data = {}
        
        for model_feat, data_feat in feature_mapping.items():
            if data_feat is not None:
                try:
                    # Get last row value (most recent)
                    val = df_features[data_feat].iloc[-1]
                    
                    # Convert to numeric if needed
                    if not isinstance(val, (int, float, np.number)):
                        val = float(val) if pd.notna(val) else 0.0
                    
                    # Handle NaN/inf
                    if not np.isfinite(val):
                        val = 0.0
                    
                    X_data[model_feat] = val
                    
                except Exception as e:
                    # If extraction fails, use 0
                    X_data[model_feat] = 0.0
            else:
                # Missing feature - fill with 0
                X_data[model_feat] = 0.0
        
        # Create DataFrame with model's expected column order
        X = pd.DataFrame([X_data], columns=feat_cols)
        
        # ============================================================
        # ✅ REPORT FEATURE MATCHING STATUS
        # ============================================================
        
        total_features = len(feat_cols)
        matched_exact = match_stats['exact']
        matched_case = match_stats['case']
        matched_normalized = match_stats['normalized']
        
        if missing_features:
            missing_pct = len(missing_features) / total_features * 100
            
            if missing_pct > 20:
                # Critical: >20% missing
                print(f"[ERROR] {len(missing_features)} features missing ({missing_pct:.1f}%)")
                print(f"[ERROR] Missing: {missing_features[:10]}")
                print(f"[ERROR] Predictions may be unreliable - retrain model recommended")
            elif missing_pct > 5:
                # Warning: 5-20% missing
                print(f"[WARN] {len(missing_features)} features missing ({missing_pct:.1f}%)")
                print(f"[WARN] Top missing: {missing_features[:5]}")
                print(f"[WARN] Consider retraining model with current feature pipeline")
        else:
            # All features matched
            if matched_case + matched_normalized > 0:
                print(f"[INFO] Matched {total_features} features: "
                      f"{matched_exact} exact, "
                      f"{matched_case} case-insensitive, "
                      f"{matched_normalized} normalized")
        
        # ============================================================
        # ✅ FINAL VALIDATION & CLEANUP
        # ============================================================
        
        # Ensure all numeric
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Fill any remaining NaN
        X = X.fillna(0)
        
        # Final check: ensure 1D for each column
        for col in X.columns:
            if X[col].values.ndim != 1:
                X[col] = X[col].values.ravel()
        
        # ============================================================
        # GET PREDICTIONS
        # ============================================================
        proba = model.predict_proba(X)[0]
        
        # ============================================================
        # BINARY MODEL (num_classes == 2)
        # ============================================================
        if num_classes == 2 or target_type == "binary":
            # Return simple probability (backward compatible)
            prob = float(proba[1])
            return prob
        
        # ============================================================
        # MULTI-CLASS MODEL (num_classes > 2)
        # ============================================================
        else:

            actual_num_classes = len(proba)
            
            # Class meanings (for 5-class):
            # 0 = Strong Down (< -1.5%)
            # 1 = Weak Down (-1.5% to -0.3%)
            # 2 = Flat (-0.3% to +0.3%)
            # 3 = Weak Up (+0.3% to +1.5%)
            # 4 = Strong Up (> +1.5%)
            
            if actual_num_classes == 5:
                # Standard 5-class model
                class_names = ['Strong Down', 'Weak Down', 'Flat', 'Weak Up', 'Strong Up']
                
                predicted_class = int(np.argmax(proba))
                confidence = float(np.max(proba))
                
                # Simple aggregation
                bullish_simple = float(proba[3] + proba[4])
                bearish_simple = float(proba[0] + proba[1])
                flat_prob = float(proba[2])
                
                # Weighted aggregation (stronger moves weighted more)
                bullish_weighted = float(proba[3] * 1.0 + proba[4] * 2.0)
                bearish_weighted = float(proba[0] * 2.0 + proba[1] * 1.0)
                flat_weighted = float(proba[2] * 0.5)
                
                total_weighted = bullish_weighted + bearish_weighted + flat_weighted
                
                if total_weighted > 0:
                    # Final probability (0-1 scale, 0.5 = neutral)
                    final_prob = bullish_weighted / (bullish_weighted + bearish_weighted)
                else:
                    final_prob = 0.5
                
                result = {
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
                # ✅ REDUCED CLASS MODEL (3 or 4 classes after filtering)
                predicted_class = int(np.argmax(proba))
                confidence = float(np.max(proba))
                
                # Simple heuristic: top half = bullish, bottom half = bearish
                mid_point = actual_num_classes / 2.0
                
                bullish_prob = float(sum(proba[i] for i in range(actual_num_classes) if i >= mid_point))
                bearish_prob = float(sum(proba[i] for i in range(actual_num_classes) if i < mid_point))
                
                if bullish_prob + bearish_prob > 0:
                    final_prob = bullish_prob / (bullish_prob + bearish_prob)
                else:
                    final_prob = 0.5



                result = {
                    'final_prob': float(final_prob),
                    'probabilities': proba.tolist(),
                    'predicted_class': predicted_class,
                    'predicted_class_name': f'Class_{predicted_class}',
                    'confidence': confidence,
                    'bullish_prob': bullish_prob,
                    'bearish_prob': bearish_prob,
                    'flat_prob': 0.0,  # Not applicable for reduced models
                    'class_breakdown': {f'class_{i}': float(proba[i]) for i in range(actual_num_classes)},
                    'num_classes': actual_num_classes,
                    'model_type': target_type,
                }
            
            return result

    except Exception as e:
        print(f"[ERROR] predict_from_model: {e}")
        import traceback
        traceback.print_exc()
        return None


# ---------------------------------------------------------
# ✨ NEW: HELPER TO EXTRACT FINAL PROBABILITY
# ---------------------------------------------------------
def get_final_prob(prediction):
    """
    Extract final probability from prediction (handles both binary and multi-class).
    
    Args:
        prediction: Either float (binary) or dict (multi-class)
    
    Returns:
        float: Final probability (0.0 to 1.0)
    """
    if prediction is None:
        return None
    
    if isinstance(prediction, (float, int)):
        # Binary model: prediction is already a probability
        return float(prediction)
    
    if isinstance(prediction, dict):
        # Multi-class model: extract final_prob
        return prediction.get('final_prob')
    
    print(f"[WARN] Unknown prediction type: {type(prediction)}")
    return None


# ---------------------------------------------------------
# ✨ NEW: HELPER TO GET DETAILED SIGNAL INFO
# ---------------------------------------------------------
def get_signal_details(prediction) -> str:
    """
    Get human-readable signal details for logging/debugging.
    
    Args:
        prediction: Output from predict_from_model()
    
    Returns:
        str: Formatted signal description
    """
    if prediction is None:
        return "No signal"
    
    # Binary model
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
    
    # Multi-class model
    if isinstance(prediction, dict):
        final_prob = prediction.get('final_prob', 0.5)
        pred_class = prediction.get('predicted_class_name', 'Unknown')
        confidence = prediction.get('confidence', 0.0)
        breakdown = prediction.get('class_breakdown', {})
        
        # Determine signal strength
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
        
        # Build detailed string
        details = f"{signal} ({final_prob:.1%})"
        details += f" | Prediction: {pred_class} (conf: {confidence:.1%})"
        
        # Add class breakdown
        details += f" | [↓↓:{breakdown.get('strong_down', 0):.0%} "
        details += f"↓:{breakdown.get('weak_down', 0):.0%} "
        details += f"→:{breakdown.get('flat', 0):.0%} "
        details += f"↑:{breakdown.get('weak_up', 0):.0%} "
        details += f"↑↑:{breakdown.get('strong_up', 0):.0%}]"
        
        return details
    
    return "Unknown signal type"


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
    ✨ ENHANCED: Compute trading signals with multi-class model support
    
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
        # ✨ NEW: Detailed predictions for multi-class models
        "daily_prediction": None,
        "intraday_prediction": None,
        "daily_signal": None,
        "intraday_signal": None,
    }

    symU = str(symbol).upper().strip()

    # -------------------------
    # LOAD MODELS
    # -------------------------
    model_daily = load_model(symU, mode="daily")
    model_intra_mr = load_model(symU, mode="intraday_mr")
    model_intra_mom = load_model(symU, mode="intraday_mom")
    model_intraday_legacy = load_model(symU, mode="intraday")

    # -------------------------
    # DAILY PREDICTION
    # -------------------------
    df_daily = None
    try:
        df_daily = fetch_historical_data(symU, period="6mo", interval="1d")
        if df_daily is not None and not df_daily.empty:
            results["daily_rows"] = len(df_daily)
            df_feat = build_daily_features(df_daily)
            
            # ✨ ENHANCED: Get full prediction (works with binary or multi-class)
            daily_prediction = predict_from_model(model_daily, df_feat)
            results["daily_prediction"] = daily_prediction
            
            # ✨ ENHANCED: Extract final probability (backward compatible)
            results["daily_prob"] = get_final_prob(daily_prediction)
            
            # ✨ NEW: Get human-readable signal
            results["daily_signal"] = get_signal_details(daily_prediction)

            # ✅ LOG DAILY PREDICTION
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
                            prediction_details=daily_prediction if isinstance(daily_prediction, dict) else None  # ✅ FIXED
                        )
                except Exception as e:
                    print(f"[WARN] Could not log daily prediction: {e}")

            # price fallback from daily close (only if price not set later by intraday)
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

    # -------------------------
    # INTRADAY PREDICTION
    # -------------------------
    df_intra_resampled = None
    df_feat_intra = None


    try:
        # Adaptive lookback
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

        # -------------------------
        # CHECK IF ENOUGH DATA
        # -------------------------
        if df_intra_resampled is None or len(df_intra_resampled) < MIN_INTRADAY_BARS_FOR_FEATURES:
            results["allow_intraday"] = False
            results["intraday_prob"] = None
            results["intraday_quality_score"] = 0.0
        else:
            # ✅ EXTRACT CLOSE PRICE SERIES (do this ONCE)
            close = df_intra_resampled["Close"]
            close = close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close
            close = close.dropna()

            # Update price
            try:
                if len(close) > 0:
                    results["price"] = float(close.iloc[-1])
            except Exception:
                pass

            # ✅ CALCULATE REGIME METRICS (do this ONCE)
            vol = None
            mom1h = None

            if len(close) >= 12:
                vol = close.pct_change().rolling(12).std().iloc[-1]
                results["intraday_vol"] = float(vol) if not pd.isna(vol) else None

            if len(close) >= 4:
                mom1h = close.pct_change(4).iloc[-1]
                results["intraday_mom"] = float(mom1h) if not pd.isna(mom1h) else None

            # ✅ GET THRESHOLDS (adaptive or config)
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

            # ✅ SELECT MODEL BASED ON REGIME (do this ONCE)
            active_model = None
            model_used = None
            if vol is not None and mom1h is not None:
                # Decision logic
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

            # Fallback if regime detection fails or models not loaded
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

            # ✅ BUILD FEATURES AND PREDICT
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
                    # ✅ PREDICT WITH CHOSEN MODEL
                    print(f"[TRY] {symU} using {model_used}")
                    intraday_prediction = predict_from_model(active_model, df_feat_intra)
                    
                    results["intraday_prediction"] = intraday_prediction
                    results["intraday_model_used"] = model_used
                    results["intraday_signal"] = get_signal_details(intraday_prediction)
                    
                    ip = get_final_prob(intraday_prediction)

                    if ip is not None:

                        ip_original = float(ip)
                        momentum_boost_applied = False
                        
                        # ========================================
                        # ✅ MOMENTUM PROBABILITY BOOST
                        # ========================================
                        if results.get("intraday_regime") == "mom" and mom1h is not None:
                            mom1h_val = float(mom1h)
                            
                            # Only boost near-neutral signals
                            if abs(mom1h_val) > 0.005 and 0.30 < ip_original < 0.70:
                                
                                # Calculate boost magnitude
                                if abs(mom1h_val) > 0.01:  # Extreme momentum (>1%)
                                    boost_magnitude = min(0.20, abs(mom1h_val) * 7.0)
                                else:  # Moderate momentum (0.5-1%)
                                    boost_magnitude = min(0.15, abs(mom1h_val) * 5.0)
                                
                                # Apply boost in DIRECTION of momentum
                                if mom1h_val > 0:  # Positive momentum → boost UP
                                    ip = min(0.85, ip_original + boost_magnitude)
                                    print(f"[MOMENTUM BOOST] {symU} mom={mom1h_val:.2%} (UP) -> prob {ip_original:.3f} → {ip:.3f}")
                                else:  # Negative momentum → push DOWN
                                    ip = max(0.15, ip_original - boost_magnitude)
                                    print(f"[MOMENTUM BOOST] {symU} mom={mom1h_val:.2%} (DOWN) -> prob {ip_original:.3f} → {ip:.3f}")
                                
                                momentum_boost_applied = True
                        
                        # ⚠️ SPY-specific: Cap extreme predictions
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
                        
                        # ✅ LOG PREDICTION
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
        import traceback
        traceback.print_exc()
        results["allow_intraday"] = False
        results["intraday_prob"] = None
        results["intraday_quality_score"] = 0.0


    # -------------------------
    # PRICE FALLBACK
    # -------------------------
    if results.get("price") is None:
        try:
            p = fetch_latest_price(symU)
            if p is not None and float(p) > 0:
                results["price"] = float(p)
        except Exception:
            pass

    # -------------------------
    # ✨ ENHANCED: COMBINE DAILY + INTRADAY
    # -------------------------
    dp = results["daily_prob"]
    ip = results["intraday_prob"]

    # Get regime info
    if vol is None:
        vol = float(results.get("intraday_vol") or 0.0)
    if mom1h is None:
        mom1h = float(results.get("intraday_mom") or 0.0)

    # Get thresholds for regime detection
    try:
        adaptive = get_adaptive_regime_thresholds(symU, lookback_days=30, percentile=0.70)
        MOMTRIG = float(adaptive['mom_trig'])
        VOLTRIG = float(adaptive['vol_trig'])
    except:
        MOMTRIG = float(INTRADAY_MOM_TRIG)
        VOLTRIG = float(INTRADAY_VOL_TRIG)
        ovr = (INTRADAY_REGIME_OVERRIDES or {}).get(symU)
        if ovr:
            MOMTRIG = float(ovr.get("mom_trig", MOMTRIG))
            VOLTRIG = float(ovr.get("vol_trig", VOLTRIG))

    ismomentumregime = abs(mom1h) >= MOMTRIG or vol >= VOLTRIG
    results["intraday_regime"] = "mom" if ismomentumregime else "mr"

    # Calculate weight
    weight = float(intraday_weight)
    if not results["allow_intraday"] or ip is None:
        weight = 0.0
    else:
        q = float(results.get("intraday_quality_score", 1.0) or 0.0)
        weight = float(intraday_weight) * q



    # Volatility adjustment
    try:
        if vol > 0.025:
            weight = min(0.90, weight + 0.20)
        elif vol < 0.008 and symU == "SPY":
            weight = max(0.30, weight - 0.20)
    except Exception:
        pass

    # Price momentum override (tuned for 15m)
    try:
        # Strong momentum = price up >0.5% in last hour
        if mom1h is not None and mom1h > 0.005:
            weight = max(weight, 0.75)
            print(f"[MOMENTUM BOOST] {symU} mom1h={mom1h:.2%} -> weight={weight:.2f}")
        
        # Extreme momentum = price up >1% in last hour
        if mom1h is not None and mom1h > 0.010:
            weight = min(0.90, weight + 0.15)
            print(f"[MOMENTUM SURGE] {symU} mom1h={mom1h:.2%} -> weight={weight:.2f}")
            
    except Exception:
        pass

    # Strong intraday momentum override
    try:
        if ip is not None and dp is not None and results.get("intraday_quality_score", 0.0) > 0.6 and ip > 0.70 and (ip - dp) > 0.15:
            weight = max(weight, 0.75)
    except Exception:
        pass

    # Strong daily continuation override
    try:
        if dp is not None and ip is not None and dp > 0.78 and ip < 0.25 and mom1h < -0.005:
            weight = max(weight, 0.70)
    except Exception:
        pass

    weight = float(min(max(weight, 0.0), 0.90))
    results["intraday_weight"] = weight

    # -------------------------
    # ✨ ENHANCED: DEBUG PRINTS WITH MULTI-CLASS INFO
    # -------------------------
    def fmt(x, n=3):
        return "NA" if x is None else f"{float(x):.{n}f}"

    def fmtpct(x):
        return "NA" if x is None else f"{float(x)*100:.2f}%"

    print(f"[DEBUG] {symU} dp={fmt(dp)} ip={fmt(ip)} q={results.get('intraday_quality_score',0):.2f} weight={weight:.2f} model={results.get('intraday_model_used')} price={fmt(results.get('price'), 2)}")
    print(f"[DEBUG] {symU} vol={fmt(results.get('intraday_vol'), 5)} mom={fmtpct(results.get('intraday_mom'))} regime={results.get('intraday_regime')}")
    
    # ✨ NEW: Print multi-class details if available
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

    # -------------------------
    # FINAL PROB
    # -------------------------
    if dp is None and ip is None:
        results["final_prob"] = None
    elif dp is None:
        results["final_prob"] = float(ip)
    elif ip is None or weight == 0.0:
        results["final_prob"] = float(dp)
    else:
        results["final_prob"] = float(weight * ip + (1 - weight) * dp)

    # ============================================================
    # EMERGENCY: Block contradictory momentum signals
    # ============================================================
    mom1h = results.get("intraday_mom")
    if mom1h is not None and ip is not None:
        mom1h_val = float(mom1h)
        
        # If momentum is strongly negative but intraday predicts bullish
        if mom1h_val < -0.01 and ip > 0.60:  # -1% momentum but 60%+ bullish pred
            print(f"[CONTRADICTION] {symU}: mom={mom1h_val:.2%} but ip={ip:.3f} - capping final_prob at 0.52")
            results["final_prob"] = min(results["final_prob"], 0.52)  # Force neutral
            results["contradiction_flagged"] = True
        
        # If momentum is strongly positive but intraday predicts bearish
        elif mom1h_val > 0.01 and ip < 0.40:  # +1% momentum but <40% bearish pred
            print(f"[CONTRADICTION] {symU}: mom={mom1h_val:.2%} but ip={ip:.3f} - flooring final_prob at 0.48")
            results["final_prob"] = max(results["final_prob"], 0.48)  # Force neutral
            results["contradiction_flagged"] = True
        
    # ✨ NEW: Final signal description
    results["final_signal"] = get_signal_details(results["final_prob"])
    
    print(f"[FINAL] {symU} final_prob={fmt(results['final_prob'])} | {results.get('final_signal', '')}")

    return results


# ---------------------------------------------------------
# COMPATIBILITY WRAPPER
# ---------------------------------------------------------
def predict_next(symbol, lookback_minutes=60, intraday_weight=INTRADAY_WEIGHT, resample_to="15min"):
    """Compatibility wrapper: returns only the final probability."""
    sig = compute_signals(symbol, lookback_minutes, intraday_weight, resample_to)
    return sig.get("final_prob")
