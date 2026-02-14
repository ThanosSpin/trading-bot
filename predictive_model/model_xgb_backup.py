# model_xgb.py
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from features import build_daily_features, build_intraday_features
from data_loader import fetch_historical_data, fetch_intraday_history, fetch_latest_price
from target_labels import create_target_label, backtest_threshold
from trading_metrics import calculate_financial_metrics, print_trading_report

from config.config import (
    INTRADAY_WEIGHT,
    MIN_INTRADAY_BARS_FOR_FEATURES,
    INTRADAY_MOM_TRIG,
    INTRADAY_VOL_TRIG,
    INTRADAY_REGIME_OVERRIDES,
)

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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
    Filters intraday feature rows into either mean-reversion regime or momentum regime.
    The goal is two specialized models trained on different market conditions.
    """
    df_feat = _add_intraday_regime_cols(df_feat)

    df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
    df_feat = df_feat.dropna(subset=["mom_12_abs", "vol_12"])

    # --- default thresholds ---
    # momentum: >= ~1% move over ~3h OR higher short-term vol
    MOM_THR = 0.010
    VOL_THR = 0.010

    if mode == "intraday_mom":
        return df_feat[(df_feat["mom_12_abs"] >= MOM_THR) | (df_feat["vol_12"] >= VOL_THR)]

    if mode == "intraday_mr":
        return df_feat[(df_feat["mom_12_abs"] < MOM_THR) & (df_feat["vol_12"] < VOL_THR)]

    return df_feat

def train_model_with_validation(
    df: pd.DataFrame,
    symbol: str,
    mode: str = "daily",
    train_pct: float = 0.7,
    val_pct: float = 0.15,
):
    """
    Train model with proper walk-forward validation.
    
    Split:
    - Train: 70% (oldest data)
    - Validation: 15% (tune threshold)
    - Test: 15% (final holdout - NEVER TOUCH during training)
    """
    from sklearn.metrics import (
        accuracy_score, log_loss, roc_auc_score,
        confusion_matrix, precision_score, recall_score, f1_score
    )
    from target_labels import create_target_label, backtest_threshold
    from trading_metrics import calculate_financial_metrics, print_trading_report
    
    df = df.copy()
    
    # -----------------------------
    # FEATURE GENERATION
    # -----------------------------
    if mode == "daily":
        df_feat = build_daily_features(df)
    elif mode in ("intraday", "intraday_mr", "intraday_mom"):
        df_feat = build_intraday_features(df)
        
        # Regime filtering
        if mode in ("intraday_mr", "intraday_mom"):
            df_feat = _filter_intraday_rows_by_mode(df_feat, mode=mode)
            for c in ["ret_12", "mom_12_abs", "vol_12"]:
                if c in df_feat.columns:
                    df_feat.drop(columns=[c], inplace=True)
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    # -----------------------------
    # COST-AWARE TARGET
    # -----------------------------
    df_feat = create_target_label(df_feat, mode=mode)
    df_feat = df_feat.dropna(subset=["target", "forward_return"])
    
    if df_feat.empty:
        raise ValueError(f"No rows after feature engineering for {symbol} ({mode})")
    
    # Store returns for threshold optimization
    forward_returns = df_feat["forward_return"].copy()
    
    # ✅ Separate features and target (safe column drop)
    X = df_feat.drop(
        columns=["target", "forward_return", "target_3class"], 
        errors='ignore'  # Skip columns that don't exist
    )
    y = df_feat["target"]
    feature_list = list(X.columns)
    
    # -----------------------------
    # WALK-FORWARD SPLIT
    # -----------------------------
    n = len(X)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))
    
    if train_end < 50 or (val_end - train_end) < 20 or (n - val_end) < 20:
        raise ValueError(
            f"Insufficient data for {symbol} ({mode}): "
            f"train={train_end}, val={val_end-train_end}, test={n-val_end}"
        )
    
    # Split data chronologically (NO SHUFFLE)
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    ret_train = forward_returns.iloc[:train_end]
    
    X_val = X.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]
    ret_val = forward_returns.iloc[train_end:val_end]
    
    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]
    ret_test = forward_returns.iloc[val_end:]
    
    print(f"\n📊 DATA SPLIT — {symbol} [{mode.upper()}]")
    print(f"Train: {len(X_train)} bars ({X_train.index[0]} → {X_train.index[-1]})")
    print(f"Val:   {len(X_val)} bars ({X_val.index[0]} → {X_val.index[-1]})")
    print(f"Test:  {len(X_test)} bars ({X_test.index[0]} → {X_test.index[-1]})")
    print(f"Class balance - Train: {y_train.mean():.2%} | Val: {y_val.mean():.2%} | Test: {y_test.mean():.2%}")
    
    # -----------------------------
    # HYPERPARAMETERS
    # -----------------------------
    params = {
        "eval_metric": "logloss",
        "tree_method": "hist",
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "max_depth": 6,
        "random_state": 42,
    }
    
    if mode == "daily":
        params.update({"n_estimators": 400, "learning_rate": 0.04})
    else:
        params.update({"n_estimators": 500, "learning_rate": 0.035})
    
    # -----------------------------
    # TRAIN MODEL
    # -----------------------------
    model = XGBClassifier(**params)
    
    print(f"🔄 Training {symbol} [{mode}] with {params['n_estimators']} trees...")
    model.fit(X_train, y_train, verbose=False)
    print(f"✅ Training complete.")
    
    # -----------------------------
    # VALIDATION: FIND OPTIMAL THRESHOLD
    # -----------------------------
    y_val_proba = model.predict_proba(X_val)[:, 1]
    
    # ✅ Updated threshold search with better parameters
    optimal_threshold_info = backtest_threshold(
        y_val, 
        y_val_proba, 
        ret_val, 
        threshold_range=(0.45, 0.60),  # ✅ Lower range
        min_trades=15,                  # ✅ Require minimum trades
        optimization_metric="composite" # ✅ Balance profit + recall
    )
    
    optimal_threshold = optimal_threshold_info.get("threshold", 0.5)
    
    print(f"\n🎯 VALIDATION RESULTS (Optimal Threshold={optimal_threshold:.2f})")
    print("--------------------------------------------")
    for k, v in optimal_threshold_info.items():
        if isinstance(v, float):
            print(f"{k:20}: {v:.4f}")
        else:
            print(f"{k:20}: {v}")
    
    # -----------------------------
    # TEST: FINAL EVALUATION (NEVER SEEN DURING TRAINING)
    # -----------------------------
    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba > optimal_threshold).astype(int)
    
    # ML Metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_test_pred)),
        "logloss": float(log_loss(y_test, y_test_proba)),
        "roc_auc": float(roc_auc_score(y_test, y_test_proba)),
        "precision": float(precision_score(y_test, y_test_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_test_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_test_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_test_pred).tolist(),
        "optimal_threshold": optimal_threshold,
    }
    
    # Financial Metrics on Test Set
    financial_metrics = calculate_financial_metrics(
        y_true=y_test.values,
        y_pred=y_test_pred,
        y_pred_proba=y_test_proba,
        returns=ret_test.values,
        initial_capital=1000.0,
        commission=0.001,  # 0.1% round-trip
    )
    
    # Print detailed report
    print_trading_report(financial_metrics)
    
    # Merge financial metrics into main metrics dict
    metrics.update(financial_metrics)
    
    # -----------------------------
    # SAVE ARTIFACT
    # -----------------------------
    artifact = {
        "model": model,
        "features": feature_list,
        "metrics": metrics,
        "optimal_threshold": optimal_threshold,
        "trained_at": datetime.now().isoformat(),
        "symbol": symbol,
        "mode": mode,
        "split_info": {
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "train_start": str(X_train.index[0]),
            "train_end": str(X_train.index[-1]),
            "test_start": str(X_test.index[0]),
            "test_end": str(X_test.index[-1]),
        }
    }
    
    return artifact


# Update the old train_model to use the new function
def train_model(df: pd.DataFrame, symbol: str, mode: str = "daily"):
    """Wrapper for backward compatibility."""
    return train_model_with_validation(df, symbol, mode)


# ---------------------------------------------------------
# LOAD MODEL ARTIFACT
# ---------------------------------------------------------
def load_model(symbol: str, mode: str):
    """
    Load saved model artifact:
       {"model", "features", "metrics", "trained_at", "symbol", "mode"}
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


# ---------------------------------------------------------
# PREDICT FROM MODEL ARTIFACT
# ---------------------------------------------------------
def predict_from_model(model_dict, df_features: pd.DataFrame):
    if model_dict is None or df_features is None or df_features.empty:
        return None
    
    try:
        model = model_dict["model"]
        feat_cols = model_dict["features"]
        
        # ✅ FIX: Reindex + FILL (never drop!)
        df_aligned = df_features.reindex(columns=feat_cols, fill_value=0.0)
        df_clean = df_aligned.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        print(f"[PREDICT] ({len(df_features)}, {len(df_features.columns)}) → aligned ({len(df_aligned)}, {len(feat_cols)}) → clean ({len(df_clean)}, {len(feat_cols)})")
        
        X = df_clean.tail(1).values
        prob = float(model.predict_proba(X)[0, 1])
        return prob
        
    except Exception as e:
        print(f"[ERROR] predict_from_model: {e}")
        return None
    
# =====================================================================
# COMPLEX SIGNAL CALCULATOR (DAILY + INTRADAY) WITH ADAPTIVE LOOKBACK
# =====================================================================
def compute_signals(
    symbol,
    lookback_minutes=60,
    intraday_weight=INTRADAY_WEIGHT,
    resample_to="15min",
):
    results = {
        "daily_prob": None,
        "intraday_prob": None,
        "final_prob": None,
        "price": None,  # ✅ NEW: always try to populate
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
    }

    symU = str(symbol).upper().strip()

    # -------------------------
    # Load Models
    # -------------------------
    model_daily = load_model(symU, mode="daily")
    model_intra_mr = load_model(symU, mode="intraday_mr")
    model_intra_mom = load_model(symU, mode="intraday_mom")
    model_intraday_legacy = load_model(symU, mode="intraday")

    # -------------------------
    # DAILY SIGNAL + DAILY PRICE FALLBACK
    # -------------------------
    df_daily = None
    try:
        df_daily = fetch_historical_data(symU, period="6mo", interval="1d")
        if df_daily is not None and not df_daily.empty:
            results["daily_rows"] = len(df_daily)
            df_feat = build_daily_features(df_daily)
            results["daily_prob"] = predict_from_model(model_daily, df_feat)

            # ✅ price fallback from daily close (only if price not set later by intraday)
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
    # INTRADAY SIGNAL (ADAPTIVE LOOKBACK)
    # -------------------------
    df_intra_resampled = None
    df_feat_intra = None
    vol = None
    mom_1h = None
    is_momentum_regime = False

    try:
        candidate_lookbacks = sorted(
            {
                lookback_minutes,
                max(lookback_minutes, 600),
                max(lookback_minutes, 900),
                max(lookback_minutes, 1200),
            }
        )

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

            if rt in ("15m", "15min", "15t"):
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

        # Not enough bars -> disable intraday cleanly
        if df_intra_resampled is None or len(df_intra_resampled) < MIN_INTRADAY_BARS_FOR_FEATURES:
            results["allow_intraday"] = False
            results["intraday_prob"] = None
            results["intraday_quality_score"] = 0.0
        else:
            df_feat_intra = build_intraday_features(df_intra_resampled)

            # ADD this volume computation:
            if "Volume" in df_intra_resampled.columns:
                vol_series = df_intra_resampled["Volume"].dropna()
                if len(vol_series) >= 20:
                    current_vol = float(vol_series.iloc[-1])
                    avg_vol_20 = float(vol_series.tail(20).mean())
                    vol_ratio = current_vol / avg_vol_20 if avg_vol_20 > 0 else 0.0
                else:
                    vol_ratio = None
            else:
                vol_ratio = None

            results["intraday_volume"] = current_vol if 'current_vol' in locals() else None

            try:
                vol_series = df_intra_resampled["Volume"]
                vol_series = vol_series.iloc[:, 0] if isinstance(vol_series, pd.DataFrame) else vol_series
                vol_series = pd.to_numeric(vol_series, errors='coerce').dropna()
                
                if len(vol_series) >= 20:
                    current_vol = float(vol_series.iloc[-1])
                    avg_vol_20 = float(vol_series.tail(20).mean())
                    vol_ratio = current_vol / avg_vol_20 if avg_vol_20 > 0 else 0.0
                else:
                    current_vol = None
                    vol_ratio = None
                
                results["intraday_volume"] = current_vol
                results["intraday_volume_ratio"] = vol_ratio
            except Exception:
                results["intraday_volume"] = None
                results["intraday_volume_ratio"] = None
            
            if df_feat_intra is None or df_feat_intra.empty:
                results["allow_intraday"] = False
                results["intraday_prob"] = None
                results["intraday_quality_score"] = 0.0
            else:
                # -------------------------
                # Regime detection + model selection
                # -------------------------
                close = df_intra_resampled["Close"]
                close = close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close
                close = close.dropna()

                # ✅ BEST price source: last intraday close
                try:
                    if len(close) > 0:
                        results["price"] = float(close.iloc[-1])
                except Exception:
                    pass

                rets = close.pct_change().dropna()
                vol = float(rets.std()) if len(rets) > 3 else 0.0

                mom_1h = float((close.iloc[-1] / close.iloc[-5]) - 1.0) if len(close) >= 5 else 0.0

                results["intraday_vol"] = vol
                results["intraday_mom"] = mom_1h

            try:
                # Extract Volume column from resampled intraday data
                if "Volume" in df_intra_resampled.columns:
                    vol_series = df_intra_resampled["Volume"]
                    
                    # Handle both DataFrame and Series formats
                    if isinstance(vol_series, pd.DataFrame):
                        vol_series = vol_series.iloc[:, 0]
                    
                    # Convert to numeric and drop NaN
                    vol_series = pd.to_numeric(vol_series, errors='coerce').dropna()
                    
                    if len(vol_series) >= 20:
                        current_vol = float(vol_series.iloc[-1])
                        avg_vol_20 = float(vol_series.tail(20).mean())
                        vol_ratio = current_vol / avg_vol_20 if avg_vol_20 > 0 else None
                    else:
                        current_vol = None
                        vol_ratio = None
                    
                    results["intraday_volume"] = current_vol
                    results["intraday_volume_ratio"] = vol_ratio
                else:
                    # No Volume column available
                    results["intraday_volume"] = None
                    results["intraday_volume_ratio"] = None
            except Exception as e:
                # If any error, set to None
                results["intraday_volume"] = None
                results["intraday_volume_ratio"] = None
                    
                # thresholds (config-driven)
                MOM_TRIG = float(INTRADAY_MOM_TRIG)
                VOL_TRIG = float(INTRADAY_VOL_TRIG)

                print(f"[REGIME] {symU} mom_1h={mom_1h:.4f} vol={vol:.5f} MOM_TRIG={MOM_TRIG:.4f} VOL_TRIG={VOL_TRIG:.4f}")

                ismomentumregime = abs(mom_1h) >= MOM_TRIG or vol >= VOL_TRIG
                results["intraday_regime"] = "mom" if ismomentumregime else "mr"

                # ✅ PRIORITY: mom > mr > legacy
                model_used = None
                ip = None
                if model_intra_mom is not None and ismomentumregime:
                    ip = predict_from_model(model_intra_mom, df_feat_intra)
                    model_used = "intraday_mom"
                elif model_intra_mr is not None:
                    ip = predict_from_model(model_intra_mr, df_feat_intra)
                    model_used = "intraday_mr"
                elif model_intraday_legacy is not None:
                    ip = predict_from_model(model_intraday_legacy, df_feat_intra)
                    model_used = "intraday"

                if ip is not None:
                    results["intraday_prob"] = float(ip)
                    results["intraday_model_used"] = model_used
                    results["intraday_quality_score"] = min(1.0, len(df_intra_resampled) / 120.0)
                    print(f"[SUCCESS] {symU} {model_used} ip={ip:.3f}")
                else:
                    print(f"[FAIL] {symU} all models None/missing")


                ovr = (INTRADAY_REGIME_OVERRIDES or {}).get(symU)
                if ovr:
                    MOM_TRIG = float(ovr.get("mom_trig", MOM_TRIG))
                    VOL_TRIG = float(ovr.get("vol_trig", VOL_TRIG))

                is_momentum_regime = (abs(mom_1h) >= MOM_TRIG) or (vol >= VOL_TRIG)
                results["intraday_regime"] = "mom" if is_momentum_regime else "mr"

                try:
                    print(
                        f"[DEBUG] {symU} regime={'MOM' if is_momentum_regime else 'MR'} "
                        f"(mom={mom_1h:+.4%} vol={vol:.5f} | "
                        f"MOM_TRIG={MOM_TRIG:.4f} VOL_TRIG={VOL_TRIG:.4f})"
                    )
                except Exception:
                    pass

                def _direction_conflict(mom_val: float, ip_val: float) -> bool:
                    if mom_val >= 0.0 and ip_val <= 0.35:
                        return True
                    if mom_val <= 0.0 and ip_val >= 0.65:
                        return True
                    return False

                model_used = None

                # MOM regime
                if is_momentum_regime and model_intra_mom is not None:
                    ip_tmp = predict_from_model(model_intra_mom, df_feat_intra)
                    ip_tmp_f = float(ip_tmp) if ip_tmp is not None else None

                    if ip_tmp_f is not None and _direction_conflict(mom_1h, ip_tmp_f):
                        if model_intra_mr is not None:
                            model_used = "intraday_mr"
                            results["intraday_prob"] = predict_from_model(model_intra_mr, df_feat_intra)
                        else:
                            model_used = "intraday"
                            results["intraday_prob"] = predict_from_model(model_intraday_legacy, df_feat_intra)
                    else:
                        model_used = "intraday_mom"
                        results["intraday_prob"] = ip_tmp_f

                # MR regime
                elif (not is_momentum_regime) and model_intra_mr is not None:
                    model_used = "intraday_mr"
                    results["intraday_prob"] = predict_from_model(model_intra_mr, df_feat_intra)

                # fallback
                else:
                    model_used = "intraday"
                    results["intraday_prob"] = predict_from_model(model_intraday_legacy, df_feat_intra)

                results["intraday_model_used"] = model_used
                ip_tmp_f = None

                if results["intraday_rows"] >= MIN_INTRADAY_BARS_FOR_FEATURES:
                    try:
                        
                        if model_used == "intraday_mom" and model_intra_mom is not None:
                            ip_tmp_f = model_intra_mom.predict_proba(df_feat_intra)[:, 1]
                        elif model_used == "intraday_mr" and model_intra_mr is not None:
                            ip_tmp_f = model_intra_mr.predict_proba(df_feat_intra)[:, 1]
                        elif model_intraday_legacy is not None:
                            ip_tmp_f = model_intraday_legacy.predict_proba(df_feat_intra)[:, 1]
                        
                        if ip_tmp_f is not None and len(ip_tmp_f) > 0:
                            results["intraday_prob"] = float(np.mean(ip_tmp_f[-5:]))  # last 5 predictions
                            results["intraday_model_used"] = model_used
                            results["intraday_quality_score"] = min(1.0, len(df_feat_intra) / 120.0)
                            print(f"[INTRADAY_SUCCESS] {symU} ip={results['intraday_prob']:.3f} q={results['intraday_quality_score']:.2f}")
                        else:
                            raise ValueError("No predictions generated")
                            
                    except Exception as e:
                        print(f"[INTRADAY_ERROR] {symU} {model_used}: {e}")
                        print(f"[INTRADAY_ERROR] feat_shape={df_feat_intra.shape if 'df_feat_intra' in locals() else 'N/A'}")
                        results["intraday_prob"] = None
                        results["intraday_model_used"] = None
                        results["intraday_quality_score"] = 0.0

                # ✅ quality score set whenever intraday is usable
                results["intraday_quality_score"] = float(
                    max(0.0, min(1.0, len(df_intra_resampled) / 120.0))
                )

    except Exception as e:
        print(f"[ERROR] Intraday prediction error: {e}")
        results["allow_intraday"] = False
        results["intraday_prob"] = None
        results["intraday_quality_score"] = 0.0

    # -------------------------
    # FINAL PRICE LAST RESORT
    # -------------------------
    # If intraday & daily both failed to set price, last resort use fetch_latest_price()
    if results.get("price") is None:
        try:
            p = fetch_latest_price(symU)
            if p is not None and float(p) > 0:
                results["price"] = float(p)
        except Exception:
            pass

    # Extract before combine
    dp = results["daily_prob"]
    ip = results["intraday_prob"]

    # ===============================
    # ADAPTIVE INTRADAY WEIGHT
    # ===============================
    weight = float(intraday_weight)

    if (not results["allow_intraday"]) or (ip is None):
        weight = 0.0
    else:
        q = float(results.get("intraday_quality_score", 1.0) or 0.0)
        weight = float(intraday_weight) * q

        # Ensure vol/mom exist
        if vol is None:
            vol = float(results.get("intraday_vol") or 0.0)
        if mom_1h is None:
            mom_1h = float(results.get("intraday_mom") or 0.0)

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
            if (dp is not None) and (dp >= 0.65) and (mom_1h >= 0.0020):
                weight = max(weight, 0.65)
        except Exception:
            pass

        # Strong intraday momentum override
        try:
            if (
                (ip is not None)
                and (dp is not None)
                and (results.get("intraday_quality_score", 0.0) >= 0.6)
                and (ip >= 0.70)
                and ((ip - dp) >= 0.15)
            ):
                weight = max(weight, 0.75)
        except Exception:
            pass

        # Strong daily continuation override
        try:
            if (
                (dp is not None)
                and (ip is not None)
                and (dp >= 0.78)
                and (ip >= 0.25)
                and (mom_1h <= -0.005)
            ):
                weight = max(weight, 0.70)
        except Exception:
            pass

    weight = float(min(max(weight, 0.0), 0.90))
    results["intraday_weight"] = weight

    # ===============================
    # FINAL PROBABILITY
    # ===============================
    if dp is None and ip is None:
        results["final_prob"] = None
    elif dp is None:
        results["final_prob"] = float(ip)
    elif ip is None or weight == 0.0:
        results["final_prob"] = float(dp)
    else:
        results["final_prob"] = float(weight * ip + (1 - weight) * dp)

    # -------------------------
    # Safe debug prints
    # -------------------------
    def _fmt(x, n=3):
        return "NA" if x is None else f"{float(x):.{n}f}"

    def _fmt_pct(x):
        return "NA" if x is None else f"{float(x)*100:.2f}%"

    print(
        f"[DEBUG] {symU} dp={_fmt(dp)} ip={_fmt(ip)} q={results.get('intraday_quality_score',0):.2f} "
        f"weight={weight:.2f} model={results.get('intraday_model_used')} price={_fmt(results.get('price'), 2)}"
    )
    print(
        f"[DEBUG] {symU} vol={_fmt(results.get('intraday_vol'), 5)} mom={_fmt_pct(results.get('intraday_mom'))}"
    )

    return results

# =====================================================================
# SIMPLE NEXT-STEP PREDICTOR (WRAPPER)
# =====================================================================
def predict_next(symbol, lookback_minutes=60, intraday_weight=INTRADAY_WEIGHT, resample_to="15min"):
    """
    Compatibility wrapper — returns only the final probability.
    """
    sig = compute_signals(symbol, lookback_minutes, intraday_weight, resample_to)
    return sig.get("final_prob")


# =====================================================================
# SIMPLE NEXT-STEP PREDICTOR (WRAPPER)
# =====================================================================
def predict_next(
    symbol: str,
    lookback_minutes: int = 60,
    intraday_weight: float = INTRADAY_WEIGHT,
    resample_to: str = "15min",
):
    """
    Compatibility wrapper — returns only the final probability.
    """
    sig = compute_signals(symbol, lookback_minutes, intraday_weight, resample_to)
    return sig.get("final_prob")