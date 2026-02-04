# model_xgb.py
import os
from datetime import datetime
import warnings

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from features import build_daily_features, build_intraday_features
from data_loader import fetch_historical_data, fetch_intraday_history, fetch_latest_price
from target_labels import create_target_label, backtest_threshold
from trading_metrics import calculate_financial_metrics, print_trading_report
from adaptive_thresholds import get_adaptive_regime_thresholds
from model_monitor import log_prediction, evaluate_predictions



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


def train_model(df: pd.DataFrame, symbol: str, mode: str = "daily"):
    """
    Train XGB model for a given symbol.
    Supported modes:
    - "daily"
    - "intraday" (legacy unfiltered)
    - "intraday_mr" (mean reversion regime)
    - "intraday_mom" (momentum regime)
    """
    from sklearn.metrics import (
        accuracy_score, log_loss, roc_auc_score,
        confusion_matrix, precision_score, recall_score, f1_score,
        brier_score_loss,
    )
    from sklearn.calibration import CalibratedClassifierCV

    df = df.copy()

    # -----------------------------
    # FEATURE GENERATION
    # -----------------------------
    if mode == "daily":
        df_feat = build_daily_features(df)
    elif mode in ("intraday", "intraday_mr", "intraday_mom"):
        df_feat = build_intraday_features(df)

        # regime filtering only for the dual intraday modes
        if mode in ("intraday_mr", "intraday_mom"):
            df_feat = _filter_intraday_rows_by_mode(df_feat, mode=mode)

            # drop regime helper cols so model doesn't learn them directly
            for c in ["ret_12", "mom_12_abs", "vol_12"]:
                if c in df_feat.columns:
                    df_feat.drop(columns=c, inplace=True)
    else:
        raise ValueError("mode must be 'daily' or one of 'intraday', 'intraday_mr', 'intraday_mom'")

    # -----------------------------
    # TARGET
    # -----------------------------
    df_feat["target"] = (df_feat["Close"].shift(-1) > df_feat["Close"]).astype(int)
    df_feat.dropna(inplace=True)

    if df_feat.empty:
        raise ValueError(f"No rows left after feature engineering for {symbol}/{mode}")

    X = df_feat.drop(columns="target")
    y = df_feat["target"]
    feature_list = list(X.columns)

    # -----------------------------
    # XGB PARAMS
    # -----------------------------
    params = {
        "eval_metric": "logloss",
        "tree_method": "hist",
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "max_depth": 6,
    }
    if mode == "daily":
        params.update({"n_estimators": 400, "learning_rate": 0.04})
    else:
        params.update({"n_estimators": 500, "learning_rate": 0.035})

    model = XGBClassifier(**params)

    # -----------------------------
    # TRAIN/TEST SPLIT
    # -----------------------------
    split_idx = int(len(X) * 0.8)
    if split_idx <= 0 or split_idx >= len(X):
        raise ValueError(f"Not enough data to split train/test for {symbol}/{mode}")

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # ========================================
    # ✅ 3. CLASS WEIGHTING (BEFORE FIT)
    # ========================================
    # ✅ UPDATED: Apply to ALL modes (not just intraday)
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())

    if n_pos > 0 and n_neg > 0:
        scale_pos_weight = n_neg / n_pos
        
        # Only apply if there's significant imbalance (>1.2x difference)
        if scale_pos_weight > 1.2 or scale_pos_weight < 0.83:
            model.set_params(scale_pos_weight=scale_pos_weight)
            print(f"[CLASS WEIGHT] {symbol}/{mode}: neg={n_neg} pos={n_pos} scale={scale_pos_weight:.2f}")
        else:
            print(f"[CLASS WEIGHT] {symbol}/{mode}: Balanced classes, no weighting needed")
    else:
        print(f"[CLASS WEIGHT] ⚠️ {symbol}/{mode}: Insufficient class samples")


    # -----------------------------
    # FIT
    # -----------------------------
    model.fit(X_train, y_train)

    # ✅ STORE BASE MODEL FOR SHAP (before calibration)
    base_model = model

    # ========================================
    # ✅ 1. CALIBRATION (AFTER FIT)
    # ========================================
    final_model = model  # Default: use uncalibrated model

    # ✅ UPDATED: Apply calibration to ALL modes
    print(f"[CALIBRATION] Applying Platt scaling for {symbol}/{mode}")
    try:
        calibrated_model = CalibratedClassifierCV(
            model, 
            method='sigmoid',
            cv='prefit',
            n_jobs=-1
        )
        
        # Fit calibration on test set
        calibrated_model.fit(X_test, y_test)
        final_model = calibrated_model
        print(f"[CALIBRATION] ✅ Successfully calibrated {symbol}/{mode}")
        
    except Exception as e:
        print(f"[CALIBRATION] ⚠️ Failed to calibrate {symbol}/{mode}: {e}")
        print(f"[CALIBRATION] Using uncalibrated model as fallback")
        final_model = model


    # -----------------------------
    # METRICS (use final_model)
    # -----------------------------
    metrics = {}
    try:
        y_pred_proba = final_model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
        metrics["logloss"] = float(log_loss(y_test, y_pred_proba))
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_pred_proba))
        metrics["precision"] = float(precision_score(y_test, y_pred, zero_division=0))
        metrics["recall"] = float(recall_score(y_test, y_pred, zero_division=0))
        metrics["f1"] = float(f1_score(y_test, y_pred, zero_division=0))
        metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
        
        # ✅ ADD: Calibration metrics
        metrics["brier_score"] = float(brier_score_loss(y_test, y_pred_proba))
        
        # Calibration error (expected vs actual)
        avg_pred = float(np.mean(y_pred_proba))
        actual_rate = float(np.mean(y_test))
        metrics["calibration_error"] = abs(avg_pred - actual_rate)
        
    except Exception as e:
        print(f"WARN: Could not compute some metrics for {symbol}/{mode}: {e}")

    # -----------------------------
    # PRINT VALIDATION SUMMARY
    # -----------------------------
    print(f"MODEL VALIDATION: {symbol}/{mode.upper()} (rows={len(df_feat)})")
    print("--------------------------------------------")
    for k, v in metrics.items():
        if k == "confusion_matrix":
            print(f"{k:20} {v}")
        else:
            try:
                print(f"{k:20} {v:.4f}")
            except Exception:
                print(f"{k:20} {v}")
    print("--------------------------------------------")

    artifact = {
        "model": final_model,  # ✅ Calibrated model for predictions
        "base_model": base_model,  # ✅ Uncalibrated XGBoost for SHAP
        "features": feature_list,
        "metrics": metrics,
        "trained_at": datetime.now().isoformat(),
        "symbol": symbol,
        "mode": mode,
        "calibrated": True,
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


# ---------------------------------------------------------
# PREDICT FROM MODEL ARTIFACT
# ---------------------------------------------------------
def predict_from_model(model_dict, df_features: pd.DataFrame):
    """Return probability using stored feature order (or None)."""
    if model_dict is None:
        return None

    if df_features is None or df_features.empty:
        print("[ERROR] predict_from_model: df_features empty.")
        return None

    try:
        # ✅ Use calibrated model for predictions (better probabilities)
        model = model_dict["model"]
        feat_cols = model_dict["features"]

        # Align features + fill missing (don't drop rows!)
        missing = [c for c in feat_cols if c not in df_features.columns]
        if missing:
            print(f"[WARN] Missing {len(missing)} features, filling with 0.0")

        # Reindex to match training feature order exactly
        df_aligned = df_features.reindex(columns=feat_cols, fill_value=0.0)

        # Replace inf/-inf with NaN, then fill with 0.0 (NO dropna!)
        df_clean = df_aligned.replace([np.inf, -np.inf], np.nan).fillna(0.0)



        if df_clean.empty:
            print("[ERROR] predict_from_model: df_clean empty after fillna")
            return None

        # Use last row for prediction
        X = df_clean.tail(1).values
        
        # ✅ Works with both calibrated and uncalibrated models
        prob = float(model.predict_proba(X)[0, 1])
        return prob

    except Exception as e:
        print(f"[ERROR] predict_from_model: {e}")
        return None


# ---------------------------------------------------------
# COMPUTE SIGNALS (MAIN ENTRY POINT)
# ---------------------------------------------------------
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
            results["daily_prob"] = predict_from_model(model_daily, df_feat)

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
                            price=float(current_price)
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
    vol = None
    mom1h = None

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
            df_feat_intra = build_intraday_features(df_intra_resampled)

            if df_feat_intra is None or df_feat_intra.empty:
                results["allow_intraday"] = False
                results["intraday_prob"] = None
                results["intraday_quality_score"] = 0.0
            else:
                # -------------------------
                # COMPUTE VOL/MOM
                # -------------------------
                close = df_intra_resampled["Close"]
                close = close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close
                close = close.dropna()

                # BEST price source = last intraday close
                try:
                    if len(close) > 0:
                        results["price"] = float(close.iloc[-1])
                except Exception:
                    pass

                rets = close.pct_change().dropna()
                vol = float(rets.std() if len(rets) >= 3 else 0.0)
                mom1h = float((close.iloc[-1] / close.iloc[-5] - 1.0) if len(close) >= 5 else 0.0)

                results["intraday_vol"] = vol
                results["intraday_mom"] = mom1h

                # ========================================
                # ✅ REGIME DETECTION + MODEL SELECTION
                # ========================================
                try:
                    adaptive = get_adaptive_regime_thresholds(symU, lookback_days=30, percentile=0.70)
                    MOMTRIG = float(adaptive['mom_trig'])
                    VOLTRIG = float(adaptive['vol_trig'])
                    print(f"[ADAPTIVE] {symU} using adaptive: mom={MOMTRIG:.4f} vol={VOLTRIG:.5f}")
                except Exception as e:
                    MOMTRIG = float(INTRADAY_MOM_TRIG)
                    VOLTRIG = float(INTRADAY_VOL_TRIG)
                    ovr = (INTRADAY_REGIME_OVERRIDES or {}).get(symU)
                    if ovr:
                        MOMTRIG = float(ovr.get("mom_trig", MOMTRIG))
                        VOLTRIG = float(ovr.get("vol_trig", VOLTRIG))
                    print(f"[CONFIG] {symU} using config: mom={MOMTRIG:.4f} vol={VOLTRIG:.5f}")

                ismomentumregime = abs(mom1h) >= MOMTRIG or vol >= VOLTRIG
                results["intraday_regime"] = "mom" if ismomentumregime else "mr"

                # ✅ PRIORITY: mom > mr > legacy (ALWAYS predict)
                model_used = None
                ip = None

                if model_intra_mom is not None and ismomentumregime:
                    print(f"[TRY] {symU} mom model_intra_mom")
                    ip = predict_from_model(model_intra_mom, df_feat_intra)
                    model_used = "intraday_mom"
                elif model_intra_mr is not None:
                    print(f"[TRY] {symU} mr model_intra_mr")
                    ip = predict_from_model(model_intra_mr, df_feat_intra)
                    model_used = "intraday_mr"
                elif model_intraday_legacy is not None:
                    print(f"[TRY] {symU} legacy model_intraday_legacy")
                    ip = predict_from_model(model_intraday_legacy, df_feat_intra)
                    model_used = "intraday"
                else:
                    print(f"[FAIL] {symU} NO MODELS LOADED AT ALL")

                if ip is not None:
                    results["intraday_prob"] = float(ip)
                    results["intraday_model_used"] = model_used
                    results["intraday_quality_score"] = min(1.0, len(df_intra_resampled) / 120.0)
                    print(f"[SUCCESS] {symU} {model_used} ip={ip:.3f}")
                
                    # ✅ LOG PREDICTION FOR MONITORING
                    try:
                        current_price = results.get("price") or fetch_latest_price(symU)
                        if current_price and float(current_price) > 0:
                            log_prediction(
                                symbol=symU,
                                mode=model_used,
                                predicted_prob=float(ip),
                                price=float(current_price)
                            )
                    except Exception as e:
                        print(f"[WARN] Could not log prediction: {e}")
                    
                else:
                    print(f"[PREDICT_FAIL] {symU} {model_used} returned None")

    except Exception as e:
        print(f"[ERROR] Intraday prediction error: {e}")
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
    # COMBINE DAILY + INTRADAY
    # -------------------------
    dp = results["daily_prob"]
    ip = results["intraday_prob"]

    # Extract before combine
    weight = float(intraday_weight)
    if not results["allow_intraday"] or ip is None:
        weight = 0.0
    else:
        q = float(results.get("intraday_quality_score", 1.0) or 0.0)
        weight = float(intraday_weight) * q

    # Ensure vol/mom exist
    if vol is None:
        vol = float(results.get("intraday_vol") or 0.0)
    if mom1h is None:
        mom1h = float(results.get("intraday_mom") or 0.0)

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
        if dp is not None and dp > 0.65 and mom1h > 0.0020:
            weight = max(weight, 0.65)
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
    # DEBUG PRINTS
    # -------------------------
    def fmt(x, n=3):
        return "NA" if x is None else f"{float(x):.{n}f}"

    def fmtpct(x):
        return "NA" if x is None else f"{float(x)*100:.2f}%"

    print(f"[DEBUG] {symU} dp={fmt(dp)} ip={fmt(ip)} q={results.get('intraday_quality_score',0):.2f} weight={weight:.2f} model={results.get('intraday_model_used')} price={fmt(results.get('price'), 2)}")
    print(f"[DEBUG] {symU} vol={fmt(results.get('intraday_vol'), 5)} mom={fmtpct(results.get('intraday_mom'))}")

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

    return results


# ---------------------------------------------------------
# COMPATIBILITY WRAPPER
# ---------------------------------------------------------
def predict_next(symbol, lookback_minutes=60, intraday_weight=INTRADAY_WEIGHT, resample_to="15min"):
    """Compatibility wrapper: returns only the final probability."""
    sig = compute_signals(symbol, lookback_minutes, intraday_weight, resample_to)
    return sig.get("final_prob")
