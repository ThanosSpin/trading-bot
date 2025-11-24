# model_xgb.py
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from features import build_daily_features, build_intraday_features
from data_loader import fetch_historical_data, fetch_intraday_history
from config import INTRADAY_WEIGHT

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ---------------------------------------------------------
# TRAIN MODEL  (returns artifact dict)
# ---------------------------------------------------------
def train_model(df: pd.DataFrame, symbol: str, mode: str = "daily"):
    """
    Train XGB model for a given symbol.

    Returns an artifact dict:
        {
          "model": XGBClassifier,
          "features": [list of feature names],
          "metrics": {...},
          "trained_at": ISO timestamp,
          "symbol": str,
          "mode": "daily" | "intraday",
        }
    """

    from sklearn.metrics import (
        accuracy_score,
        log_loss,
        roc_auc_score,
        confusion_matrix,
        precision_score,
        recall_score,
        f1_score,
    )

    df = df.copy()

    # -----------------------------
    # FEATURE GENERATION
    # -----------------------------
    if mode == "daily":
        df_feat = build_daily_features(df)
    elif mode == "intraday":
        df_feat = build_intraday_features(df)
    else:
        raise ValueError("mode must be 'daily' or 'intraday'")

    # Label: next candle up/down
    df_feat["target"] = (df_feat["Close"].shift(-1) > df_feat["Close"]).astype(int)
    df_feat.dropna(inplace=True)

    if df_feat.empty:
        raise ValueError(f"No rows left after feature engineering for {symbol} ({mode})")

    X = df_feat.drop(columns=["target"])
    y = df_feat["target"]
    feature_list = list(X.columns)

    # -----------------------------
    # HYPERPARAMETERS
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
    # TIME-SERIES TRAIN/TEST SPLIT (80/20, no shuffle)
    # -----------------------------
    split_idx = int(len(X) * 0.8)
    if split_idx <= 0 or split_idx >= len(X):
        raise ValueError(f"Not enough data to split train/test for {symbol} ({mode})")

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # -----------------------------
    # TRAIN
    # -----------------------------
    model.fit(X_train, y_train)

    # -----------------------------
    # VALIDATION METRICS
    # -----------------------------
    metrics = {}
    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
        metrics["logloss"] = float(log_loss(y_test, y_pred_proba))
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_pred_proba))
        metrics["precision"] = float(precision_score(y_test, y_pred))
        metrics["recall"] = float(recall_score(y_test, y_pred))
        metrics["f1"] = float(f1_score(y_test, y_pred))
        metrics["confusion_matrix"] = (
            confusion_matrix(y_test, y_pred).tolist()
        )
    except Exception as e:
        print(f"[WARN] Could not compute some metrics for {symbol} ({mode}): {e}")

    # -----------------------------
    # LOG RESULTS
    # -----------------------------
    print(f"\n📊 MODEL VALIDATION — {symbol} [{mode.upper()}]")
    print("--------------------------------------------")
    for k, v in metrics.items():
        if k == "confusion_matrix":
            print(f"{k:20}: {v}")
        else:
            print(f"{k:20}: {v:.4f}")
    print("--------------------------------------------\n")

    artifact = {
        "model": model,
        "features": feature_list,
        "metrics": metrics,
        "trained_at": datetime.now().isoformat(),
        "symbol": symbol,
        "mode": mode,
    }

    return artifact


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
    """Return probability using stored feature order (or None)."""
    if model_dict is None:
        return None

    try:
        model = model_dict["model"]
        feat_cols = model_dict["features"]

        missing = [c for c in feat_cols if c not in df_features.columns]
        if missing:
            # Feature schema mismatch — safer to skip
            print(f"[WARN] Missing features for prediction: {missing}")
            return None

        X = df_features[feat_cols].tail(1).values
        prob = float(model.predict_proba(X)[0][1])
        return prob

    except Exception as e:
        print(f"[ERROR] predict_from_model: {e}")
        return None


# =====================================================================
# COMPLEX SIGNAL CALCULATOR (DAILY + INTRADAY) WITH ADAPTIVE WEIGHTING
# =====================================================================
def compute_signals(
    symbol: str,
    lookback_minutes: int = 60,
    intraday_weight: float = INTRADAY_WEIGHT,
    resample_to: str = "5min",
):
    """
    Combined signal generator with adaptive intraday weighting.

    Returns dict:
        {
          "daily_prob": float or None,
          "intraday_prob": float or None,
          "final_prob": float or None,
          "intraday_weight": float,
          "allow_intraday": bool,
          "intraday_rows": int,
          "intraday_before": int,
          "daily_rows": int
        }
    """

    results = {
        "daily_prob": None,
        "intraday_prob": None,
        "final_prob": None,
        "intraday_weight": intraday_weight,
        "allow_intraday": True,
        "intraday_rows": 0,
        "intraday_before": 0,
        "daily_rows": 0,
    }

    # -------------------------
    # LOAD MODELS
    # -------------------------
    model_daily = load_model(symbol, mode="daily")
    model_intraday = load_model(symbol, mode="intraday")

    # -------------------------
    # DAILY SIGNAL
    # -------------------------
    try:
        df_daily = fetch_historical_data(symbol, period="6mo", interval="1d")
        if df_daily is not None and not df_daily.empty:
            results["daily_rows"] = len(df_daily)
            df_feat = build_daily_features(df_daily)
            results["daily_prob"] = predict_from_model(model_daily, df_feat)
    except Exception as e:
        print(f"[ERROR] Daily prediction error for {symbol}: {e}")

    # -------------------------
    # INTRADAY SIGNAL
    # -------------------------
    df_intra = None
    try:
        df_intra = fetch_intraday_history(symbol, lookback_minutes=lookback_minutes)
        if df_intra is not None and not df_intra.empty:
            results["intraday_before"] = len(df_intra)

            # Resample smoothing
            df_intra = df_intra.resample(resample_to).last().dropna()
            results["intraday_rows"] = len(df_intra)

            if len(df_intra) >= 20:
                df_feat_intra = build_intraday_features(df_intra)
                results["intraday_prob"] = predict_from_model(model_intraday, df_feat_intra)
            else:
                print(f"[INFO] Intraday dataset too small after resample for {symbol} ({len(df_intra)} rows).")
                results["allow_intraday"] = False
        else:
            results["allow_intraday"] = False

    except Exception as e:
        print(f"[ERROR] Intraday prediction error for {symbol}: {e}")
        results["allow_intraday"] = False

    dp = results["daily_prob"]
    ip = results["intraday_prob"]

    # -------------------------
    # ADAPTIVE INTRADAY WEIGHT
    # -------------------------
    weight = intraday_weight

    if not results["allow_intraday"] or ip is None or df_intra is None or df_intra.empty:
        weight = 0.0
    else:
        try:
            close = df_intra["Close"]
            returns = close.pct_change().dropna()
            vol = returns.std()

            # Simple volatility-based scaling
            if vol > 0.025:       # high vol → trust intraday more
                weight = min(0.90, intraday_weight + 0.20)
            elif vol < 0.008:     # calm → reduce intraday influence
                weight = max(0.30, intraday_weight - 0.20)
            else:                 # medium → keep base weight
                weight = intraday_weight
        except Exception:
            # On any failure, just keep original weight
            weight = intraday_weight

    results["intraday_weight"] = weight

    # -----------------------------
    # FINAL COMBINED PROB
    # -----------------------------
    if dp is None and ip is None:
        results["final_prob"] = None
    elif dp is None:
        results["final_prob"] = ip
    elif ip is None:
        results["final_prob"] = dp
    else:
        results["final_prob"] = float(weight * ip + (1 - weight) * dp)

    return results


# =====================================================================
# SIMPLE NEXT-STEP PREDICTOR (WRAPPER)
# =====================================================================
def predict_next(
    symbol: str,
    lookback_minutes: int = 60,
    intraday_weight: float = INTRADAY_WEIGHT,
    resample_to: str = "5min",
):
    """
    Compatibility wrapper — returns only the final probability.
    """
    sig = compute_signals(symbol, lookback_minutes, intraday_weight, resample_to)
    return sig.get("final_prob")