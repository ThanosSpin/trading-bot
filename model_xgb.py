# model_xgb.py
import os
import joblib
import numpy as np
from xgboost import XGBClassifier
import pandas as pd

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
# Train Model
# ---------------------------------------------------------
def train_model(df, symbol, mode="daily"):
    """
    Train XGB model for a given symbol.
    mode: "daily" or "intraday"
    """
    df = df.copy()

    # Build features
    if mode == "daily":
        df_feat = build_daily_features(df)
    elif mode == "intraday":
        df_feat = build_intraday_features(df)
    else:
        raise ValueError("mode must be 'daily' or 'intraday'")

    # Target: next candle up/down
    df_feat["target"] = (df_feat["Close"].shift(-1) > df_feat["Close"]).astype(int)
    df_feat.dropna(inplace=True)

    X = df_feat.drop(columns=["target"])
    y = df_feat["target"]

    # Model hyperparameters
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
    model.fit(X, y)

    save_path = os.path.join(MODEL_DIR, f"{symbol}_{mode}_xgb.pkl")
    joblib.dump({"model": model, "features": list(X.columns)}, save_path)
    print(f"✅ Model trained and saved: {save_path}")

    return model


# ---------------------------------------------------------
# Load
# ---------------------------------------------------------
def load_model(symbol, mode):
    """Load (model, feature list) or None."""
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
# Predict from model
# ---------------------------------------------------------
def predict_from_model(model_dict, df_features):
    """Return probability using stored feature order."""
    if model_dict is None:
        return None

    try:
        model = model_dict["model"]
        feat_cols = model_dict["features"]

        # Align feature order
        missing = [c for c in feat_cols if c not in df_features.columns]
        if missing:
            return None

        X = df_features[feat_cols].tail(1).values
        prob = float(model.predict_proba(X)[0][1])
        return prob

    except Exception as e:
        print(f"[ERROR] predict_from_model: {e}")
        return None


# =====================================================================
# COMPLEX SIGNAL CALCULATOR (DAILY + INTRADAY)
# =====================================================================
def compute_signals(
    symbol,
    lookback_minutes=60,
    intraday_weight=INTRADAY_WEIGHT,
    resample_to="5min",
):
    """
    Combined signal generator with adaptive intraday weighting.

    Returns:
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
    # Load Models
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
        print(f"[ERROR] Daily prediction error: {e}")

    # -------------------------
    # INTRADAY SIGNAL
    # -------------------------
    try:
        df_intra = fetch_intraday_history(symbol, lookback_minutes=lookback_minutes)
        if df_intra is not None and not df_intra.empty:

            results["intraday_before"] = len(df_intra)

            # Resample smoothing
            df_intra = df_intra.resample(resample_to).last().dropna()
            results["intraday_rows"] = len(df_intra)

            if len(df_intra) >= 20:
                df_feat = build_intraday_features(df_intra)
                results["intraday_prob"] = predict_from_model(model_intraday, df_feat)
            else:
                print(f"[INFO] Intraday dataset too small after resample ({len(df_intra)} rows).")
                results["allow_intraday"] = False

    except Exception as e:
        print(f"[ERROR] Intraday prediction error: {e}")
        results["allow_intraday"] = False

    # Extract before combine
    dp = results["daily_prob"]
    ip = results["intraday_prob"]

    # ===============================
    # ADAPTIVE INTRADAY WEIGHT
    # ===============================
    weight = intraday_weight

    if not results["allow_intraday"] or ip is None:
        weight = 0.0  # cannot use intraday
    else:
        try:
            # Estimate volatility from intraday prices
            close = df_intra["Close"]
            returns = close.pct_change().dropna()

            vol = returns.std()

            # adaptive scaling thresholds
            if vol > 0.025:       # High volatility → trust intraday more
                weight = min(0.90, intraday_weight + 0.20)
            elif vol < 0.008:     # Very calm → reduce intraday influence
                weight = max(0.30, intraday_weight - 0.20)
            else:                 # Medium volatility → slight adjustment
                weight = intraday_weight

        except Exception:
            pass

    results["intraday_weight"] = weight

    # ===============================
    # FINAL PROBABILITY
    # ===============================
    if dp is None and ip is None:
        results["final_prob"] = None
    elif dp is None:
        results["final_prob"] = ip
    elif ip is None:
        results["final_prob"] = dp
    else:
        results["final_prob"] = float(weight * ip + (1 - weight) * dp)

    return results

    # -----------------------------
    # COMBINE DAILY + INTRADAY
    # -----------------------------
    dp = results["daily_prob"]
    ip = results["intraday_prob"]

    if dp is None and ip is None:
        results["final_prob"] = None
    elif dp is None:
        results["final_prob"] = ip
    elif ip is None:
        results["final_prob"] = dp
    else:
        results["final_prob"] = float(intraday_weight * ip + (1 - intraday_weight) * dp)

    if debug:
        print(f"[DEBUG] {symbol} final_prob={results['final_prob']}")

    return results

# =====================================================================
# SIMPLE NEXT-STEP PREDICTOR (WRAPPER)
# =====================================================================
def predict_next(symbol, lookback_minutes=60, intraday_weight=INTRADAY_WEIGHT, resample_to="5min"):
    """
    Compatibility wrapper — returns only the final probability.
    """
    sig = compute_signals(symbol, lookback_minutes, intraday_weight, resample_to)
    return sig.get("final_prob")