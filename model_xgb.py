# model_xgb.py
from xgboost import XGBClassifier
import pandas as pd
import pickle
import os
from config_multi import get_model_path  # ✅ per-symbol model paths


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to dataframe and drop NaNs.
    """
    df = df.copy()
    df['Return'] = df['Close'].pct_change()
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
    df['Volatility_10'] = df['Return'].rolling(window=10).std()
    df['Volume_Change'] = df['Volume'].pct_change()
    return df.dropna()


def train_model(df: pd.DataFrame, symbol: str):
    """
    Train and save XGBoost model for a given symbol.
    """
    df = prepare_features(df)
    df['Target'] = (df['Return'].shift(-1) > 0).astype(int)
    df = df.dropna()

    features = ['Return', 'MA_5', 'MA_20',
                'Momentum_10', 'Volatility_10', 'Volume_Change']
    X = df[features]
    y = df['Target']

    model = XGBClassifier(eval_metric='logloss')  # use_label_encoder deprecated
    model.fit(X, y)

    model_path = get_model_path(symbol)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"✅ Model trained and saved for {symbol} → {model_path}")
    return model


def load_model(symbol: str):
    """
    Load the XGBoost model for a specific symbol.
    """
    model_path = get_model_path(symbol)
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found for {symbol} at {model_path}")
        return None

    with open(model_path, 'rb') as f:
        return pickle.load(f)


def predict_next(df, model):
    if model is None:
        print("[ERROR] No model loaded. Did you run retrain_model.py?")
        return None

    if df.empty or 'Close' not in df.columns or 'Volume' not in df.columns:
        print("[WARN] DataFrame is empty or missing required columns.")
        return None

    df = df.copy()
    df['Return'] = df['Close'].pct_change()
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
    df['Volatility_10'] = df['Return'].rolling(window=10).std()
    df['Volume_Change'] = df['Volume'].pct_change()
    df = df.dropna()

    features = ['Return', 'MA_5', 'MA_20', 'Momentum_10', 'Volatility_10', 'Volume_Change']
    if df.empty or not all(col in df.columns for col in features):
        print("[WARN] Not enough data to compute features for prediction.")
        return None

    latest = df[features].iloc[-1:]
    if latest.empty:
        print("[WARN] No valid row to predict.")
        return None

    return model.predict_proba(latest)[0][1]