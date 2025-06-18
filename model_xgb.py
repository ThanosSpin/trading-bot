# model_xgb.py
from xgboost import XGBClassifier
import pandas as pd
import pickle
import os
from config import MODEL_PATH

def train_model(df):
    df = df.copy()
    df['Target'] = (df['Return'].shift(-1) > 0).astype(int)
    df = df.dropna()

    features = ['Return', 'MA_5', 'MA_20', 'Momentum_10', 'Volatility_10', 'Volume_Change']
    X = df[features]
    y = df['Target']

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    print("✅ Model trained with technical indicators.")
    return model

def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

def predict_next(df, model):
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