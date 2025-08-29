# model_rf.py
import pickle
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = "models/random_forest_model.pkl"

def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    else:
        print("[INFO] No existing RF model found. Training a new one...")
        return train_model()

def train_model():
    from data_loader import fetch_historical_data
    df = fetch_historical_data()
    
    df['Return'] = df['Close'].pct_change()
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
    df['Volatility_10'] = df['Return'].rolling(window=10).std()
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()

    features = ['Return', 'MA_5', 'MA_20', 'Momentum_10', 'Volatility_10', 'Volume_Change']
    X = df[features]
    y = df['Target']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return model

def predict_next(df, model):
    df = df.copy()
    df['Return'] = df['Close'].pct_change()
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
    df['Volatility_10'] = df['Return'].rolling(window=10).std()
    df['Volume_Change'] = df['Volume'].pct_change()
    df = df.dropna()

    if df.empty:
        return None

    latest = df[['Return', 'MA_5', 'MA_20', 'Momentum_10', 'Volatility_10', 'Volume_Change']].iloc[-1:]

    try:
        return model.predict_proba(latest)[0][1]
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return None