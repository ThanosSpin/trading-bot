# model.py
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle
import os

from predictive_model.data_loader import fetch_historical_data
from config.config import get_model_path, SYMBOL


def train_model(df, symbol):
    """
    Train and save a RandomForest model for a given symbol.
    """
    df = df.copy()
    df['Target'] = (df['Return'].shift(-1) > 0).astype(int)
    df = df.dropna()

    X = df[['Return']]
    y = df['Target']

    model = RandomForestClassifier()
    model.fit(X, y)

    model_path = get_model_path(symbol)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"✅ RandomForest model trained and saved for {symbol} at {model_path}")
    return model


def load_model(symbol):
    """
    Load model for the given symbol.
    """
    model_path = get_model_path(symbol)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found for {symbol} at {model_path}")

    with open(model_path, 'rb') as f:
        return pickle.load(f)


def predict_next(df, model):
    """
    Predict probability that next move will be up.
    """
    if df.empty or 'Return' not in df.columns:
        return None

    latest = df[['Return']].iloc[-1:]
    if latest.empty:
        return None

    return model.predict_proba(latest)[0][1]


def predict_market_direction(symbol):
    """
    Loads data and model for a symbol, returns probability of price going up.
    """
    df = fetch_historical_data(symbol)
    df['Return'] = df['Close'].pct_change()
    df = df.dropna()

    model = load_model(symbol)
    return predict_next(df, model)
