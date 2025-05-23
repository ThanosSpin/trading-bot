# model.py
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle
import os

from data_loader import fetch_historical_data
from config import MODEL_PATH

def train_model(df):
    df = df.copy()
    df['Target'] = (df['Return'].shift(-1) > 0).astype(int)
    df = df.dropna()

    X = df[['Return']]
    y = df['Target']
    model = RandomForestClassifier()
    model.fit(X, y)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    return model

def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

def predict_next(df, model):
    latest = df[['Return']].iloc[-1:]
    return model.predict_proba(latest)[0][1]


def predict_market_direction():
    """Loads data and model, returns the probability that the stock will go up."""
    df = fetch_historical_data()
    df['Return'] = df['Close'].pct_change()
    df = df.dropna()

    model = load_model()
    return predict_next(df, model)
