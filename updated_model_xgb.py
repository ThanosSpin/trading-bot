# updated_model_xgb.py
from xgboost import XGBClassifier
import pandas as pd
import pickle
import os
from datetime import datetime
from config_multi import get_model_path, MODEL_DIR

def prepare_features(df):
    """
    Add technical indicators and drop rows with NaNs.
    """
    df = df.copy()
    df['Return'] = df['Close'].pct_change()
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
    df['Volatility_10'] = df['Return'].rolling(window=10).std()
    df['Volume_Change'] = df['Volume'].pct_change()
    return df.dropna()

def train_model(df, symbol):
    """
    Train XGBoost model and save it with a monthly backup.
    """
    df = prepare_features(df)
    df['Target'] = (df['Return'].shift(-1) > 0).astype(int)
    df = df.dropna()

    features = ['Return', 'MA_5', 'MA_20', 'Momentum_10', 'Volatility_10', 'Volume_Change']
    X = df[features]
    y = df['Target']

    model = XGBClassifier(eval_metric='logloss')  # use_label_encoder deprecated
    model.fit(X, y)

    # Save main model
    model_path = get_model_path(symbol)
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Save monthly backup
    month_str = datetime.now().strftime("%Y_%m")
    backup_path = os.path.join(MODEL_DIR, f"model_{symbol}_{month_str}.pkl")
    with open(backup_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"✅ Model trained and saved for {symbol}. Backup: {backup_path}")
    return model

def load_model(symbol):
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
    """
    Generate probability prediction using the model and input dataframe.
    """
    df = prepare_features(df)
    features = ['Return', 'MA_5', 'MA_20', 'Momentum_10', 'Volatility_10', 'Volume_Change']

    if df.empty or not all(col in df.columns for col in features):
        print("[WARN] Not enough data to compute features for prediction.")
        return None

    latest = df[features].iloc[-1:]
    if latest.empty:
        print("[WARN] No valid row to predict.")
        return None

    return model.predict_proba(latest)[0][1]