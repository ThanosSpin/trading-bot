from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle
import os

def train_model(df):
    df['Target'] = (df['Return'].shift(-1) > 0).astype(int)
    X = df[['Return']].dropna()
    y = df['Target'].dropna()
    model = RandomForestClassifier()
    model.fit(X, y)
    
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    return model

def load_model():
    with open('models/model.pkl', 'rb') as f:
        return pickle.load(f)

def predict_next(df, model):
    latest = df[['Return']].iloc[-1:]
    return model.predict_proba(latest)[0][1]
