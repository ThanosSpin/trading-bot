# main.py
from data_loader import fetch_historical_data, fetch_latest_price
from indicators import add_indicators
from model import train_model, load_model, predict_next
from strategy import should_trade
from portfolio import load_portfolio, update_portfolio, save_portfolio, portfolio_value
from trader import execute_trade

import os

# Load or train model
print("[INFO] Fetching data...")
df = fetch_historical_data()
df = add_indicators(df)
df['Return'] = df['Close'].pct_change()
model = train_model(df) if not os.path.exists("models/model.pkl") else load_model()

# Predict next move
prob_up = predict_next(df, model)
action = should_trade(prob_up)

# Load portfolio and price
portfolio = load_portfolio()
price = fetch_latest_price()

print(f"Prediction: {prob_up:.2f}, Action: {action}")

# Execute trade and update portfolio
if action in ["buy", "sell"] and price:
    execute_trade(action)
    portfolio = update_portfolio(action, price, portfolio)
    save_portfolio(portfolio)
    print(f"[INFO] Updated Portfolio Value: {portfolio_value(portfolio):.2f}")
else:
    print("[INFO] No action taken.")
