# main.py
from data_loader import fetch_historical_data, fetch_latest_price
from model_xgb import load_model, predict_next
from strategy import should_trade
from portfolio import load_portfolio, update_portfolio, save_portfolio, portfolio_value
from trader import execute_trade, is_market_open

def main():
    # Load historical data and model
    df = fetch_historical_data()
    model = load_model()

    # Make prediction
    prob_up = predict_next(df, model)
    if prob_up is None:
        print("[INFO] Skipping trade decision due to invalid prediction.")
        return

    # Load portfolio and price
    portfolio = load_portfolio()
    price = fetch_latest_price()

    # Decide trade action
    action, quantity = should_trade(prob_up)
    print(f"Prediction: {prob_up:.2f}, Action: {action.upper()} {quantity}")

    # Check market status
    if not is_market_open():
        print("⏳ Market is closed. Skipping this trade.")
        return

    # Execute trade if valid
    if action in ["buy", "sell"] and quantity > 0 and price:
        execute_trade(action, quantity)
        portfolio = update_portfolio(action, price, portfolio)
        save_portfolio(portfolio)
        print(f"✅ Updated Portfolio Value: ${portfolio_value(portfolio):.2f}")
    else:
        print("[INFO] No action taken.")

if __name__ == "__main__":
    main()