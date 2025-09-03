# main.py
from data_loader import fetch_historical_data, fetch_latest_price
from model_xgb import load_model, predict_next
from strategy import should_trade
from portfolio import load_portfolio, update_portfolio, save_portfolio, portfolio_value
from trader import execute_trade, is_market_open
from config import SYMBOL  # list of SYMBOLs

def main():
    # Check market status
    if not is_market_open():
        print("⏳ Market is closed. Skipping this trade.")
        return

    print(f"\n--- Processing {SYMBOL} ---")

    # Load historical data and model
    df = fetch_historical_data(SYMBOL)
    model = load_model(SYMBOL)

    if model is None:
        print(f"[ERROR] No model found for {SYMBOL}. Skipping.")

    # Make prediction
    prob_up = predict_next(df, model)
    if prob_up is None:
        print(f"[INFO] Skipping trade decision for {SYMBOL} due to invalid prediction.")

    # Decide trade action
    action, quantity = should_trade(SYMBOL, prob_up, total_symbols=len(SYMBOL))
    print(f"{SYMBOL} → Prediction: {prob_up:.2f}, Action: {action.upper()} {quantity}")

    # Load portfolio and price
    portfolio = load_portfolio(SYMBOL)
    price = fetch_latest_price(SYMBOL)

    # Execute trade if valid
    if action in ["buy", "sell"] and quantity > 0 and price:
        execute_trade(action, quantity, SYMBOL)
        portfolio = update_portfolio(action, price, portfolio, SYMBOL)
        save_portfolio(portfolio, SYMBOL)
        print(f"✅ Updated Portfolio Value for {SYMBOL}: ${portfolio_value(portfolio):.2f}")
    else:
        print(f"[INFO] No action taken for {SYMBOL}.")

if __name__ == "__main__":
    main()