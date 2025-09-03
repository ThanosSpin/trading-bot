# main.py
from data_loader import fetch_historical_data, fetch_latest_price
from model_xgb import load_model, predict_next
from strategy import should_trade
from portfolio import load_portfolio, update_portfolio, save_portfolio, portfolio_value
from trader import execute_trade, is_market_open
from config import SYMBOL  # This can now be a list of symbols

def process_symbol(symbol):
    print(f"\n--- Processing {symbol} ---")

    # Load historical data and model
    df = fetch_historical_data(symbol)
    model = load_model(symbol)

    if model is None:
        print(f"[ERROR] No model found for {symbol}. Skipping.")
        return

    # Make prediction
    prob_up = predict_next(df, model)
    if prob_up is None:
        print(f"[INFO] Skipping trade decision for {symbol} due to invalid prediction.")
        return

    # Decide trade action
    action, quantity = should_trade(symbol, prob_up, total_symbols=len(SYMBOL) if isinstance(SYMBOL, list) else 1)
    print(f"{symbol} → Prediction: {prob_up:.2f}, Action: {action.upper()} {quantity}")

    # Load portfolio and latest price
    portfolio = load_portfolio(symbol)
    price = fetch_latest_price(symbol)

    # Execute trade if valid
    if action in ["buy", "sell"] and quantity > 0 and price:
        execute_trade(action, quantity, symbol)
        portfolio = update_portfolio(action, price, portfolio, symbol)
        save_portfolio(portfolio, symbol)
        print(f"✅ Updated Portfolio Value for {symbol}: ${portfolio_value(portfolio):.2f}")
    else:
        print(f"[INFO] No action taken for {symbol}.")


def main():
    # if not is_market_open():
    #     print("⏳ Market is closed. Skipping all trades.")
    #     return

    # Handle single or multiple symbols
    symbols = SYMBOL if isinstance(SYMBOL, list) else [SYMBOL]

    for symbol in symbols:
        process_symbol(symbol)


if __name__ == "__main__":
    main()