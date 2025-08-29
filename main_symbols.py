# main.py
from config import SYMBOL
from data_loader_multi import fetch_historical_data, fetch_latest_price
from updated_model_xgb import load_model, predict_next
from strategy_multi import should_trade
from portfolio_multi import load_portfolio, update_portfolio, save_portfolio, portfolio_value
from trader_multi import execute_trade, is_market_open

def main():
    if not is_market_open():
        print("⏳ Market is closed. Skipping all trades.")
        return

    for symbol in SYMBOL:
        print(f"\n--- Processing {symbol} ---")

        # Load historical data and model
        df = fetch_historical_data(symbol)
        model = load_model(symbol)

        if model is None or df.empty:
            print(f"[ERROR] No model or data available for {symbol}. Skipping.")
            continue

        # Predict probability of price going up
        prob_up = predict_next(df, model)
        if prob_up is None:
            print(f"[INFO] Skipping {symbol}: Invalid prediction.")
            continue

        # Decide trade action & quantity based on shared capital
        action, quantity = should_trade(symbol, prob_up, total_symbols=len(SYMBOL))
        print(f"{symbol} → Prediction: {prob_up:.2f}, Action: {action.upper()} {quantity}")

        if action not in ["buy", "sell"] or quantity <= 0:
            print(f"[INFO] No trade action required for {symbol}.")
            continue

        # Execute trade
        price = fetch_latest_price(symbol)
        if price is None:
            print(f"[ERROR] Could not fetch price for {symbol}.")
            continue

        execute_trade(action, quantity, symbol)
        portfolio = update_portfolio(action, price, load_portfolio(symbol), symbol)
        save_portfolio(portfolio, symbol)
        print(f"✅ Updated Portfolio Value for {symbol}: ${portfolio_value(portfolio):.2f}")

if __name__ == "__main__":
    main()