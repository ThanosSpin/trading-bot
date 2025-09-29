# main.py
from data_loader import fetch_historical_data, fetch_latest_price
from model_xgb import load_model, predict_next
from strategy import should_trade
from portfolio import load_portfolio, update_portfolio, save_portfolio, portfolio_value
from trader import execute_trade, is_market_open
from config import SYMBOL  # This can now be a list of symbols

# Prediction settings: use last 6 months for prediction
PREDICTION_PERIOD = "6mo"
PREDICTION_INTERVAL = "1d"


def process_symbol(symbol):
    print(f"\n--- Processing {symbol} ---")

    # Load recent historical data for prediction only (6 months)
    df_recent = fetch_historical_data(symbol, period=PREDICTION_PERIOD, interval=PREDICTION_INTERVAL)
    if df_recent is None or df_recent.empty:
        print(f"[WARN] Recent data missing for {symbol}. Skipping prediction.")
        return

    # Load trained model (trained on 2 years of data)
    model = load_model(symbol)
    if model is None:
        print(f"[ERROR] No model found for {symbol}. Skipping.")
        print(f"[DEBUG] Failed to load model for {symbol}. Check if model file exists.")
        return

    # Make prediction
    prob_up = predict_next(df_recent, model)
    if prob_up is None:
        print(f"[INFO] Skipping trade decision for {symbol} due to invalid prediction.")
        print(f"[DEBUG] df_recent last rows:\n{df_recent.tail()}")
        print(f"[DEBUG] Columns available: {df_recent.columns.tolist()}")
        return

    # Load portfolio before making a trade
    portfolio = load_portfolio(symbol)
    cash = float(portfolio.get("cash", 0))
    shares = float(portfolio.get("shares", 0))
    value = portfolio_value(portfolio)

    print(f"[INFO] Current Portfolio for {symbol}: Cash=${cash:.2f}, Shares={shares}, Value=${value:.2f}")

    # Decide trade action
    action, quantity = should_trade(symbol, prob_up)
    print(f"{symbol} → Prediction: {prob_up:.2f}, Action: {action.upper()} {quantity}")

    # Fetch latest market price
    price = fetch_latest_price(symbol)
    if price is None:
        print(f"[WARN] Could not fetch latest price for {symbol}. Skipping trade.")
        print(f"[DEBUG] Current portfolio: {portfolio}")
        return

    # Execute trade if valid
    if action in ["buy", "sell"] and quantity > 0:
        filled_qty, filled_price = execute_trade(action, quantity, symbol)

        if filled_qty and filled_price is not None:
            # use the actual filled values to update portfolio
            portfolio = update_portfolio(action, filled_price, portfolio, symbol, quantity=filled_qty)
            save_portfolio(portfolio, symbol)
            print(f"✅ Updated Portfolio for {symbol}: Cash=${portfolio['cash']:.2f}, Shares={portfolio['shares']}, Value=${portfolio_value(portfolio):.2f}")
        elif filled_qty and filled_price is None:
            # For simulation we may have filled_qty and no precise price
            portfolio = update_portfolio(action, price, portfolio, symbol, quantity=filled_qty)
            save_portfolio(portfolio, symbol)
            print(f"✅ (Sim) Updated Portfolio for {symbol}: Cash=${portfolio['cash']:.2f}, Shares={portfolio['shares']}, Value=${portfolio_value(portfolio):.2f}")
        else:
            print(f"[ERROR] Trade not filled for {symbol}. No portfolio update performed.")

def main():
    if not is_market_open():
        print("⏳ Market is closed. Skipping all trades.")
        return

    # Handle single or multiple symbols
    symbols = SYMBOL if isinstance(SYMBOL, list) else [SYMBOL]

    for symbol in symbols:
        process_symbol(symbol)


if __name__ == "__main__":
    main()