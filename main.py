# main.py
from data_loader import fetch_historical_data, fetch_latest_price
from model_xgb import load_model, predict_next
from strategy import should_trade
from portfolio import load_portfolio, update_portfolio, save_portfolio, portfolio_value, get_live_portfolio
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

    # Load trained model
    model = load_model(symbol)
    if model is None:
        print(f"[ERROR] No model found for {symbol}. Skipping.")
        return

    # Make prediction
    prob_up = predict_next(df_recent, model)
    if prob_up is None:
        print(f"[INFO] Skipping trade decision for {symbol} due to invalid prediction.")
        return

    # Decide trade action
    action, quantity = should_trade(symbol, prob_up)
    print(f"{symbol} → Prediction: {prob_up:.2f}, Action: {action.upper()} {quantity}")

    # 🔹 Show current portfolio (using live data if available)
    try:
        portfolio = get_live_portfolio(symbol)
        print(f"[INFO] Current Portfolio for {symbol}: "
              f"Cash=${portfolio['cash']:.2f}, Shares={portfolio['shares']}, "
              f"Value=${portfolio['cash'] + portfolio['shares']*portfolio['last_price']:.2f}")
    except Exception as e:
        print(f"[WARN] Could not fetch live portfolio for {symbol}: {e}")
        portfolio = load_portfolio(symbol)

    price = fetch_latest_price(symbol)
    if price is None:
        print(f"[WARN] Could not fetch latest price for {symbol}. Skipping trade.")
        return

    # Execute trade if valid
    if action in ["buy", "sell"] and quantity > 0:
        execute_trade(action, quantity, symbol)
        portfolio = update_portfolio(action, price, portfolio, symbol)
        save_portfolio(portfolio, symbol)
        print(f"✅ Updated Portfolio Value for {symbol}: ${portfolio_value(portfolio):.2f}")
    else:
        print(f"[INFO] No action taken for {symbol}.")

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