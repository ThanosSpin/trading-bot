from strategy import should_trade
from broker import api
from portfolio import load_portfolio, update_portfolio
from model import load_model, predict_next
from data_loader import fetch_historical_data, fetch_latest_price
from config import USE_LIVE_TRADING, SYMBOL
from market import is_market_open, is_trading_day
import time


def execute_trade(action, quantity, symbol):
    if not is_trading_day() or not is_market_open():
        print(f"⏳ Market is closed or it's a holiday. Skipping trade for {symbol}.")
        return

    if not USE_LIVE_TRADING:
        print(f"[SIMULATION] {action.upper()} {quantity} share(s) of {symbol}")
        return

    try:
        order = api.submit_order(
            symbol=symbol,
            qty=quantity,
            side=action,
            type='market',
            time_in_force='gtc'
        )
        print(f"[LIVE] {action.upper()} order submitted for {symbol}: ID={order.id}, status={order.status}")
        time.sleep(2)
        order_result = api.get_order(order.id)
        print(f"[LIVE] Order status for {symbol}: {order_result.status}")

    except Exception as e:
        print(f"[ERROR] Failed to execute trade for {symbol}: {e}")


def process_symbol(symbol):
    print(f"\n--- Processing {symbol} ---")

    # Prediction: last 6 months of data
    df_recent = fetch_historical_data(symbol, period="6mo", interval="1d")
    if df_recent is None or df_recent.empty:
        print(f"[WARN] Recent data missing for {symbol}. Skipping prediction.")
        return

    model = load_model(symbol)
    if model is None:
        print(f"[ERROR] No model found for {symbol}. Skipping.")
        return

    prob_up = predict_next(df_recent, model)
    if prob_up is None:
        print(f"[INFO] Skipping trade decision for {symbol} due to invalid prediction.")
        return

    total_symbols = len(SYMBOL) if isinstance(SYMBOL, list) else 1
    action, quantity = should_trade(symbol, prob_up, total_symbols=total_symbols)
    print(f"{symbol} → Prediction: {prob_up:.2f}, Action: {action.upper()} {quantity}")

    portfolio = load_portfolio(symbol)
    price = fetch_latest_price(symbol)

    if action in ["buy", "sell"] and quantity > 0 and price:
        execute_trade(action, quantity, symbol)
        portfolio = update_portfolio(action, price, portfolio, symbol)
    else:
        print(f"[INFO] No action taken for {symbol}.")


def run_trading_bot(symbols):
    if isinstance(symbols, str):
        symbols = [symbols]

    for symbol in symbols:
        process_symbol(symbol)


if __name__ == "__main__":
    run_trading_bot(SYMBOL)