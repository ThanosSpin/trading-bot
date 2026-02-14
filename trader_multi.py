# trader.py

import time
import alpaca_trade_api as tradeapi
from config.config import API_MARKET_KEY, API_MARKET_SECRET, MARKET_BASE_URL, USE_LIVE_TRADING, SYMBOL
from market import is_market_open, is_trading_day
from strategy_multi import should_trade
from updated_model_xgb import load_model, predict_next  # or dynamically select model per symbol
from portfolio_multi import load_portfolio, update_portfolio, portfolio_value
from predictive_model.data_loader import fetch_historical_data, fetch_latest_price

# Alpaca API instance
api = tradeapi.REST(API_MARKET_KEY, API_MARKET_SECRET, MARKET_BASE_URL, api_version='v2')

def execute_trade(symbol: str, action: str, quantity: int):
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
        result = api.get_order(order.id)
        print(f"[LIVE] {symbol} Order status: {result.status}")
        if result.status == "filled":
            print(f"✅ {symbol} order filled at ${result.filled_avg_price} for {result.filled_qty} share(s).")
        elif result.status == "rejected":
            print(f"❌ {symbol} order rejected. Reason: {result.fail_reason or 'Not specified'}")
        else:
            print(f"ℹ️ {symbol} order is pending or partially filled.")
    except Exception as e:
        print(f"[ERROR] Failed to execute trade for {symbol}: {e}")


def run_trading_bot():
    if not is_trading_day() or not is_market_open():
        print("⏳ Market is closed or it's a holiday. Skipping trades.")
        return

    for symbol in SYMBOL:
        print(f"\n🔍 Processing symbol: {symbol}")

        # Load model and historical data
        model = load_model(symbol)
        df = fetch_historical_data(symbol)

        prob_up = predict_next(df, model)
        if prob_up is None:
            print(f"[INFO] Skipping {symbol} — invalid prediction.")
            continue

        action, quantity = should_trade(prob_up, symbol)
        price = fetch_latest_price(symbol)
        portfolio = load_portfolio(symbol)

        print(f"📈 Prediction: {prob_up:.2f}, Action: {action.upper()} {quantity}, Price: ${price:.2f}")

        if action in ["buy", "sell"] and quantity > 0:
            execute_trade(symbol, action, quantity)
            updated = update_portfolio(action, price, portfolio, symbol)
            print(f"💼 {symbol} Portfolio Value: ${portfolio_value(updated):.2f}")
        else:
            print(f"🟡 {symbol}: No action taken.")


if __name__ == "__main__":
    run_trading_bot()