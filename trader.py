import time
import alpaca_trade_api as tradeapi
from config import API_MARKET_KEY, API_MARKET_SECRET, MARKET_BASE_URL, USE_LIVE_TRADING, SYMBOL
from market import is_market_open, is_trading_day
from strategy import should_trade
from model import predict_market_direction
from portfolio import load_portfolio

# Initialize Alpaca API
api = tradeapi.REST(API_MARKET_KEY, API_MARKET_SECRET, MARKET_BASE_URL, api_version='v2')

def execute_trade(action, quantity, symbol):
    if not is_trading_day() or not is_market_open():
        print("⏳ Market is closed or it's a holiday. Skipping this trade.")
        return

    if not USE_LIVE_TRADING:
        print(f"[SIMULATION] {action.upper()} {quantity} share(s) of {symbol}")
        return

    try:
        # Submit order
        order = api.submit_order(
            symbol=symbol,
            qty=quantity,
            side=action,
            type='market',
            time_in_force='gtc'
        )

        print(f"[LIVE] {action.upper()} order submitted: ID={order.id}, status={order.status}")
        time.sleep(2)  # Wait for order to process
        order_result = api.get_order(order.id)

        print(f"[LIVE] Order status: {order_result.status}")
        if order_result.status == "filled":
            print(f"✅ Order filled at ${order_result.filled_avg_price} "
                  f"for {order_result.filled_qty} share(s) of {symbol}.")
        elif order_result.status == "rejected":
            print(f"❌ Order rejected. Reason: {order_result.fail_reason or 'Not specified'}")
        else:
            print(f"ℹ️ Order still pending or partially filled.")
    except Exception as e:
        print(f"[ERROR] Failed to execute trade for {symbol}: {e}")

def run_trading_bot():
    prob_up = predict_market_direction()
    action = should_trade(prob_up)
    portfolio = load_portfolio()

    print(f"📈 Prediction: {prob_up:.2f}, Action: {action}")

    if action == "buy":
        last_price = portfolio["last_price"]
        cash = portfolio["cash"]
        quantity = int(cash // last_price)
        if quantity > 0:
            execute_trade("buy", quantity)
        else:
            print("[INFO] Not enough cash to buy.")
    elif action == "sell":
        quantity = int(portfolio["shares"])
        if quantity > 0:
            execute_trade("sell", quantity)
        else:
            print("[INFO] No shares to sell.")
    else:
        print("🟡 Hold — No action taken.")

if __name__ == "__main__":
    run_trading_bot()