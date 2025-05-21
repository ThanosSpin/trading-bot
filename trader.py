# trader.py
import time
import alpaca_trade_api as tradeapi
from config import API_MARKET_KEY, API_MARKET_SECRET, MARKET_BASE_URL, USE_LIVE_TRADING, SYMBOL
from market import is_market_open 

# Alpaca API setup
api = tradeapi.REST(API_MARKET_KEY, API_MARKET_SECRET, MARKET_BASE_URL, api_version='v2')

def execute_trade(action, quantity=1):
    if not is_market_open():
        print("⏳ Market is closed. Skipping this trade.")
        return

    if not USE_LIVE_TRADING:
        print(f"[SIMULATION] {action.upper()} {quantity} share(s) of {SYMBOL}")
        return

    try:
        # Submit the live order
        order = api.submit_order(
            symbol=SYMBOL,
            qty=quantity,
            side=action,
            type='market',
            time_in_force='gtc'
        )

        print(f"[LIVE] {action.upper()} order submitted: ID={order.id}, status={order.status}")

        # Give the order time to process (e.g., 2 seconds)
        time.sleep(2)
        order_result = api.get_order(order.id)

        print(f"[LIVE] Order status: {order_result.status}")
        if order_result.status == "filled":
            print(f"✅ Order filled at ${order_result.filled_avg_price} for {order_result.filled_qty} share(s).")
        elif order_result.status == "rejected":
            print(f"❌ Order rejected. Reason: {order_result.fail_reason or 'Not specified'}")
        else:
            print(f"ℹ️ Order still pending or partially filled.")

    except Exception as e:
        print(f"[ERROR] Failed to execute trade: {e}")

# Example usage
if __name__ == "__main__":
    execute_trade("buy", 1)  # Try buying 1 share
    execute_trade("sell", 1)  # Try selling 1 share