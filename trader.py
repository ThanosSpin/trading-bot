# trader.py
import time
import alpaca_trade_api as tradeapi
from config import API_MARKET_KEY, API_MARKET_SECRET, MARKET_BASE_URL, USE_LIVE_TRADING
from market import is_market_open, is_trading_day
from data_loader import fetch_latest_price  # for simulation price
from config import SYMBOL  # optional usage

# Initialize Alpaca API
api = tradeapi.REST(API_MARKET_KEY, API_MARKET_SECRET, MARKET_BASE_URL, api_version='v2')

def execute_trade(action, quantity, symbol):
    """
    Submit an order and return (filled_qty: float, filled_avg_price: float)
    - In simulation mode returns (quantity, latest_price) if available.
    - On error returns (0.0, None).
    """
    action = action.lower()
    quantity = float(quantity)

    if not is_trading_day() or not is_market_open():
        print(f"⏳ Market is closed or it's a holiday. Skipping trade for {symbol}.")
        return 0.0, None

    if not USE_LIVE_TRADING:
        # Simulation: use latest minute price if available
        sim_price = None
        try:
            sim_price = fetch_latest_price(symbol)
        except Exception:
            pass
        print(f"[SIMULATION] {action.upper()} {quantity:g} share(s) of {symbol} at simulated price {sim_price}")
        return quantity, float(sim_price) if sim_price is not None else None

    try:
        # Submit live market order
        order = api.submit_order(
            symbol=symbol,
            qty=quantity,
            side=action,
            type='market',
            time_in_force='gtc'
        )
        print(f"[LIVE] {action.upper()} order submitted for {symbol}: ID={order.id}, status={order.status}")
        # short wait then fetch updated order
        time.sleep(2)
        order_result = api.get_order(order.id)
        print(f"[LIVE] Order status for {symbol}: {order_result.status}")

        # If filled, return actual filled quantity and average price
        if getattr(order_result, "filled_qty", None) and float(order_result.filled_qty) > 0:
            filled_qty = float(order_result.filled_qty)
            filled_price = float(order_result.filled_avg_price) if getattr(order_result, "filled_avg_price", None) else None
            print(f"[LIVE] Order filled: {filled_qty} @ {filled_price}")
            return filled_qty, filled_price
        else:
            # Could be pending/partial/other
            print(f"[LIVE] Order not fully filled immediately: filled_qty={getattr(order_result, 'filled_qty', None)}")
            return 0.0, None

    except Exception as e:
        print(f"[ERROR] Failed to execute trade for {symbol}: {e}")
        return 0.0, None