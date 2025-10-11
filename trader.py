# trader.py
import time
import alpaca_trade_api as tradeapi
import pytz
from datetime import datetime, timedelta
from collections import defaultdict
from config import API_MARKET_KEY, API_MARKET_SECRET, MARKET_BASE_URL, USE_LIVE_TRADING
from market import is_market_open, is_trading_day
from data_loader import fetch_latest_price  # for simulation price

# Initialize Alpaca API
api = tradeapi.REST(API_MARKET_KEY, API_MARKET_SECRET, MARKET_BASE_URL, api_version='v2')


def estimate_daytrade_count(api_client, days=5):
    """
    Conservative estimator of day-trades in the last `days` business days.
    A "day trade" is counted as a matched buy+sell (pair) for the same symbol on the same day.
    We return the total number of such matched pairs across symbols/dates.
    """
    cutoff = datetime.utcnow() - timedelta(days=days)
    cutoff = cutoff.replace(tzinfo=pytz.UTC)
    try:
        orders = api_client.list_orders(status="filled", limit=1000, nested=True)
    except Exception as e:
        print(f"[WARN] Unable to fetch filled orders for PDT estimate: {e}")
        return 0

    # Filter and bucket by (symbol, date)
    buckets = defaultdict(lambda: {"buy": 0, "sell": 0})
    for o in orders:
        filled_at = getattr(o, "filled_at", None)
        if filled_at is None:
            continue
        # ensure tz-aware and compare
        if filled_at.tzinfo is None:
            filled_at = filled_at.replace(tzinfo=pytz.UTC)
        if filled_at < cutoff:
            continue
        date = filled_at.date()
        key = (getattr(o, "symbol", None), date)
        side = getattr(o, "side", "").lower()
        if side in ("buy", "sell"):
            buckets[key][side] += 1

    # Count matched pairs (min(buys, sells)) as day-trades
    matched_pairs = sum(min(v["buy"], v["sell"]) for v in buckets.values())
    return matched_pairs


def is_buy_allowed_by_pdt(api_client, symbol, quantity):
    """
    Return True if buying `quantity` for `symbol` is allowed under PDT heuristics.
    Conservative policy: if account equity < 25k and (API daytrade_count >= 3 or estimated >= 3),
    block buys.
    """
    try:
        acc = api_client.get_account()
        equity = float(getattr(acc, "equity", 0.0) or 0.0)
        api_daytrade_count = int(getattr(acc, "daytrade_count", 0) or 0)
    except Exception as e:
        print(f"[WARN] Cannot fetch account for PDT check: {e}. Blocking BUY as precaution.")
        return False

    # If account is already flagged as PDT, block buys if equity < 25k
    if getattr(acc, "pattern_day_trader", False) and equity < 25000:
        print(f"[WARN] Account flagged as Pattern Day Trader — blocking BUY for {symbol}.")
        return False

    # Estimate day-trades from recent filled orders
    est_count = estimate_daytrade_count(api_client, days=5)

    # If under $25k equity and either API count or estimated count >= 4, block BUY
    if equity < 25000 and (api_daytrade_count >= 4 or est_count >= 4):
        print(f"[PDT PROTECTION] Blocking BUY for {symbol}: equity=${equity:.2f}, "
              f"api_daytrade_count={api_daytrade_count}, est_daytrades={est_count}")
        return False

    return True


def get_pdt_status():
    """
    Return a summary of PDT-related information for logging.
    """
    try:
        account = api.get_account()
        daytrade_count = int(account.daytrade_count or 0)
        equity = float(account.equity or 0)
        is_pdt = account.pattern_day_trader

        remaining = max(0, 4 - daytrade_count) if equity < 25000 else "Unlimited"
        return {
            "daytrade_count": daytrade_count,
            "remaining": remaining,
            "is_pdt": is_pdt,
            "equity": equity,
        }

    except Exception as e:
        print(f"[WARN] Could not fetch PDT status: {e}")
        return None


def execute_trade(action, quantity, symbol):
    """
    Submit an order and return (filled_qty: float, filled_avg_price: float)
    - Simulation mode: returns (quantity, latest_price) where possible.
    - Live mode: returns actual filled_qty and filled_avg_price if filled, or (0.0, None) otherwise.
    """
    action = action.lower()
    try:
        quantity = float(quantity)
    except Exception:
        quantity = float(int(quantity) if isinstance(quantity, (int, float)) else 0.0)

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

    # Live trading path
    try:
        account = api.get_account()

        # pre-check PDT/Buying power for BUYs
        if action == "buy":
            if not is_buy_allowed_by_pdt(api, symbol, quantity):
                # conservative block
                return 0.0, None

            buying_power = float(getattr(account, "buying_power", 0.0) or 0.0)
            try:
                latest_trade = api.get_latest_trade(symbol)
                latest_price = float(getattr(latest_trade, "price", 0.0) or 0.0)
            except Exception:
                latest_price = None

            if latest_price is not None:
                estimated_cost = latest_price * quantity
                if estimated_cost > buying_power:
                    print(f"[WARN] Insufficient buying power to buy {quantity} {symbol} "
                          f"(need ${estimated_cost:.2f}, have ${buying_power:.2f}).")
                    return 0.0, None

        # Submit order
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

        if getattr(order_result, "filled_qty", None) and float(order_result.filled_qty) > 0:
            filled_qty = float(order_result.filled_qty)
            filled_price = float(order_result.filled_avg_price) if getattr(order_result, "filled_avg_price", None) else None
            print(f"[LIVE] Order filled: {filled_qty} @ {filled_price}")
            return filled_qty, filled_price
        else:
            print(f"[LIVE] Order not fully filled immediately: filled_qty={getattr(order_result, 'filled_qty', None)}")
            return 0.0, None

    except Exception as e:
        error_msg = str(e).lower()
        # Handle PDT-specific text returned by Alpaca
        if "pattern day trading" in error_msg or "day trading" in error_msg:
            print(f"[WARN] Trade blocked by Pattern Day Trading rule for {symbol}.")
            print("[INFO] Skipping this trade to comply with PDT restrictions.")
            return 0.0, None
        else:
            print(f"[ERROR] Failed to execute trade for {symbol}: {e}")
            return 0.0, None