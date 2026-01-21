# trader.py
import time
import alpaca_trade_api as tradeapi
import pytz
import math
from alpaca_client import api
from datetime import datetime, timedelta
from collections import defaultdict
from pdt_guardrails import max_sell_allowed
from pdt_tracker import add_opened_today, reduce_opened_today

from config import (
    API_MARKET_KEY, API_MARKET_SECRET, MARKET_BASE_URL,
    USE_LIVE_TRADING
)
from market import is_market_open, is_trading_day
from data_loader import fetch_latest_price
from portfolio import PortfolioManager


api = tradeapi.REST(API_MARKET_KEY, API_MARKET_SECRET, MARKET_BASE_URL, api_version='v2')
UTC = pytz.UTC


# =====================================================================
# PDT (Pattern Day Trading) Utilities
# =====================================================================
def estimate_daytrade_count(api_client, days=5):
    """
    Less noisy estimate:
    - groups by (symbol, date)
    - uses total filled qty on buy/sell instead of order count
    - converts to an approximate number of round-trips by flooring matched qty.
    NOTE: still an estimate; use as WARNING only, not enforcement.
    """
    cutoff = datetime.utcnow().replace(tzinfo=UTC) - timedelta(days=days)

    try:
        orders = api_client.list_orders(status="filled", limit=1000, nested=True)
    except Exception as e:
            print(f"[WARN] Unable to fetch filled orders for PDT estimate: {e}")
            return 0

    qty = defaultdict(lambda: {"buy": 0.0, "sell": 0.0})

    for o in orders:
        ts = getattr(o, "filled_at", None)
        if ts is None:
            continue
        
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        if ts < cutoff:
            continue

        sym = getattr(o, "symbol", None)
        side = (getattr(o, "side", "") or "").lower()
        if not sym or side not in ("buy", "sell"):
            continue

        try:
            q = float(getattr(o, "filled_qty", 0) or 0)
        except Exception:
            q = 0.0
        if q <= 0:
            continue

        qty[(sym, ts.date())][side] += q

    # 1 per symbol/day if both sides traded
    est = 0
    for v in qty.values():
        if v["buy"] > 0 and v["sell"] > 0:
            est += 1
    return est


def is_buy_allowed_by_pdt(api_client, symbol, quantity):
    """Block BUY only when Alpaca indicates PDT restriction is active."""
    try:
        acct = api_client.get_account()
        equity = float(acct.equity or 0)
        dt_api = int(acct.daytrade_count or 0)
        is_flagged = bool(getattr(acct, "pattern_day_trader", False))
        trading_blocked = bool(getattr(acct, "trading_blocked", False))
    except Exception as e:
        print(f"[WARN] PDT account fetch failed: {e}")
        return False  # fail-closed for safety

    # If Alpaca says trading is blocked, stop.
    if trading_blocked:
        print(f"[PDT BLOCK] Account trading_blocked=true → cannot BUY {symbol}.")
        return False

    # If PDT-flagged and under 25k, buys are typically blocked/restricted.
    if is_flagged and equity < 25000:
        print(f"[PDT BLOCK] Account PDT-flagged and under 25k → cannot BUY {symbol}.")
        return False

    # ✅ TRUST Alpaca's official daytrade_count for enforcement.
    # PDT rule (under 25k): 4+ day trades in 5 days triggers restriction.
    if equity < 25000 and dt_api >= 4:
        print(f"[PDT BLOCK] {symbol}: BUY blocked by Alpaca daytrade_count={dt_api} (equity={equity:.2f}).")
        return False

    # Optional: use estimator as WARNING only (do NOT block on it)
    try:
        dt_est = estimate_daytrade_count(api_client)
        if equity < 25000 and dt_est >= 4 and dt_api < 4:
            print(
                f"[PDT WARN] Estimator suggests dt_est={dt_est} but Alpaca says dt_api={dt_api}. "
                f"Allowing BUY; verify filled orders if you see broker rejections."
            )
    except Exception:
        pass

    return True

def get_pdt_status():
    try:
        acct = api.get_account()
        eq = float(acct.equity or 0)
        dt_api = int(acct.daytrade_count or 0)

        return {
            "daytrade_count": dt_api,
            "remaining": (4 - dt_api) if eq < 25000 else "Unlimited",
            "is_pdt": bool(getattr(acct, "pattern_day_trader", False)),
            "equity": eq,
            "trading_blocked": bool(getattr(acct, "trading_blocked", False)),
        }
    
    except Exception as e:
        print(f"[WARN] PDT status unavailable: {e}")
        return None


# =====================================================================
# Safe fetch latest trade price
# =====================================================================
def _get_live_price(symbol):
    try:
        latest_trade = api.get_latest_trade(symbol)
        px = float(getattr(latest_trade, "price", 0) or 0)
        if px > 0:
            return px
    except Exception:
        pass
    return fetch_latest_price(symbol)


# =====================================================================
# ORDER EXECUTION
# =====================================================================
def execute_trade(action, quantity, symbol):
    """
    Execute a BUY/SELL.
    In simulation: use fetch_latest_price.
    In live mode: submits actual orders.
    """
    action = (action or "").lower().strip()
    try:
        quantity = float(quantity)
    except Exception:
        quantity = 0.0

    if quantity <= 0:
        return 0.0, None

    # -------------------------------------------------------
    # Market closed?
    # -------------------------------------------------------
    if not is_trading_day() or not is_market_open():
        print(f"⏳ Market closed → skipping {action.upper()} {symbol}.")
        return 0.0, None

    # -------------------------------------------------------
    # SIMULATED TRADING
    # -------------------------------------------------------
    if not USE_LIVE_TRADING:
        price = fetch_latest_price(symbol)
        if price:
            print(f"[SIM] {action.upper()} {quantity:g} {symbol} @ {price}")
            return quantity, float(price)
        print(f"[SIM WARN] No price for {symbol}")
        return 0.0, None

    # -------------------------------------------------------
    # LIVE TRADING
    # -------------------------------------------------------
    try:
        acct = api.get_account()
        pdt_status = get_pdt_status()

        # Optional: if Alpaca says trading is blocked, fail fast
        if pdt_status and pdt_status.get("trading_blocked"):
            print("[WARN] Account trading_blocked=true — skipping order.")
            return 0.0, None

        # Default: no cap
        allowed_qty = quantity

        if action == "buy":
            # PDT buy guardrail
            if not is_buy_allowed_by_pdt(api, symbol, quantity):
                return 0.0, None

            price = _get_live_price(symbol)
            if not price or price <= 0:
                print(f"[WARN] No live price for {symbol} — skipping BUY.")
                return 0.0, None

            bp = float(getattr(acct, "buying_power", 0) or 0)
            if price * quantity > bp:
                print(f"[WARN] Buying power insufficient for {symbol}: need {price * quantity:.2f}, have {bp:.2f}")
                return 0.0, None

        elif action == "sell":
            allowed_qty = max_sell_allowed(api, symbol, quantity, pdt_status)

            if allowed_qty <= 0:
                print(f"[INFO] SELL suppressed by PDT guardrail → {symbol}")
                return 0.0, None

        else:
            print(f"[WARN] Unknown action '{action}' for {symbol}")
            return 0.0, None

        # ✅ Only cap after computing allowed_qty (sell caps; buy stays original)
        quantity = allowed_qty

        order = api.submit_order(
            symbol=symbol,
            qty=quantity,
            side=action,
            type="market",
            time_in_force="gtc",
        )
        print(f"[LIVE] Submitted {action.upper()} {quantity:g} {symbol} (id={order.id})")

        time.sleep(2)
        result = api.get_order(order.id)

        filled_qty = float(getattr(result, "filled_qty", 0) or 0)
        filled_price = float(getattr(result, "filled_avg_price", 0) or 0)

        if filled_qty <= 0:
            print(f"[LIVE] Order for {symbol} not filled yet.")
            return 0.0, None

        print(f"[LIVE] Filled {filled_qty} {symbol} @ {filled_price}")

        # Track shares opened today
        if action == "buy":
            add_opened_today(symbol, filled_qty)
        elif action == "sell":
            reduce_opened_today(symbol, filled_qty)

        return filled_qty, filled_price

    except Exception as e:
        msg = str(e)
        low = msg.lower()


        print(f"[ERROR] Trade failed for {symbol}: {msg}")


        pdt_markers = [
            "pattern day trader",
            "day trade buying power",
            "dtbp",
            "daytrade",
            "day-trade",
            "opening trades would exceed",
            "insufficient day trade buying power",
        ]

        if any(m in low for m in pdt_markers):
            print(f"[PDT WARNING] Possible PDT/day-trade restriction for {symbol}.")
           
            return 0.0, None


        return 0.0, None