# trader.py
import time
import alpaca_trade_api as tradeapi
import pytz

from datetime import datetime, timedelta
from collections import defaultdict

from alpaca_client import api  # NOTE: you also re-init api below; keep one source if possible
from pdt_guardrails import max_sell_allowed
from pdt_tracker import add_opened_today, reduce_opened_today, get_opened_today_qty

from config import (
    API_MARKET_KEY, API_MARKET_SECRET, MARKET_BASE_URL,
    USE_LIVE_TRADING
)
from market import is_market_open, is_trading_day
from data_loader import fetch_latest_price
from portfolio import PortfolioManager


# Keep ONE api instance. If alpaca_client.api is already set, you don't need this.
# But leaving it here is ok as long as you don't import trader.api from data_loader (circular).
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
    - converts to an approximate number of round-trips by counting symbol-days with both sides.
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

        qty[(str(sym).upper(), ts.date())][side] += q

    # 1 per symbol/day if both sides traded
    est = 0
    for v in qty.values():
        if v["buy"] > 0 and v["sell"] > 0:
            est += 1
    return est


def is_buy_allowed_by_pdt(api_client, symbol, quantity):
    """Block BUY only when Alpaca indicates PDT restriction is active."""
    symU = str(symbol).upper().strip()

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
        print(f"[PDT BLOCK] Account trading_blocked=true → cannot BUY {symU}.")
        return False

    # If PDT-flagged and under 25k, buys are typically blocked/restricted.
    if is_flagged and equity < 25000:
        print(f"[PDT BLOCK] Account PDT-flagged and under 25k → cannot BUY {symU}.")
        return False

    # ✅ TRUST Alpaca's official daytrade_count for enforcement.
    if equity < 25000 and dt_api >= 4:
        print(f"[PDT BLOCK] {symU}: BUY blocked by Alpaca daytrade_count={dt_api} (equity={equity:.2f}).")
        return False

    # Optional: estimator as WARNING only (do NOT block on it)
    try:
        dt_est = estimate_daytrade_count(api_client)
        remaining = max(0, 4 - dt_api) if equity < 25000 else None
        if dt_est is not None and dt_est != dt_api and equity < 25000:
            print(f"[PDT WARN] API daytrade_count={dt_api} (remaining {remaining}) | estimator={dt_est}")
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
    symU = str(symbol).upper().strip()
    try:
        latest_trade = api.get_latest_trade(symU)
        px = float(getattr(latest_trade, "price", 0) or 0)
        if px > 0:
            return px
    except Exception:
        pass
    return fetch_latest_price(symU)


# =====================================================================
# ORDER EXECUTION
# =====================================================================
def execute_trade(action, quantity, symbol, decision=None):
    """
    Execute a BUY/SELL.
    In simulation: use fetch_latest_price.
    In live mode: submits actual orders.
    """
    symU = str(symbol).upper().strip()
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
        print(f"⏳ Market closed → skipping {action.upper()} {symU}.")
        return 0.0, None

    # -------------------------------------------------------
    # SIMULATED TRADING
    # -------------------------------------------------------
    if not USE_LIVE_TRADING:
        price = fetch_latest_price(symU)
        if price:
            print(f"[SIM] {action.upper()} {quantity:g} {symU} @ {price}")
            return quantity, float(price)
        print(f"[SIM WARN] No price for {symU}")
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
        

        allowed_qty = quantity

        if action == "buy":
            if not is_buy_allowed_by_pdt(api, symU, quantity):
                return 0.0, None

            price = _get_live_price(symU)
            if not price or price <= 0:
                print(f"[WARN] No live price for {symU} — skipping BUY.")
                return 0.0, None

            bp = float(getattr(acct, "buying_power", 0) or 0)
            if price * quantity > bp:
                print(f"[WARN] Buying power insufficient for {symU}: need {price * quantity:.2f}, have {bp:.2f}")
                return 0.0, None

        elif action == "sell":
            emergency = bool((decision or {}).get("pdt_emergency", False))

            # Backward compatible: if your max_sell_allowed() doesn't accept emergency yet,
            # fall back to the old signature.
            try:
                allowed_qty = max_sell_allowed(api, symU, quantity, pdt_status, emergency=emergency)
            except TypeError:
                allowed_qty = max_sell_allowed(api, symU, quantity, pdt_status)

            if allowed_qty <= 0:
                print(f"[INFO] SELL suppressed by PDT guardrail → {symU}")
                return 0.0, None

        else:
            print(f"[WARN] Unknown action '{action}' for {symU}")
            return 0.0, None

        quantity = float(allowed_qty)

        order = api.submit_order(
            symbol=symU,
            qty=quantity,
            side=action,
            type="market",
            time_in_force="gtc",
        )
        print(f"[LIVE] Submitted {action.upper()} {quantity:g} {symU} (id={order.id})")

        time.sleep(2)
        result = api.get_order(order.id)

        filled_qty = float(getattr(result, "filled_qty", 0) or 0)
        filled_price = float(getattr(result, "filled_avg_price", 0) or 0)

        if filled_qty <= 0:
            print(f"[LIVE] Order for {symU} not filled yet.")
            return 0.0, None

        print(f"[LIVE] Filled {filled_qty} {symU} @ {filled_price}")

        # Track shares opened today (PDT tracker)
        if action == "buy":
            add_opened_today(symU, filled_qty)
        elif action == "sell":
            reduce_opened_today(symU, filled_qty)

        return filled_qty, filled_price

    except Exception as e:
        msg = str(e)
        low = msg.lower()

        print(f"[ERROR] Trade failed for {symU}: {msg}")

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
            print(f"[PDT WARNING] Possible PDT/day-trade restriction for {symU}.")
            return 0.0, None

        return 0.0, None