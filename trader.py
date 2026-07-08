# trader.py
import time
import pytz

from datetime import datetime, timedelta
from collections import defaultdict

from broker import get_trading_api
from pdt.pdt_guardrails import max_sell_allowed
from pdt.pdt_tracker import add_opened_today, reduce_opened_today, get_opened_today_qty

from config import USE_LIVE_TRADING, PAPER_TRADE_SYMBOLS
from market import is_market_open, is_trading_day
from predictive_model.data_loader import fetch_latest_price
from portfolio import PortfolioManager



UTC = pytz.UTC

def _api():
    return get_trading_api()

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


def is_buy_allowed_by_margin(api_client, symbol, quantity, min_equity_for_margin=2000.0):
    """
    Margin-aware BUY guard under Alpaca's intraday margin framework.
    - Uses equity and buying_power instead of PDT/daytrade_count.
    - Respects trading_blocked flag.
    - Optionally enforces a minimum equity threshold for leveraged trading.
    """
    symU = str(symbol).upper().strip()

    try:
        acct = api_client.get_account()
        equity = float(getattr(acct, "equity", 0.0) or 0.0)
        buying_power = float(getattr(acct, "buying_power", 0.0) or 0.0)
        trading_blocked = bool(getattr(acct, "trading_blocked", False))
    except Exception as e:
        print(f"[WARN] Margin account fetch failed: {e}")
        # Fail-closed for safety: if we don't know margin state, do not open new buys.
        return False

    # Broker-side block
    if trading_blocked:
        print(f"[MARGIN BLOCK] Account trading_blocked=true → cannot BUY {symU}.")
        return False

    # Optional: require some minimum equity before allowing margin-driven buys
    if equity < min_equity_for_margin:
        print(f"[MARGIN CAUTION] Equity={equity:.2f} below bot threshold {min_equity_for_margin:.2f} → suppressing margin BUY for {symU}.")
        # You can choose to allow small cash-only buys here instead of outright blocking.
        # For now, keep it conservative:
        return False

    # Check if this order would clearly exceed buying power (rough sanity check)
    try:
        qty_f = float(quantity or 0.0)
    except Exception:
        qty_f = 0.0

    if qty_f <= 0:
        print(f"[MARGIN WARN] Non-positive quantity for {symU}, skipping BUY.")
        return False

    # Use live price approximation to estimate required notional
    price = _get_live_price(symU)
    if not price or price <= 0:
        print(f"[MARGIN WARN] No live price for {symU} — cannot validate margin, skipping BUY.")
        return False

    required_notional = qty_f * price
    if required_notional > buying_power:
        print(f"[MARGIN BLOCK] {symU}: required {required_notional:.2f} > buying_power {buying_power:.2f} → cannot BUY.")
        return False

    return True


def get_margin_status(api=None):
    """
    Lightweight snapshot of account margin state under intraday margin framework.
    Replaces deprecated PDT-based status.
    """
    try:
        account = _api().get_account()

        equity = float(getattr(account, "equity", 0.0) or 0.0)
        buying_power = float(getattr(account, "buying_power", 0.0) or 0.0)
        multiplier = getattr(account, "multiplier", None)  # e.g., "1", "2"
        trading_blocked = bool(getattr(account, "trading_blocked", False))

        return {
            "equity": equity,
            "buying_power": buying_power,
            "multiplier": multiplier,
            "trading_blocked": trading_blocked,
        }
    except Exception as e:
        print(f"[WARN] Margin account fetch failed: {e}")
        return None


# =====================================================================
# Safe fetch latest trade price
# =====================================================================
def _get_live_price(symbol):
    symU = str(symbol).upper().strip()
    try:
        latest_trade = _api().get_latest_trade(symU)
        px = float(getattr(latest_trade, "price", 0) or 0)
        if px > 0:
            return px
    except Exception:
        pass
    return fetch_latest_price(symU)


# =====================================================================
# Log paper trade
# =====================================================================
def log_paper_trade(symbol, action, quantity, price, cash=0.0, shares_after=0.0, value=0.0):
    """
    Log paper trades with the same schema as trades_<symbol>.csv (pm.log()).
    Ensures analyze_trades.py works on paper_trades_<symbol>.csv directly.
    """
    import csv, os
    from datetime import datetime


    filename = f'paper_trades_{symbol}.csv'
    file_exists = os.path.exists(filename)

    try:
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)


            if not file_exists:
                writer.writerow([
                    "timestamp", "symbol", "action", "qty",
                    "price", "cash", "shares", "value",
                    "shares_before", "shares_after",
                ])


            writer.writerow([
                datetime.utcnow().isoformat(),
                symbol,
                action.upper(),
                f"{float(quantity):g}",
                f"{float(price):.2f}",
                f"{float(cash):.2f}",
                f"{float(shares_after):.8g}",
                f"{float(value):.2f}",
                "",
                f"{float(shares_after):.8g}",
            ])

    except Exception as e:
        print(f"[WARN] Failed to log paper trade for {symbol}: {e}")

# =====================================================================
# ORDER EXECUTION
# =====================================================================
def execute_trade(action, quantity, symbol, decision=None):
    """
    Execute a BUY/SELL.
    In simulation: use fetch_latest_price.
    In live mode: submits actual orders.
    Per-symbol paper trading: respects PAPER_TRADE_SYMBOLS even if USE_LIVE_TRADING=True.
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
    # CHECK: Per-Symbol Paper Trading Override
    # -------------------------------------------------------
    # If this symbol is in PAPER_TRADE_SYMBOLS, force paper mode
    # even if USE_LIVE_TRADING=True
    is_paper_symbol = symU in [s.upper() for s in PAPER_TRADE_SYMBOLS]
    
    # -------------------------------------------------------
    # SIMULATED/PAPER TRADING
    # -------------------------------------------------------
    if not USE_LIVE_TRADING or is_paper_symbol:
        price = fetch_latest_price(symU)
        if price:
            mode = "[PAPER]" if is_paper_symbol else "[SIM]"
            print(f"{mode} {action.upper()} {quantity:g} {symU} @ {price}")
            
            # Log paper trades separately for analysis
            if is_paper_symbol:
                         # Load pm state to capture cash/shares/value for correct schema
                try:
                    pm = PortfolioManager(symU)
                    pm.refresh_live()
                    _cash   = float(pm.data.get("cash", 0.0))
                    _shares = float(pm.data.get("shares", 0.0))
                    # Simulate post-trade shares for logging
                    if action == "buy":
                        _shares_after = _shares + quantity
                        _cash_after   = _cash - quantity * price   # spent cash
                    else:
                        _shares_after = max(0.0, _shares - quantity)
                        _cash_after   = _cash + quantity * price   # received cash
                    _value = _cash_after + _shares_after * price
                except Exception:
                    _cash, _shares_after, _value = 0.0, 0.0, 0.0

                log_paper_trade(symU, action, quantity, price,
                                cash=_cash, shares_after=_shares_after, value=_value)
            
            return quantity, float(price)
        
        mode = "PAPER" if is_paper_symbol else "SIM"
        print(f"[{mode} WARN] No price for {symU}")
        return 0.0, None

    # -------------------------------------------------------
    # LIVE TRADING (only if not in PAPER_TRADE_SYMBOLS)
    # -------------------------------------------------------
    try:
        client = _api()
        acct = client.get_account()
        margin = get_margin_status()

        if margin and margin.get("trading_blocked"):
            print("[WARN] Account trading_blocked=true — skipping order.")
            return 0.0, None

        allowed_qty = quantity

        if action == "buy":
            if not is_buy_allowed_by_margin(client, symU, quantity):
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
            margin_status = get_margin_status()

            try:
                allowed_qty = max_sell_allowed(client, symU, quantity, margin_status, emergency=emergency)
            except TypeError:
                allowed_qty = max_sell_allowed(client, symU, quantity, margin_status)

            if allowed_qty <= 0:
                print(f"[INFO] SELL suppressed by PDT guardrail → {symU}")
                return 0.0, None

        else:
            print(f"[WARN] Unknown action '{action}' for {symU}")
            return 0.0, None

        quantity = float(allowed_qty)

        order = client.submit_order(
            symbol=symU,
            qty=quantity,
            side=action,
            type="market",
            time_in_force="gtc",
        )
        print(f"🟢 [LIVE] Submitted {action.upper()} {quantity:g} {symU} (id={order.id})")

        time.sleep(2)
        result = client.get_order(order.id)

        filled_qty = float(getattr(result, "filled_qty", 0) or 0)
        filled_price = float(getattr(result, "filled_avg_price", 0) or 0)

        if filled_qty <= 0:
            print(f"[LIVE] Order for {symU} not filled yet.")
            return 0.0, None

        print(f"🟢 [LIVE] Filled {filled_qty} {symU} @ {filled_price}")

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
