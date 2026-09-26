# trader.py
import time
import pytz

from datetime import datetime, timedelta, timezone
from collections import defaultdict

from broker import get_trading_api
from pdt.pdt_tracker import add_opened_today, reduce_opened_today

from config import USE_LIVE_TRADING, PAPER_TRADE_SYMBOLS, ORDER_FILL_TIMEOUT_SECONDS
from market import is_market_open, is_trading_day
from predictive_model.data_loader import fetch_latest_price
from portfolio import PortfolioManager

UTC = pytz.UTC


def _api():
    return get_trading_api()


def _wait_for_terminal_order(client, order_id, timeout_seconds=None):
    """Poll an order until terminal state; cancel any remainder on timeout."""
    timeout = float(timeout_seconds or ORDER_FILL_TIMEOUT_SECONDS)
    deadline = time.monotonic() + max(timeout, 1.0)
    terminal = {"filled", "canceled", "cancelled", "rejected", "expired"}
    result = client.get_order(order_id)

    while str(getattr(result, "status", "")).lower() not in terminal:
        if time.monotonic() >= deadline:
            print(
                f"[LIVE] Order {order_id} did not reach terminal status within "
                f"{timeout:.0f}s; canceling unfilled remainder."
            )
            try:
                client.cancel_order(order_id)
            except Exception as exc:
                print(f"[WARN] Could not cancel timed-out order {order_id}: {exc}")
            cancel_deadline = time.monotonic() + 5.0
            while time.monotonic() < cancel_deadline:
                time.sleep(0.5)
                result = client.get_order(order_id)
                if str(getattr(result, "status", "")).lower() in terminal:
                    break
            return result
        time.sleep(1.0)
        result = client.get_order(order_id)

    return result


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


def is_buy_allowed_by_margin(
    api_client, symbol, quantity, min_equity_for_margin=2000.0
):
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
        print(
            f"[MARGIN CAUTION] Equity={equity:.2f} below bot threshold {min_equity_for_margin:.2f} → suppressing margin BUY for {symU}."
        )
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
        print(
            f"[MARGIN WARN] No live price for {symU} — cannot validate margin, skipping BUY."
        )
        return False

    required_notional = qty_f * price
    if required_notional > buying_power:
        print(
            f"[MARGIN BLOCK] {symU}: required {required_notional:.2f} > buying_power {buying_power:.2f} → cannot BUY."
        )
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


def get_recent_filled_sells(symbols, lookback_hours=24):
    since_dt = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)

    orders = _api().list_orders(
        status="closed",
        limit=1000,
        direction="desc",
        nested=False,
        # Only fetch orders updated within the lookback window
        # (Alpaca's 'until' and 'after' filter by order submission time,
        # so we also filter by filled_at manually below.)
    )

    fills = []
    for o in orders:
        if o.symbol not in [s.upper() for s in symbols]:
            continue
        if o.side != "sell":
            continue

        # Safely convert filled_qty to float
        try:
            filled_qty = float(o.filled_qty)
        except (TypeError, ValueError):
            continue

        if filled_qty <= 0:
            continue

        # Use filled_at if available, otherwise skip
        filled_at = getattr(o, "filled_at", None)
        if filled_at is None:
            continue

        # Ensure filled_at is timezone-aware
        if filled_at.tzinfo is None:
            filled_at = filled_at.replace(tzinfo=timezone.utc)

        # Only include fills within the lookback window
        if filled_at < since_dt:
            continue

        fills.append(
            {
                "id": str(o.id),
                "symbol": o.symbol,
                "side": o.side,
                "filled_qty": filled_qty,
                "filled_at": filled_at.isoformat(),
            }
        )

    return fills


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
def log_paper_trade(
    symbol, action, quantity, price, cash=0.0, shares_after=0.0, value=0.0
):
    """
    Log paper trades with the same schema as trades_<symbol>.csv (pm.log()).
    Ensures analyze_trades.py works on paper_trades_<symbol>.csv directly.
    """
    import csv, os
    from datetime import datetime

    filename = f"paper_trades_{symbol}.csv"
    file_exists = os.path.exists(filename)

    try:
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow(
                    [
                        "timestamp",
                        "symbol",
                        "action",
                        "qty",
                        "price",
                        "cash",
                        "shares",
                        "value",
                        "shares_before",
                        "shares_after",
                    ]
                )

            writer.writerow(
                [
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
                ]
            )

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
                    _cash = float(pm.data.get("cash", 0.0))
                    _shares = float(pm.data.get("shares", 0.0))
                    # Simulate post-trade shares for logging
                    if action == "buy":
                        _shares_after = _shares + quantity
                        _cash_after = _cash - quantity * price  # spent cash
                    else:
                        _shares_after = max(0.0, _shares - quantity)
                        _cash_after = _cash + quantity * price  # received cash
                    _value = _cash_after + _shares_after * price
                except Exception:
                    _cash, _shares_after, _value = 0.0, 0.0, 0.0

                log_paper_trade(
                    symU,
                    action,
                    quantity,
                    price,
                    cash=_cash,
                    shares_after=_shares_after,
                    value=_value,
                )

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
            # if not is_buy_allowed_by_margin(client, symU, quantity):
            #     return 0.0, None

            price = _get_live_price(symU)
            if not price or price <= 0:
                print(f"[WARN] No live price for {symU} — skipping BUY.")
                return 0.0, None

            bp = float(getattr(acct, "buying_power", 0) or 0)
            if price * quantity > bp:
                print(
                    f"[WARN] Buying power insufficient for {symU}: need {price * quantity:.2f}, have {bp:.2f}"
                )
                return 0.0, None

        elif action == "sell":
            # ---------------------------------------------------------
            # SELL validation
            # ---------------------------------------------------------
            # Legacy PDT restrictions removed.
            #
            # Same-day exits are allowed. The only execution-level
            # protection here is that we never sell more shares than
            # the broker says we currently own, preventing an accidental
            # transition from long -> short.
            # ---------------------------------------------------------

            requested_qty = max(0, int(quantity))

            if requested_qty <= 0:
                print(f"[INFO] SELL skipped → {symU}: " f"invalid quantity={quantity}")
                return 0.0, None

            try:
                position = client.get_position(symU)

                current_long_qty = max(0, int(float(position.qty)))

            except Exception as e:
                print(
                    f"[WARN] SELL skipped → {symU}: "
                    f"could not verify broker position: {e}"
                )
                return 0.0, None

            allowed_qty = min(
                requested_qty,
                current_long_qty,
            )

            if allowed_qty <= 0:
                print(
                    f"[INFO] SELL skipped → {symU}: "
                    f"requested={requested_qty}, "
                    f"current_long_position={current_long_qty}"
                )
                return 0.0, None

            if allowed_qty < requested_qty:
                print(
                    f"[SELL CLAMP] {symU}: "
                    f"requested={requested_qty} → "
                    f"allowed={allowed_qty} "
                    f"(broker long position={current_long_qty})"
                )

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
        print(
            f"🟢 [LIVE] Submitted {action.upper()} {quantity:g} {symU} (id={order.id})"
        )

        result = _wait_for_terminal_order(client, order.id)

        filled_qty = float(getattr(result, "filled_qty", 0) or 0)
        filled_price = float(getattr(result, "filled_avg_price", 0) or 0)
        final_status = str(getattr(result, "status", "unknown")).lower()

        if filled_qty <= 0:
            print(f"[LIVE] Order for {symU} ended status={final_status} with no fill.")
            return 0.0, None

        fill_label = "Filled" if filled_qty >= quantity else "Partially filled"
        print(
            f"🟢 [LIVE] {fill_label} {filled_qty}/{quantity:g} {symU} "
            f"@ {filled_price} (status={final_status})"
        )

        # Track shares opened today (PDT tracker)
        if action == "buy":
            add_opened_today(symU, filled_qty)
        elif action == "sell":
            reduce_opened_today(symU, filled_qty)

        return filled_qty, filled_price

    except Exception as e:
        msg = str(e)

        print(f"[ERROR] Trade failed for {symU}: {msg}")

        return 0.0, None
