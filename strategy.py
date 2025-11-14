# strategy.py
import math
from config import (
    BUY_THRESHOLD,
    SELL_THRESHOLD,
    STOP_LOSS,
    TAKE_PROFIT,
    RISK_FRACTION,
    SYMBOL,
)
from portfolio import load_portfolio, portfolio_value, get_live_portfolio
from data_loader import fetch_latest_price


def should_trade(symbol, prob_up, total_symbols=1, concurrent_buys=1):
    """
    Upgraded multi-symbol allocation logic:

    RULES:
    - If total_symbols == 1:
        allocate = cash  (full cash)
    - If total_symbols == 2 AND concurrent_buys == 2:
        allocate = value * RISK_FRACTION
    - If total_symbols == 2 AND concurrent_buys == 1:
        allocate = value   (full portfolio value)
    """

    # Prefer live portfolio if available
    try:
        portfolio = get_live_portfolio(symbol)
    except Exception:
        portfolio = load_portfolio(symbol)

    shares = float(portfolio.get("shares", 0.0))
    cash = float(portfolio.get("cash", 0.0))
    last_price = float(portfolio.get("last_price", 0.0))
    price = fetch_latest_price(symbol)

    if price is None or price <= 0:
        print(f"[WARN] {symbol} invalid price: {price}")
        return ("hold", 0)

    value = portfolio_value(portfolio)

    # --------------------------
    # ALLOCATION LOGIC (your rules)
    # --------------------------
    if total_symbols == 1:
        max_invest = cash

    elif total_symbols == 2:
        if concurrent_buys == 2:
            # both symbols buy → use risk fraction
            max_invest = value * RISK_FRACTION
        else:
            # only one buy signal → use full capital
            max_invest = value

    else:
        # fallback for >2 symbols (not currently used)
        max_invest = value * RISK_FRACTION

    # --------------------------
    # Stop-loss / Take-profit
    # --------------------------
    if shares > 0 and last_price > 0:
        if price <= last_price * STOP_LOSS:
            return ("sell", int(shares))

        if price >= last_price * TAKE_PROFIT:
            return ("sell", int(shares))

    # --------------------------
    # BUY LOGIC
    # --------------------------
    if prob_up >= BUY_THRESHOLD:
        affordable = int(cash // price)
        target = int(max_invest // price)
        quantity = min(affordable, target)

        print(f"[DEBUG] BUY {symbol}: prob_up={prob_up:.2f}, price={price:.2f}, "
              f"cash={cash:.2f}, value={value:.2f}, max_invest={max_invest:.2f}, "
              f"affordable={affordable}, target={target}, qty={quantity}")

        if quantity > 0:
            return ("buy", quantity)
        return ("hold", 0)

    # --------------------------
    # SELL LOGIC
    # --------------------------
    if prob_up <= SELL_THRESHOLD and shares > 0:
        return ("sell", int(shares))

    return ("hold", 0)

