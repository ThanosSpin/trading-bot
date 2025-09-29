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
# prefer live portfolio when available
from portfolio import load_portfolio, portfolio_value, get_live_portfolio
from data_loader import fetch_latest_price

def should_trade(symbol, prob_up):
    """
    Decide trade action and quantity based on model probability, portfolio, and risk management.

    - Use live Alpaca portfolio (get_live_portfolio) if available to avoid stale local state.
    - If not available, fall back to load_portfolio (local JSON).
    - Use RISK_FRACTION allocation across multiple symbols.
    - Apply stop-loss / take-profit first (full sell).
    - Buy only when there's sufficient cash for at least 1 share.
    """
    # Try to get live portfolio first (preferred)
    try:
        portfolio = get_live_portfolio(symbol)
        source = "live"
    except Exception as e:
        portfolio = load_portfolio(symbol)
        source = "local"

    # normalize numeric types
    shares = float(portfolio.get("shares", 0.0))
    cash = float(portfolio.get("cash", 0.0))
    last_price = float(portfolio.get("last_price", 0.0))
    price = fetch_latest_price(symbol)

    if price is None or price <= 0:
        print(f"[WARN] {symbol} latest price invalid: {price}")
        return ("hold", 0)

    # current portfolio market value
    value = portfolio_value(portfolio)

    # determine number of symbols we are managing from config
    symbols = SYMBOL if isinstance(SYMBOL, list) else [SYMBOL]
    total_symbols = len(symbols)

    # allocation rule
    if total_symbols <= 1:
        max_invest = cash   # single-symbol: use available cash
    else:
        max_invest = (value * RISK_FRACTION) / total_symbols

    # --- Risk management checks (stop-loss / take-profit) ---
    if shares > 0 and last_price > 0:
        if price <= last_price * STOP_LOSS:
            print(f"[DEBUG] Stop-loss triggered for {symbol}: price {price:.2f} <= last_price * STOP_LOSS ({last_price:.2f} * {STOP_LOSS})")
            return ("sell", int(shares))
        if price >= last_price * TAKE_PROFIT:
            print(f"[DEBUG] Take-profit triggered for {symbol}: price {price:.2f} >= last_price * TAKE_PROFIT ({last_price:.2f} * {TAKE_PROFIT})")
            return ("sell", int(shares))

    # --- Confidence-based trading: BUY ---
    if prob_up >= BUY_THRESHOLD:
        # how many shares cash can afford now
        affordable_shares = int(math.floor(cash / price)) if price > 0 else 0
        # how many shares the allocation allows
        target_shares = int(math.floor(max_invest / price)) if (max_invest > 0 and price > 0) else 0

        if total_symbols > 1:
            quantity = min(affordable_shares, target_shares)
        else:
            quantity = affordable_shares  # single symbol => use as many as cash affords

        # debug info to understand decision
        print(f"[DEBUG] BUY decision variables for {symbol} (source={source}): prob_up={prob_up:.2f}, price={price:.2f}, cash={cash:.2f}, "
              f"affordable_shares={affordable_shares}, max_invest={max_invest:.2f}, target_shares={target_shares}, selected_qty={quantity}")

        # ensure we actually have money to buy at least one share
        if quantity <= 0:
            print(f"[DEBUG] Skipping BUY for {symbol}: not enough cash to buy 1 share (cash={cash:.2f}, price={price:.2f})")
            return ("hold", 0)

        cost = quantity * price
        # final safeguard (floating rounding tolerance)
        if cost > cash + 1e-8:
            print(f"[DEBUG] Skipping BUY for {symbol}: insufficient cash={cash:.2f}, needed={cost:.2f}")
            return ("hold", 0)

        print(f"[DEBUG] Confidence BUY for {symbol}: prob_up={prob_up:.2f}, quantity={quantity}, cost={cost:.2f}, cash={cash:.2f}")
        return ("buy", int(quantity))

    # --- Confidence-based trading: SELL ---
    if prob_up <= SELL_THRESHOLD and shares > 0:
        if total_symbols <= 1:
            # single-symbol => sell all
            print(f"[DEBUG] Confidence SELL (full) for {symbol}: prob_up={prob_up:.2f}, shares={shares}")
            return ("sell", int(shares))
        else:
            # partial sell according to risk fraction
            sell_shares = max(1, int(math.ceil(shares * RISK_FRACTION)))
            sell_shares = min(int(shares), sell_shares)
            print(f"[DEBUG] Confidence SELL (partial) for {symbol}: prob_up={prob_up:.2f}, shares_to_sell={sell_shares}, total_shares={shares}")
            return ("sell", int(sell_shares))

    return ("hold", 0)