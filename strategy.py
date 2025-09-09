# strategy.py
from config import (
    BUY_THRESHOLD,
    SELL_THRESHOLD,
    STOP_LOSS,
    TAKE_PROFIT,
    RISK_FRACTION,
)
from portfolio import load_portfolio, portfolio_value
from data_loader import fetch_latest_price


def should_trade(symbol, prob_up, total_symbols=1):
    """
    Decide trade action and quantity based on model probability, portfolio, and risk management.
    - Requires higher confidence for buy/sell.
    - Includes stop-loss and take-profit rules.
    - Allocation:
        - If one symbol: trade with all available cash.
        - If multiple symbols: max allocation = RISK_FRACTION of portfolio per symbol.
    """
    portfolio = load_portfolio(symbol)
    shares = portfolio.get("shares", 0)
    cash = portfolio.get("cash", 0)
    last_price = portfolio.get("last_price", 0)  # stored in portfolio.json
    price = fetch_latest_price(symbol)

    if price is None or price <= 0:
        return ("hold", 0)

    value = portfolio_value(portfolio)

    # Allocation rule
    if total_symbols <= 1:
        max_invest = cash  # one symbol → use all cash
    else:
        max_invest = (value * RISK_FRACTION) / total_symbols

    # --- Risk management overrides ---
    if shares > 0 and last_price > 0:
        if price <= last_price * STOP_LOSS:
            return ("sell", shares)  # stop-loss triggered
        elif price >= last_price * TAKE_PROFIT:
            return ("sell", shares)  # take-profit triggered

    # --- Confidence-based trading ---
    if prob_up >= BUY_THRESHOLD:
        affordable_shares = int(cash // price)
        target_shares = int(max_invest // price)
        quantity = min(
            affordable_shares,
            target_shares if total_symbols > 1 else affordable_shares,
        )
        if quantity > 0:
            return ("buy", quantity)

    elif prob_up <= SELL_THRESHOLD and shares > 0:
        # Default: sell RISK_FRACTION of holdings (or at least 1)
        sell_shares = max(1, int(shares * RISK_FRACTION))
        return ("sell", min(sell_shares, shares))

    return ("hold", 0)
