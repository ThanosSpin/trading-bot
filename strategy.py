# strategy.py
from config import THRESHOLD
from portfolio import load_portfolio, portfolio_value
from data_loader import fetch_latest_price

def should_trade(symbol, prob_up, total_symbols=1, risk_fraction=0.5):
    """
    Decide trade action and quantity based on model probability and portfolio state.
    - If only one symbol: trade with all available cash.
    - If multiple symbols: max allocation = 50% of portfolio per symbol.
    """
    portfolio = load_portfolio(symbol)
    shares = portfolio.get("shares", 0)
    cash = portfolio.get("cash", 0)
    price = fetch_latest_price(symbol)

    upper = 0.5 + THRESHOLD
    lower = 0.5 - THRESHOLD

    if price is None or price <= 0:
        return ("hold", 0)

    value = portfolio_value(portfolio)

    # Allocation rule
    if total_symbols <= 1:
        max_invest = cash  # one symbol → use all cash
    else:
        max_invest = (value * risk_fraction) / total_symbols

    # BUY logic
    if prob_up > upper:
        affordable_shares = int(cash // price)
        target_shares = int(max_invest // price)
        quantity = min(affordable_shares, target_shares if total_symbols > 1 else affordable_shares)
        if quantity > 0:
            return ("buy", quantity)

    # SELL logic
    elif prob_up < lower and shares > 0:
        # Default: sell 50% of holdings (or at least 1)
        sell_shares = max(1, int(shares * risk_fraction))
        return ("sell", min(sell_shares, shares))

    return ("hold", 0)