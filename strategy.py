# strategy.py
from config_multi import THRESHOLD
from portfolio import load_portfolio
from data_loader import fetch_latest_price

def should_trade(symbol, prob_up, total_symbols=2):
    """
    Decide trade action and quantity based on model probability and portfolio state.
    - Capital is split across all symbols.
    - Buys if prob_up is above threshold and cash allows.
    - Sells if prob_up is below threshold and shares are held.
    """
    portfolio = load_portfolio(symbol)
    shares = portfolio.get("shares", 0)
    cash = portfolio.get("cash", 0)
    price = fetch_latest_price(symbol)

    upper = 0.5 + THRESHOLD
    lower = 0.5 - THRESHOLD

    if price is None:
        return ("hold", 0)

    # Split available cash across all symbols
    cash_per_symbol = cash / total_symbols if total_symbols > 0 else cash

    if prob_up > upper and cash_per_symbol >= price:
        quantity = int(cash_per_symbol // price)
        return ("buy", quantity if quantity > 0 else 0)

    elif prob_up < lower and shares > 0:
        return ("sell", int(shares))

    else:
        return ("hold", 0)