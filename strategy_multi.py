# strategy.py

from config.config_multi import THRESHOLD
from portfolio_multi import load_portfolio
from data_loader_multi import fetch_latest_price

def should_trade(symbol, prob_up, total_symbols=2):
    """
    Decide trade action and quantity based on model probability and portfolio state.

    Capital is split among all symbols.
    """
    portfolio = load_portfolio(symbol)
    shares = portfolio.get("shares", 0)
    print(f"{symbol} → You currently hold {shares} shares.")
    cash = portfolio.get("cash", 0)
    price = fetch_latest_price(symbol)

    upper = 0.5 + THRESHOLD
    lower = 0.5 - THRESHOLD

    if price is None:
        return ("hold", 0)

    # Split cash across number of symbols
    cash_per_symbol = cash / total_symbols

    if prob_up > upper and cash_per_symbol >= price:
        quantity = int(cash_per_symbol // price)
        return ("buy", quantity if quantity > 0 else 0)
    elif prob_up < lower and shares > 0:
        return ("sell", int(shares))
    else:
        return ("hold", 0)
