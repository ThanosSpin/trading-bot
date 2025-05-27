# strategy.py
from config import THRESHOLD
from portfolio import load_portfolio
from data_loader import fetch_latest_price

def should_trade(prob_up: float):
    """
    Decide trade action and quantity based on model probability and portfolio state.

    - If prediction is high and cash >= 1 share: BUY max with all cash
    - If prediction is low and have shares: SELL all shares
    - Else: HOLD
    """
    portfolio = load_portfolio()
    shares = portfolio.get("shares", 0)
    cash = portfolio.get("cash", 0)
    price = fetch_latest_price()

    upper = 0.5 + THRESHOLD
    lower = 0.5 - THRESHOLD

    if price is None:
        return ("hold", 0)

    if prob_up > upper and cash >= price:
        quantity = int(cash // price)
        return ("buy", quantity if quantity > 0 else 0)
    elif prob_up < lower and shares > 0:
        return ("sell", int(shares))
    else:
        return ("hold", 0)

if __name__ == "__main__":
    print(should_trade(0.62))  # Example
    print(should_trade(0.40))
    print(should_trade(0.51))