# decision_logic.py
from config import THRESHOLD
from portfolio import load_portfolio

def should_trade(prob_up: float) -> str:
    """
    Decide trade action based on model probability and current portfolio.
    Only buy if not holding shares. Only sell if holding shares.
    """
    portfolio = load_portfolio()
    shares = portfolio.get("shares", 0)

    if prob_up > 0.5 + THRESHOLD and shares == 0:
        return "buy"
    elif prob_up < 0.5 - THRESHOLD and shares > 0:
        return "sell"
    else:
        return "hold"

if __name__ == "__main__":
    print(should_trade(0.62))  # depends on portfolio
    print(should_trade(0.48))  # hold
    print(should_trade(0.40))  # depends on portfolio