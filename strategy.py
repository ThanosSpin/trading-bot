# strategy.py
from config import THRESHOLD
from portfolio import load_portfolio

def should_trade(prob_up: float) -> str:
    """
    Decide trade action based on model probability and current portfolio state.
    
    Rules:
    - If shares > 0 and cash > 0: allow buy or sell
    - If shares == 0: only allow buy
    - If cash == 0: only allow sell
    - Otherwise: hold
    """
    portfolio = load_portfolio()
    shares = portfolio.get("shares", 0)
    cash = portfolio.get("cash", 0)

    if shares > 0 and cash > 0:
        if prob_up > 0.5 + THRESHOLD:
            return "buy"
        elif prob_up < 0.5 - THRESHOLD:
            return "sell"
        else:
            return "hold"
    elif shares == 0:
        return "buy" if prob_up > 0.5 + THRESHOLD else "hold"
    elif cash == 0 and shares > 0:
        return "sell" if prob_up < 0.5 - THRESHOLD else "hold"
    else:
        return "hold"

if __name__ == "__main__":
    print(should_trade(0.62))  # Example test
    print(should_trade(0.40))
    print(should_trade(0.51))