# strategy.py
from config import THRESHOLD
from portfolio import load_portfolio

def should_trade(prob_up: float) -> str:
    """
    Decide trade action based on model probability and current portfolio state.

    Strategy:
    - If prediction is high and no shares: BUY
    - If prediction is low and have shares: SELL
    - Else: HOLD
    """
    portfolio = load_portfolio()
    shares = portfolio.get("shares", 0)
    cash = portfolio.get("cash", 0)

    # Decision thresholds
    upper = 0.5 + THRESHOLD
    lower = 0.5 - THRESHOLD

    if prob_up > upper:
        if shares == 0 and cash > 0:
            return "buy"
        else:
            return "hold"  # Already holding shares, wait
    elif prob_up < lower:
        if shares > 0:
            return "sell"
        else:
            return "hold"  # No shares to sell
    else:
        return "hold"  # No strong signal

if __name__ == "__main__":
    print("0.62 →", should_trade(0.62))  # buy if no shares
    print("0.40 →", should_trade(0.40))  # sell if have shares
    print("0.51 →", should_trade(0.51))  # hold