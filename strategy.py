# decision_logic.py
from config import THRESHOLD

def should_trade(prob_up: float) -> str:
    """
    Decide trade action based on model probability.
    """
    if prob_up > 0.5 + THRESHOLD:
        return "buy"
    elif prob_up < 0.5 - THRESHOLD:
        return "sell"
    else:
        return "hold"

if __name__ == "__main__":
    print(should_trade(0.62))  # buy
    print(should_trade(0.48))  # hold
    print(should_trade(0.40))  # sell

