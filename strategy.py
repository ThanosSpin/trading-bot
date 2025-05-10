from config import THRESHOLD

def should_trade(prob_up):
    if prob_up > 0.5 + THRESHOLD:
        return 'buy'
    elif prob_up < 0.5 - THRESHOLD:
        return 'sell'
    else:
        return 'hold'
