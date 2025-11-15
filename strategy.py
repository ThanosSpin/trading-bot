from config import BUY_THRESHOLD, SELL_THRESHOLD, STOP_LOSS, TAKE_PROFIT, RISK_FRACTION
from portfolio import load_portfolio, portfolio_value, get_live_portfolio
from data_loader import fetch_latest_price

def should_trade(symbol, prob_up, total_symbols=1, concurrent_buys=1, available_cash=None):
    """
    Decide trade action for a symbol, considering live cash available.
    
    Parameters:
        symbol (str): stock symbol
        prob_up (float): predicted probability stock will go up
        total_symbols (int): total symbols in watchlist
        concurrent_buys (int): number of symbols with buy signal
        available_cash (float, optional): cash to use for this trade
    Returns:
        action (str): 'buy', 'sell', or 'hold'
        quantity (int): number of shares
    """

    # Get live portfolio
    try:
        portfolio = get_live_portfolio(symbol)
    except Exception:
        portfolio = load_portfolio(symbol)

    shares = float(portfolio.get("shares", 0.0))
    cash = float(portfolio.get("cash", 0.0))
    last_price = float(portfolio.get("last_price", 0.0))
    price = fetch_latest_price(symbol)
    if price is None or price <= 0:
        print(f"[WARN] {symbol} invalid price: {price}")
        return ("hold", 0)

    # Use provided available cash if given
    if available_cash is not None:
        cash = available_cash

    # Compute maximum investable capital
    value = portfolio_value(portfolio)
    if total_symbols == 1:
        max_invest = cash
    elif total_symbols == 2:
        if concurrent_buys == 2:
            max_invest = cash * RISK_FRACTION  # split capital proportionally
        else:
            max_invest = cash
    else:
        max_invest = cash * RISK_FRACTION

    # --------------------------
    # Stop-loss / Take-profit
    # --------------------------
    if shares > 0 and last_price > 0:
        if price <= last_price * STOP_LOSS or price >= last_price * TAKE_PROFIT:
            return ("sell", int(shares))

    # --------------------------
    # BUY logic
    # --------------------------
    if prob_up >= BUY_THRESHOLD:
        affordable = int(cash // price)
        target = int(max_invest // price)
        quantity = min(affordable, target)
        if quantity > 0:
            return ("buy", quantity)
        return ("hold", 0)

    # --------------------------
    # SELL logic
    # --------------------------
    if prob_up <= SELL_THRESHOLD and shares > 0:
        return ("sell", int(shares))

    return ("hold", 0)