# strategy.py
from config import (
    BUY_THRESHOLD,
    SELL_THRESHOLD,
    STOP_LOSS,
    TAKE_PROFIT,
    RISK_FRACTION,
)
from portfolio import load_portfolio, portfolio_value
from data_loader import fetch_latest_price


def should_trade(symbol, prob_up, total_symbols=1):
    """
    Decide trade action and quantity based on model probability, portfolio, and risk management.
    - Requires higher confidence for buy/sell.
    - Includes stop-loss and take-profit rules.
    - Allocation:
        - If one symbol: trade with all available cash.
        - If multiple symbols: max allocation = RISK_FRACTION of portfolio per symbol.
    """
    portfolio = load_portfolio(symbol)
    shares = portfolio.get("shares", 0)
    cash = portfolio.get("cash", 0)
    last_price = portfolio.get("last_price", 0)  # stored in portfolio.json
    price = fetch_latest_price(symbol)

    if price is None or price <= 0:
        return ("hold", 0)

    value = portfolio_value(portfolio)

    # Allocation rule
    if total_symbols <= 1:
        max_invest = cash  # one symbol → use all cash
    else:
        max_invest = (value * RISK_FRACTION) / total_symbols

    # --- Risk management overrides ---
    if shares > 0 and last_price > 0:
        if price <= last_price * STOP_LOSS:
            print(f"[DEBUG] Stop-loss triggered for {symbol} at {price:.2f} (last price {last_price:.2f})")
            return ("sell", shares)  # sell everything
        elif price >= last_price * TAKE_PROFIT:
            print(f"[DEBUG] Take-profit triggered for {symbol} at {price:.2f} (last price {last_price:.2f})")
            return ("sell", shares)  # sell everything

    # --- Confidence-based trading ---
    if prob_up >= BUY_THRESHOLD:
        affordable_shares = int(cash // price)
        target_shares = int(max_invest // price)
        quantity = min(
            affordable_shares,
            target_shares if total_symbols > 1 else affordable_shares,
        )
        if quantity > 0:
            print(f"[DEBUG] Confidence BUY for {symbol}: prob_up={prob_up:.2f}, quantity={quantity}")
            return ("buy", quantity)

    elif prob_up <= SELL_THRESHOLD and shares > 0:
        if total_symbols <= 1:
            # Sell everything if trading one symbol
            print(f"[DEBUG] Confidence SELL (full) for {symbol}: prob_up={prob_up:.2f}, shares={shares}")
            return ("sell", shares)
        else:
            # Sell a fraction if managing multiple symbols
            sell_shares = max(1, int(shares * RISK_FRACTION))
            print(f"[DEBUG] Confidence SELL (partial) for {symbol}: prob_up={prob_up:.2f}, shares={sell_shares}")
            return ("sell", min(sell_shares, shares))

    return ("hold", 0)
