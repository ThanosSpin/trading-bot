# portfolio.py
import os
import json
import csv
import pandas as pd
from datetime import datetime
import pytz
from config import PORTFOLIO_PATH, TIMEZONE
from broker import api_market

# ----------------------------
# File helpers
# ----------------------------
def get_portfolio_file(symbol):
    os.makedirs(os.path.dirname(PORTFOLIO_PATH), exist_ok=True)
    return os.path.join(os.path.dirname(PORTFOLIO_PATH), f"portfolio_{symbol}.json")

def get_trade_log_file(symbol):
    os.makedirs(os.path.dirname(PORTFOLIO_PATH), exist_ok=True)
    return os.path.join(os.path.dirname(PORTFOLIO_PATH), f"trades_{symbol}.csv")

def get_daily_portfolio_file():
    os.makedirs(os.path.dirname(PORTFOLIO_PATH), exist_ok=True)
    return os.path.join(os.path.dirname(PORTFOLIO_PATH), f"daily_portfolio.csv")

# ----------------------------
# Portfolio state
# ----------------------------
def get_live_portfolio(symbol):
    """Fetch live portfolio state from Alpaca account for a given symbol."""
    account = api_market.get_account()
    positions = api_market.list_positions()
    cash = float(account.cash)
    shares = 0.0
    last_price = 0.0
    for p in positions:
        if p.symbol == symbol:
            shares = float(p.qty)
            last_price = float(p.current_price)
    return {"cash": cash, "shares": shares, "last_price": last_price}

def save_portfolio(portfolio, symbol):
    with open(get_portfolio_file(symbol), "w") as f:
        json.dump(portfolio, f, indent=4)

def load_portfolio(symbol):
    if os.path.exists(get_portfolio_file(symbol)):
        with open(get_portfolio_file(symbol), "r") as f:
            return json.load(f)
    live = get_live_portfolio(symbol)
    return {"cash": live["cash"], "shares": 0.0, "last_price": 0.0}

# ----------------------------
# Portfolio value helper
# ----------------------------
def portfolio_value(portfolio: dict) -> float:
    return float(portfolio.get("cash", 0.0)) + float(portfolio.get("shares", 0.0)) * float(portfolio.get("last_price", 0.0))

# ----------------------------
# Trade logging
# ----------------------------
def log_trade(symbol, action, price, portfolio, quantity=0, timestamp=None):
    log_path = get_trade_log_file(symbol)
    file_exists = os.path.isfile(log_path)
    value = portfolio_value(portfolio)
    timestamp = timestamp or datetime.utcnow().isoformat()

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "symbol", "action", "qty", "price", "cash", "shares", "value"])
        writer.writerow([
            timestamp,
            symbol,
            action,
            f"{float(quantity):g}",
            f"{float(price):.2f}" if price is not None else "N/A",
            f"{float(portfolio.get('cash',0.0)):.2f}",
            f"{float(portfolio.get('shares',0.0)):.8g}",
            f"{float(value):.2f}",
        ])

# ----------------------------
# Update portfolio after a trade
# ----------------------------
def update_portfolio(action, price, portfolio, symbol, quantity=1.0):
    action = str(action).lower()
    quantity = float(quantity)
    price = float(price) if price is not None else 0.0

    if action == "buy":
        portfolio["shares"] += quantity
        portfolio["cash"] -= quantity * price
        portfolio["last_price"] = price
    elif action == "sell":
        portfolio["shares"] -= quantity
        portfolio["cash"] += quantity * price
        portfolio["last_price"] = price

    log_trade(symbol, action, price, portfolio, quantity)
    save_portfolio(portfolio, symbol)
    return portfolio

# ----------------------------
# Daily portfolio helpers
# ----------------------------
def save_daily_portfolio_csv(trades_list, initial_cash=1000.0):
    """
    trades_list: list of all trades from all symbols in chronological order
    """
    tz = pytz.timezone(TIMEZONE)
    cash = initial_cash
    portfolio_state = {}
    daily_values = {}

    for t in trades_list:
        sym = t["symbol"]
        action = t["action"]
        qty = t["qty"]
        price = t["price"]
        timestamp = t["timestamp"]

        if sym not in portfolio_state:
            portfolio_state[sym] = {"shares": 0.0, "last_price": 0.0}

        if action == "buy":
            portfolio_state[sym]["shares"] += qty
            cash -= qty * price
        elif action == "sell":
            portfolio_state[sym]["shares"] -= qty
            cash += qty * price

        portfolio_state[sym]["last_price"] = price

        # Calculate total portfolio value
        total_value = cash + sum(p["shares"] * p["last_price"] for p in portfolio_state.values())
        daily_values[timestamp.date()] = total_value

    df_daily = pd.DataFrame([{"date": pd.Timestamp(d).tz_localize(tz), "value": v} for d, v in daily_values.items()])
    df_daily.sort_values("date", inplace=True)
    df_daily.to_csv(get_daily_portfolio_file(), index=False)
    print(f"✅ Daily portfolio CSV saved: {get_daily_portfolio_file()}")
    return df_daily