# portfolio.py
import json
import os
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from config import SYMBOL, INITIAL_CAPITAL, PORTFOLIO_PATH, \
    USE_LIVE_TRADING, API_MARKET_KEY, API_MARKET_SECRET, MARKET_BASE_URL
import alpaca_trade_api as tradeapi

TRADE_LOG_PATH = "data/trade_log.csv"

# Alpaca API setup
api = tradeapi.REST(API_MARKET_KEY, API_MARKET_SECRET, MARKET_BASE_URL)

def get_live_portfolio():
    """Fetch live portfolio state from Alpaca account"""
    account = api.get_account()
    positions = api.list_positions()

    cash = float(account.cash)
    shares = 0.0
    last_price = 0.0

    for position in positions:
        if position.symbol == SYMBOL:
            shares = float(position.qty)
            last_price = float(position.current_price)

    return {
        "cash": cash,
        "shares": shares,
        "last_price": last_price
    }

def load_portfolio():
    return get_live_portfolio()

def save_portfolio(portfolio):
    os.makedirs(os.path.dirname(PORTFOLIO_PATH), exist_ok=True)
    with open(PORTFOLIO_PATH, 'w') as f:
        json.dump(portfolio, f, indent=2)

def portfolio_value(portfolio):
    return portfolio["cash"] + portfolio["shares"] * portfolio["last_price"]

def log_trade(action, price, portfolio):
    os.makedirs(os.path.dirname(TRADE_LOG_PATH), exist_ok=True)
    file_exists = os.path.isfile(TRADE_LOG_PATH)
    value = portfolio_value(portfolio)

    with open(TRADE_LOG_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "action", "price", "value"])
        writer.writerow([datetime.utcnow().isoformat(), action, f"{price:.2f}", f"{value:.2f}"])

    plot_portfolio_performance()

def update_portfolio(action, price, portfolio):
    # For live mode, we won't simulate changes, but we still log
    portfolio = get_live_portfolio()
    log_trade(action, price, portfolio)
    return portfolio

import pytz
from config import TIMEZONE

def plot_portfolio_performance():
    if not os.path.exists(TRADE_LOG_PATH):
        print("No trade log found to plot performance.")
        return

    timestamps = []
    values = []

    local_tz = pytz.timezone(TIMEZONE)

    with open(TRADE_LOG_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert UTC timestamp to local timezone
            utc_time = datetime.fromisoformat(row["timestamp"])
            local_time = utc_time.replace(tzinfo=pytz.utc).astimezone(local_tz)
            timestamps.append(local_time)
            values.append(float(row["value"]))

    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, values, marker='o')
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Time")
    plt.ylabel("Value ($)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("data/portfolio_performance.png")
    plt.close()

if __name__ == "__main__":
    TEST = True
    if TEST:
        p = load_portfolio()
        print("Portfolio snapshot:", p)
        print("Portfolio Value: $", portfolio_value(p))
    plot_portfolio_performance()