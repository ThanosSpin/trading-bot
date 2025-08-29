# portfolio.py

import json
import os
import csv
import pytz
from datetime import datetime
import matplotlib.pyplot as plt
from config import TIMEZONE, INITIAL_CAPITAL, USE_LIVE_TRADING, API_MARKET_KEY, API_MARKET_SECRET, MARKET_BASE_URL
import alpaca_trade_api as tradeapi

api = tradeapi.REST(API_MARKET_KEY, API_MARKET_SECRET, MARKET_BASE_URL)

def get_portfolio_path(symbol):
    return f"data/portfolio_{symbol}.json"

def get_trade_log_path(symbol):
    return f"data/trade_log_{symbol}.csv"

def get_live_portfolio(symbol):
    account = api.get_account()
    positions = api.list_positions()

    cash = float(account.cash)
    shares = 0.0
    last_price = 0.0

    for position in positions:
        if position.symbol == symbol:
            shares = float(position.qty)
            last_price = float(position.current_price)

    return {
        "cash": cash,
        "shares": shares,
        "last_price": last_price
    }

def load_portfolio(symbol):
    if USE_LIVE_TRADING:
        return get_live_portfolio(symbol)

    path = get_portfolio_path(symbol)
    if not os.path.exists(path):
        return {"cash": INITIAL_CAPITAL, "shares": 0, "last_price": 0}

    with open(path, 'r') as f:
        return json.load(f)

def save_portfolio(portfolio, symbol):
    path = get_portfolio_path(symbol)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(portfolio, f, indent=2)

def portfolio_value(portfolio):
    return portfolio["cash"] + portfolio["shares"] * portfolio["last_price"]

def log_trade(action, price, portfolio, symbol):
    path = get_trade_log_path(symbol)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.isfile(path)
    value = portfolio_value(portfolio)

    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "action", "price", "value"])
        writer.writerow([datetime.utcnow().isoformat(), action, f"{price:.2f}", f"{value:.2f}"])

    plot_portfolio_performance(symbol)

def update_portfolio(action, price, portfolio, symbol):
    if USE_LIVE_TRADING:
        portfolio = get_live_portfolio(symbol)

    log_trade(action, price, portfolio, symbol)
    return portfolio

def plot_portfolio_performance(symbol):
    path = get_trade_log_path(symbol)
    if not os.path.exists(path):
        print(f"No trade log found to plot for {symbol}.")
        return

    timestamps = []
    values = []
    local_tz = pytz.timezone(TIMEZONE)

    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            utc_time = datetime.fromisoformat(row["timestamp"])
            local_time = utc_time.replace(tzinfo=pytz.utc).astimezone(local_tz)
            timestamps.append(local_time)
            values.append(float(row["value"]))

    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, values, marker='o')
    plt.title(f"{symbol} Portfolio Value Over Time")
    plt.xlabel("Time")
    plt.ylabel("Value ($)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"data/portfolio_performance_{symbol}.png")
    plt.close()