# portfolio.py
import json
import os
import csv
import pytz
from datetime import datetime
import matplotlib.pyplot as plt
import alpaca_trade_api as tradeapi

from config import (
    SYMBOL, INITIAL_CAPITAL, PORTFOLIO_PATH, TIMEZONE,
    USE_LIVE_TRADING, API_MARKET_KEY, API_MARKET_SECRET, MARKET_BASE_URL
)

TRADE_LOG_PATH = "data/trade_log.csv"

# Alpaca API setup
api = tradeapi.REST(API_MARKET_KEY, API_MARKET_SECRET, MARKET_BASE_URL)


def get_live_portfolio(symbol: str):
    """Fetch live portfolio state for a specific symbol from Alpaca account"""
    account = api.get_account()
    positions = api.list_positions()

    cash = float(account.cash)
    shares = 0.0
    last_price = 0.0

    for position in positions:
        if position.symbol.upper() == symbol.upper():
            shares = float(position.qty)
            last_price = float(position.current_price)

    return {
        "symbol": symbol,
        "cash": cash,
        "shares": shares,
        "last_price": last_price
    }


def load_portfolio(symbol: str):
    """Load portfolio for one symbol"""
    if USE_LIVE_TRADING:
        return get_live_portfolio(symbol)

    # simulated portfolio (JSON file)
    if os.path.exists(PORTFOLIO_PATH):
        with open(PORTFOLIO_PATH, "r") as f:
            all_portfolios = json.load(f)
        return all_portfolios.get(symbol, {"cash": INITIAL_CAPITAL, "shares": 0, "last_price": 0.0})

    return {"cash": INITIAL_CAPITAL, "shares": 0, "last_price": 0.0}


def save_portfolio(symbol: str, portfolio: dict):
    """Save portfolio state (per-symbol)"""
    os.makedirs(os.path.dirname(PORTFOLIO_PATH), exist_ok=True)
    all_portfolios = {}
    if os.path.exists(PORTFOLIO_PATH):
        with open(PORTFOLIO_PATH, "r") as f:
            all_portfolios = json.load(f)

    all_portfolios[symbol] = portfolio
    with open(PORTFOLIO_PATH, "w") as f:
        json.dump(all_portfolios, f, indent=2)


def portfolio_value(portfolio: dict) -> float:
    return portfolio["cash"] + portfolio["shares"] * portfolio["last_price"]


def log_trade(symbol: str, action: str, price: float, portfolio: dict):
    """Log trade to CSV and update performance plot"""
    os.makedirs(os.path.dirname(TRADE_LOG_PATH), exist_ok=True)
    file_exists = os.path.isfile(TRADE_LOG_PATH)
    value = portfolio_value(portfolio)

    with open(TRADE_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "symbol", "action", "price", "value"])
        writer.writerow([datetime.utcnow().isoformat(), symbol, action, f"{price:.2f}", f"{value:.2f}"])

    plot_portfolio_performance()


def update_portfolio(symbol: str, action: str, price: float, portfolio: dict):
    """Update portfolio after a trade"""
    if USE_LIVE_TRADING:
        portfolio = get_live_portfolio(symbol)
    log_trade(symbol, action, price, portfolio)
    return portfolio


def plot_portfolio_performance():
    """Generate a chart of portfolio value over time"""
    if not os.path.exists(TRADE_LOG_PATH):
        print("No trade log found to plot performance.")
        return

    timestamps = []
    values = []
    local_tz = pytz.timezone(TIMEZONE)

    with open(TRADE_LOG_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            utc_time = datetime.fromisoformat(row["timestamp"])
            local_time = utc_time.replace(tzinfo=pytz.utc).astimezone(local_tz)
            timestamps.append(local_time)
            values.append(float(row["value"]))

    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, values, marker="o")
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
        for sym in SYMBOL:
            p = load_portfolio(sym)
            print(f"Portfolio snapshot for {sym}:", p)
            print(f"Portfolio Value: ${portfolio_value(p):.2f}")
    plot_portfolio_performance()