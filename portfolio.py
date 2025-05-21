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

def load_portfolio():
    if USE_LIVE_TRADING:
        try:
            api = tradeapi.REST(API_MARKET_KEY, API_MARKET_SECRET, MARKET_BASE_URL)
            account = api.get_account()
            cash = float(account.cash)

            try:
                position = api.get_position(SYMBOL)
                shares = float(position.qty)
                last_price = float(position.current_price)
            except tradeapi.rest.APIError as e:
                if "position does not exist" in str(e).lower():
                    shares = 0.0
                    last_price = 0.0
                else:
                    raise e

            return {"cash": cash, "shares": shares, "last_price": last_price}

        except Exception as e:
            print(f"[ERROR] Failed to fetch Alpaca portfolio: {e}")
            # Fallback to JSON
            if os.path.exists(PORTFOLIO_PATH):
                with open(PORTFOLIO_PATH, 'r') as f:
                    return json.load(f)
            return {"cash": INITIAL_CAPITAL, "shares": 0.0, "last_price": 0.0}

    # SIMULATION / OFFLINE
    if os.path.exists(PORTFOLIO_PATH):
        with open(PORTFOLIO_PATH, 'r') as f:
            return json.load(f)
    return {"cash": INITIAL_CAPITAL, "shares": 0.0, "last_price": 0.0}

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
    if action == "buy" and portfolio["cash"] >= price:
        portfolio["shares"] += 1
        portfolio["cash"] -= price
    elif action == "sell" and portfolio["shares"] > 0:
        portfolio["shares"] -= 1
        portfolio["cash"] += price
    elif action == "sell" and portfolio["shares"] == 0:
        print("[WARN] Tried to sell with 0 shares — skipping.")
    portfolio["last_price"] = price
    log_trade(action, price, portfolio)
    return portfolio

def plot_portfolio_performance():
    if not os.path.exists(TRADE_LOG_PATH):
        print("No trade log found to plot performance.")
        return

    timestamps = []
    values = []

    with open(TRADE_LOG_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            time = datetime.fromisoformat(row["timestamp"])
            value = float(row["value"])
            timestamps.append(time)
            values.append(value)

    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, values, marker='o')
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Time")
    plt.ylabel("Value (€)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("data/portfolio_performance.png")
    plt.close()

if __name__ == "__main__":
    TEST = False
    if TEST:
        p = load_portfolio()
        print("Initial:", p)
        p = update_portfolio("buy", 100, p)
        print("After buy:", p)
        print("Total value:", portfolio_value(p))
    plot_portfolio_performance()