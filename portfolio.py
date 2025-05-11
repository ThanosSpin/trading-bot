# portfolio.py
import json
import os
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from config import PORTFOLIO_PATH, INITIAL_CAPITAL

TRADE_LOG_PATH = "data/trade_log.csv"

def load_portfolio():
    if os.path.exists(PORTFOLIO_PATH):
        with open(PORTFOLIO_PATH, 'r') as f:
            return json.load(f)
    return {"cash": 0.0, "shares": 8.7, "last_price": 0.0}

def save_portfolio(portfolio):
    os.makedirs(os.path.dirname(PORTFOLIO_PATH), exist_ok=True)
    with open(PORTFOLIO_PATH, 'w') as f:
        json.dump(portfolio, f, indent=2)

def log_trade(action, price):
    os.makedirs(os.path.dirname(TRADE_LOG_PATH), exist_ok=True)
    file_exists = os.path.isfile(TRADE_LOG_PATH)
    with open(TRADE_LOG_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "action", "price"])
        writer.writerow([datetime.utcnow().isoformat(), action, f"{price:.2f}"])
    plot_portfolio_performance()  # Auto-update plot after each trade

def update_portfolio(action, price, portfolio):
    if action == "buy" and portfolio["cash"] >= price:
        portfolio["shares"] += 1
        portfolio["cash"] -= price
        log_trade("buy", price)
    elif action == "sell" and portfolio["shares"] > 0:
        portfolio["shares"] -= 1
        portfolio["cash"] += price
        log_trade("sell", price)
    elif action == "sell" and portfolio["shares"] == 0:
        print("[WARN] Tried to sell with 0 shares — skipping.")
    portfolio["last_price"] = price
    return portfolio

def portfolio_value(portfolio):
    return portfolio["cash"] + portfolio["shares"] * portfolio["last_price"]

def plot_portfolio_performance():
    if not os.path.exists(TRADE_LOG_PATH):
        print("No trade log found to plot performance.")
        return

    timestamps = []
    values = []
    cash = 0.0
    shares = 8.7
    last_price = 0.0

    with open(TRADE_LOG_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            time = datetime.fromisoformat(row["timestamp"])
            action = row["action"]
            price = float(row["price"])
            if action == "buy":
                cash -= price
                shares += 1
            elif action == "sell":
                cash += price
                shares -= 1
            last_price = price
            timestamps.append(time)
            values.append(cash + shares * last_price)

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
    TEST = False  # Change to True only for testing
    if TEST:
        p = load_portfolio()
        print("Initial:", p)
        p = update_portfolio("buy", 100, p)
        print("After buy:", p)
        print("Total value:", portfolio_value(p))
    plot_portfolio_performance()
