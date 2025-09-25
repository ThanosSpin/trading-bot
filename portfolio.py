import os
import json
import csv
import pandas as pd
from datetime import datetime
import pytz
from config import PORTFOLIO_PATH, TIMEZONE
from broker import api

# ----------------------------
# File helpers
# ----------------------------
def _portfolio_file(symbol):
    os.makedirs(os.path.dirname(PORTFOLIO_PATH), exist_ok=True)
    return os.path.join(os.path.dirname(PORTFOLIO_PATH), f"portfolio_{symbol}.json")

def _trade_log_file(symbol):
    os.makedirs(os.path.dirname(PORTFOLIO_PATH), exist_ok=True)
    return os.path.join(os.path.dirname(PORTFOLIO_PATH), f"trades_{symbol}.csv")

def _daily_portfolio_file(symbol):
    os.makedirs(os.path.dirname(PORTFOLIO_PATH), exist_ok=True)
    return os.path.join(os.path.dirname(PORTFOLIO_PATH), f"daily_portfolio_{symbol}.csv")

# ----------------------------
# Portfolio state
# ----------------------------
def get_live_portfolio(symbol):
    account = api.get_account()
    positions = api.list_positions()
    cash = float(account.cash)
    shares = 0.0
    last_price = 0.0
    for p in positions:
        if p.symbol == symbol:
            shares = float(p.qty)
            last_price = float(p.current_price)
    return {"cash": cash, "shares": shares, "last_price": last_price}

def save_portfolio(portfolio, symbol):
    with open(_portfolio_file(symbol), "w") as f:
        json.dump(portfolio, f, indent=4)

def load_portfolio(symbol):
    if os.path.exists(_portfolio_file(symbol)):
        with open(_portfolio_file(symbol), "r") as f:
            return json.load(f)
    return {"cash": 10000, "shares": 0, "last_price": 0.0}

# ----------------------------
# Trade logging
# ----------------------------
def log_trade(symbol, action, price, portfolio):
    log_path = _trade_log_file(symbol)
    file_exists = os.path.isfile(log_path)
    value = portfolio["cash"] + portfolio["shares"] * portfolio["last_price"]
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "symbol", "action", "price", "cash", "shares", "value"])
        writer.writerow([
            datetime.utcnow().isoformat(),
            symbol,
            action,
            f"{price:.2f}" if price else "N/A",
            f"{portfolio['cash']:.2f}",
            f"{portfolio['shares']:.2f}",
            f"{value:.2f}",
        ])

# ----------------------------
# Sync trades from Alpaca
# ----------------------------
def sync_trades_from_alpaca(symbol):
    trades = api.list_orders(status="filled", symbols=[symbol], limit=500, nested=True)
    if not trades:
        return
    trades.sort(key=lambda t: t.filled_at)
    rows = []
    cash = 0.0
    shares = 0.0
    for t in trades:
        ts = t.filled_at
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=pytz.UTC)
        side = t.side.lower()
        qty = float(t.filled_qty)
        price = float(t.filled_avg_price)
        if side == "buy":
            shares += qty
            cash -= qty * price
        elif side == "sell":
            shares -= qty
            cash += qty * price
        value = cash + shares * price
        rows.append([
            ts.isoformat(),
            symbol,
            side,
            f"{price:.2f}",
            f"{cash:.2f}",
            f"{shares:.2f}",
            f"{value:.2f}"
        ])
    with open(_trade_log_file(symbol), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "symbol", "action", "price", "cash", "shares", "value"])
        writer.writerows(rows)

# ----------------------------
# Save daily portfolio CSV
# ----------------------------
def save_daily_portfolio_csv(symbol):
    trade_path = _trade_log_file(symbol)
    if not os.path.exists(trade_path):
        return
    df = pd.read_csv(trade_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    local_tz = pytz.timezone(TIMEZONE)
    df["timestamp_local"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert(local_tz)
    df.set_index("timestamp_local", inplace=True)
    daily = df[["cash", "shares", "value"]].astype(float).resample("D").last()
    daily.reset_index(inplace=True)
    daily.rename(columns={"timestamp_local": "date"}, inplace=True)
    daily.to_csv(_daily_portfolio_file(symbol), index=False)
    print(f"✅ Saved daily portfolio CSV for {symbol}")

# ----------------------------
# Load daily portfolio history
# ----------------------------
def get_daily_portfolio_history(symbol):
    daily_path = _daily_portfolio_file(symbol)
    if not os.path.exists(daily_path):
        return pd.DataFrame(columns=["date", "value"])
    df_daily = pd.read_csv(daily_path)
    df_daily["date"] = pd.to_datetime(df_daily["date"])
    df_daily["value"] = df_daily["value"].astype(float)
    return df_daily.sort_values("date").reset_index(drop=True)

# ----------------------------
# Portfolio value helper
# ----------------------------
def portfolio_value(portfolio: dict) -> float:
    """
    Calculate total portfolio value (cash + shares * last_price).
    Expects a dict with keys: 'cash', 'shares', 'last_price'.
    """
    return float(portfolio.get("cash", 0.0)) + float(portfolio.get("shares", 0.0)) * float(portfolio.get("last_price", 0.0))


# ----------------------------
# Update portfolio after a trade
# ----------------------------
def update_portfolio(action, price, portfolio, symbol):
    """
    Update the portfolio state after a trade, log it, and return the new portfolio.
    This keeps backward compatibility with main.py.
    """
    action = action.lower()
    price = float(price)

    if action == "buy":
        portfolio["shares"] += 1  # assumes quantity=1 per trade
        portfolio["cash"] -= price
    elif action == "sell":
        portfolio["shares"] -= 1
        portfolio["cash"] += price

    # update last price
    portfolio["last_price"] = price

    # log the trade
    log_trade(symbol, action, price, portfolio)

    # persist portfolio JSON
    save_portfolio(portfolio, symbol)

    return portfolio