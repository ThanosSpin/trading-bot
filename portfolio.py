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
    with open(_portfolio_file(symbol), "w") as f:
        json.dump(portfolio, f, indent=4)

def load_portfolio(symbol):
    if os.path.exists(_portfolio_file(symbol)):
        with open(_portfolio_file(symbol), "r") as f:
            return json.load(f)
    return {"cash": 10000.0, "shares": 0.0, "last_price": 0.0}

# ----------------------------
# Portfolio value helper
# ----------------------------
def portfolio_value(portfolio: dict) -> float:
    """Calculate total portfolio value (cash + shares * last_price)."""
    return float(portfolio.get("cash", 0.0)) + float(portfolio.get("shares", 0.0)) * float(portfolio.get("last_price", 0.0))

# ----------------------------
# Trade logging
# ----------------------------
def log_trade(symbol, action, price, portfolio, quantity=0):
    """
    Append a trade row to trades_{symbol}.csv.
    Columns: timestamp, symbol, action, qty, price, cash, shares, value
    """
    log_path = _trade_log_file(symbol)
    file_exists = os.path.isfile(log_path)
    value = portfolio_value(portfolio)

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "symbol", "action", "qty", "price", "cash", "shares", "value"])
        writer.writerow([
            datetime.utcnow().isoformat(),
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
    """
    Update portfolio state after a trade, log it, and return the new portfolio.
    - quantity can be a float (partial fills).
    - We try to refresh from Alpaca live portfolio to avoid stale local state.
    """
    action = str(action).lower()
    quantity = float(quantity)
    price = float(price) if price is not None else 0.0

    # Try to baseline from live Alpaca to avoid stale/local drift
    try:
        live = get_live_portfolio(symbol)
        # Only overwrite if live returned plausible values
        if isinstance(live, dict) and "cash" in live:
            portfolio = {
                "cash": float(live["cash"]),
                "shares": float(live["shares"]),
                "last_price": float(live.get("last_price", price))
            }
    except Exception:
        # If live fetch fails, keep the passed-in portfolio
        pass

    # Apply the change
    if action == "buy":
        portfolio["shares"] = float(portfolio.get("shares", 0.0)) + quantity
        portfolio["cash"] = float(portfolio.get("cash", 0.0)) - quantity * price
        portfolio["last_price"] = price
    elif action == "sell":
        portfolio["shares"] = float(portfolio.get("shares", 0.0)) - quantity
        portfolio["cash"] = float(portfolio.get("cash", 0.0)) + quantity * price
        # last_price stays as the last buy price (we keep last_price as entry price)

    # Log, persist and return
    log_trade(symbol, action, price, portfolio, quantity=quantity)
    save_portfolio(portfolio, symbol)
    return portfolio

# ----------------------------
# Sync trades from Alpaca (rebuild CSV from filled orders)
# ----------------------------
def sync_trades_from_alpaca(symbol):
    """
    Fetch all filled orders for symbol from Alpaca and update local trade log.
    Reconstructs cash/shares as a ledger using the filled_qty & filled_avg_price.
    """
    try:
        trades = api_market.list_orders(status="filled", symbols=[symbol], limit=1000, nested=True)
        if not trades:
            return
        trades.sort(key=lambda t: t.filled_at)
        rows = []
        cash = 0.0
        shares = 0.0
        for t in trades:
            ts = t.filled_at
            if ts is None:
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=pytz.UTC)
            side = t.side.lower()
            qty = float(getattr(t, "filled_qty", 0) or 0)
            price = float(getattr(t, "filled_avg_price", 0) or 0)

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
                f"{qty:g}",
                f"{price:.2f}",
                f"{cash:.2f}",
                f"{shares:.8g}",
                f"{value:.2f}"
            ])

        with open(_trade_log_file(symbol), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "symbol", "action", "qty", "price", "cash", "shares", "value"])
            writer.writerows(rows)
    except Exception as e:
        print(f"[ERROR] Failed to sync trades from Alpaca for {symbol}: {e}")

# ----------------------------
# Daily CSV helpers
# ----------------------------
def save_daily_portfolio_csv(symbol):
    trade_path = _trade_log_file(symbol)
    if not os.path.exists(trade_path):
        return
    df = pd.read_csv(trade_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    local_tz = pytz.timezone(TIMEZONE)

    # If timestamps are naive assume UTC
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")

    df["timestamp_local"] = df["timestamp"].dt.tz_convert(local_tz)
    df.set_index("timestamp_local", inplace=True)
    daily = df[["cash", "shares", "value"]].astype(float).resample("D").last()
    daily.reset_index(inplace=True)
    daily.rename(columns={"timestamp_local": "date"}, inplace=True)
    daily.to_csv(_daily_portfolio_file(symbol), index=False)
    print(f"✅ Saved daily portfolio CSV for {symbol}")

def get_daily_portfolio_history(symbol):
    daily_path = _daily_portfolio_file(symbol)
    if not os.path.exists(daily_path):
        return pd.DataFrame(columns=["date", "value"])
    df_daily = pd.read_csv(daily_path)
    df_daily["date"] = pd.to_datetime(df_daily["date"])
    df_daily["value"] = df_daily["value"].astype(float)
    return df_daily.sort_values("date").reset_index(drop=True)