import os
import json
import csv
import pandas as pd
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
from config import PORTFOLIO_PATH, TIMEZONE
from broker import api

# ----------------------------
# Portfolio file helpers
# ----------------------------
def _portfolio_file(symbol):
    folder = os.path.dirname(PORTFOLIO_PATH)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"portfolio_{symbol}.json")

def _trade_log_file(symbol):
    folder = os.path.dirname(PORTFOLIO_PATH)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"trades_{symbol}.csv")

def _performance_chart_file(symbol):
    folder = os.path.dirname(PORTFOLIO_PATH)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"portfolio_{symbol}_performance.png")


# ----------------------------
# Portfolio state
# ----------------------------
def get_live_portfolio(symbol):
    """Fetch live portfolio state from Alpaca account for given symbol."""
    account = api.get_account()
    positions = api.list_positions()

    cash = float(account.cash)
    shares = 0.0
    last_price = 0.0

    for position in positions:
        if position.symbol == symbol:
            shares = float(position.qty)
            last_price = float(position.current_price)

    return {"cash": cash, "shares": shares, "last_price": last_price}


def save_portfolio(portfolio, symbol):
    file_path = _portfolio_file(symbol)
    with open(file_path, "w") as f:
        json.dump(portfolio, f, indent=4)
    print(f"💾 Portfolio saved: {file_path}")


def load_portfolio(symbol):
    file_path = _portfolio_file(symbol)
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return {"cash": 10000, "shares": 0, "last_price": 0.0}


def portfolio_value(portfolio):
    return float(portfolio["cash"]) + float(portfolio["shares"]) * float(portfolio["last_price"])


# ----------------------------
# Trade logging
# ----------------------------
def log_trade(symbol, action, price, portfolio):
    log_path = _trade_log_file(symbol)
    file_exists = os.path.isfile(log_path)
    value = portfolio_value(portfolio)

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "symbol", "action", "price", "value"])

        writer.writerow([
            datetime.utcnow().isoformat(),
            symbol,
            action,
            f"{float(price):.2f}" if price is not None else "N/A",
            f"{float(value):.2f}" if value is not None else "N/A",
        ])

    plot_portfolio_performance(symbol)


def update_portfolio(action, price, portfolio, symbol):
    """
    Update portfolio and log trades.
    - Refresh cash and shares from Alpaca.
    - Keep last_price as the entry price (only updated on BUY).
    """
    try:
        # Refresh cash and shares from Alpaca
        portfolio_live = get_live_portfolio(symbol)
        portfolio["cash"] = portfolio_live.get("cash", portfolio["cash"])
        portfolio["shares"] = portfolio_live.get("shares", portfolio["shares"])
    except Exception as e:
        print(f"[DEBUG] Could not fetch live portfolio for {symbol}: {e}")
        print(f"[DEBUG] Using local portfolio: {portfolio}")

    # ✅ Only update entry price if it was a BUY
    if action == "buy" and price is not None:
        portfolio["last_price"] = price

    # Log trade and save portfolio
    log_trade(symbol, action, price, portfolio)
    save_portfolio(portfolio, symbol)
    return portfolio


# ----------------------------
# Performance plotting
# ----------------------------
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

def plot_portfolio_performance(symbol):
    log_path = _trade_log_file(symbol)
    if not os.path.exists(log_path):
        print(f"No trade log found for {symbol} to plot performance.")
        return

    df = pd.read_csv(log_path, parse_dates=["timestamp"])
    local_tz = pytz.timezone(TIMEZONE)
    df["timestamp"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert(local_tz)
    df["timestamp_str"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")

    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.plot(df["timestamp_str"], df["value"], marker="o")
    ax.set_title(f"Portfolio Value Over Time ({symbol})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value ($)")
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.2f}'))  # Dollar formatting
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    chart_path = _performance_chart_file(symbol)
    plt.savefig(chart_path)
    plt.close()
    print(f"📊 Saved portfolio performance chart: {chart_path}")


# ----------------------------
# Sync trades from Alpacca
# ----------------------------    
def sync_trades_from_alpaca(symbol):
    """
    Fetch trade history for the given symbol from Alpaca
    and merge it into the local trades CSV log.
    Ensures portfolio 'value' reflects actual Alpaca equity.
    """
    log_path = _trade_log_file(symbol)
    existing = pd.DataFrame()

    if os.path.exists(log_path):
        existing = pd.read_csv(log_path, parse_dates=["timestamp"])

    # Fetch order history from Alpaca
    orders = api.list_orders(status="all", symbols=[symbol], limit=1000)
    rows = []
    for order in orders:
        if order.filled_at is None:
            continue
        filled_time = pd.to_datetime(order.filled_at)

        # Get current account equity at this moment
        try:
            account = api.get_account()
            equity = float(account.equity)
        except Exception:
            equity = None

        rows.append({
            "timestamp": filled_time,
            "symbol": order.symbol,
            "action": order.side,
            "price": float(order.filled_avg_price or 0),
            "value": equity,
        })

    new_df = pd.DataFrame(rows)

    # Merge with existing
    if not new_df.empty:
        if not existing.empty:
            combined = pd.concat([existing, new_df])
            combined = combined.drop_duplicates(subset=["timestamp", "action"]).sort_values("timestamp")
        else:
            combined = new_df
        combined.to_csv(log_path, index=False)
        print(f"✅ Synced {len(combined)} trades for {symbol} into {log_path}")
    else:
        print(f"ℹ️ No new trades fetched for {symbol}")