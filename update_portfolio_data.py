import os
import csv
import pandas as pd
import pytz
from datetime import timedelta
from broker import api_market
from portfolio import _trade_log_file, PORTFOLIO_PATH
from config import SYMBOL, TIMEZONE  # Read symbols and timezone from config
import time

# -----------------------------
# Helper for fetching bars
# -----------------------------
def safe_get_bars(symbol, timeframe, start, end, max_retries=3):
    """
    Fetch bars with retry. Returns empty list if all retries fail.
    """
    for attempt in range(max_retries):
        try:
            bars = api_market.get_bars(symbol, timeframe, start=start, end=end)
            if bars:
                return bars
        except Exception as e:
            pass
        time.sleep(1)  # small delay between retries
    return []

# -----------------------------
# Portfolio reconstruction
# -----------------------------
def fetch_filled_trades(symbol):
    """Fetch all filled trades for a symbol from Alpaca and sort by execution time"""
    trades = api_market.list_orders(status="filled", symbols=[symbol], limit=1000, nested=True)
    trades.sort(key=lambda t: t.filled_at)
    return trades

def reconstruct_portfolio(symbol):
    """Rebuild trade log CSV with accurate cash, shares, and portfolio value for each trade"""
    log_path = _trade_log_file(symbol)
    trades = fetch_filled_trades(symbol)
    rows = []

    cash = 1000.0  # starting cash
    shares = 0.0
    tz = pytz.timezone(TIMEZONE)

    for t in trades:
        ts = t.filled_at
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=pytz.UTC)
        ts_local = ts.astimezone(tz)

        qty = float(t.filled_qty)
        price = float(t.filled_avg_price)
        action = t.side.lower()

        if action == "buy":
            cash -= qty * price
            shares += qty
        elif action == "sell":
            cash += qty * price
            shares -= qty

        # Portfolio value using market price at trade time (fallback to trade price)
        bar = safe_get_bars(symbol, "1Min", start=ts.isoformat(), end=(ts + timedelta(minutes=1)).isoformat())
        last_price = float(bar[-1].c) if bar else price

        value = cash + shares * last_price

        rows.append([
            ts.isoformat(),
            symbol,
            action,
            f"{price:.2f}",
            f"{cash:.2f}",
            f"{shares:.2f}",
            f"{value:.2f}",
            ts_local.isoformat(),
            ts_local.strftime("%Y-%m-%d %H:%M")
        ])

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "symbol", "action", "price", "cash", "shares", "value", "timestamp_local", "timestamp_str"])
        writer.writerows(rows)

    print(f"[INFO] Trade log saved for {symbol}: {log_path}")
    return rows

# -----------------------------
# Daily portfolio generation
# -----------------------------
def generate_daily_portfolio(symbol):
    """Generate daily portfolio values using last trade of each day and market close price"""
    log_path = _trade_log_file(symbol)
    if not os.path.exists(log_path):
        print(f"[WARN] No trade log for {symbol}")
        return pd.DataFrame()

    df = pd.read_csv(log_path)
    df["timestamp_local"] = pd.to_datetime(df["timestamp_local"])
    df = df.sort_values("timestamp_local")

    daily_values = {}
    tz = pytz.timezone(TIMEZONE)

    for _, row in df.iterrows():
        day = row["timestamp_local"].date()
        cash = float(row["cash"])
        shares = float(row["shares"])
        price = float(row["price"])

        # Fetch market close for that day (fallback to trade price)
        bar = safe_get_bars(symbol, "1D", start=str(day), end=str(day + timedelta(days=1)))
        close_price = float(bar[-1].c) if bar else price

        # Keep last trade per day
        daily_values[day] = cash + shares * close_price

    df_daily = pd.DataFrame([
        {"date": pd.Timestamp(d).tz_localize(tz), "value": v}
        for d, v in daily_values.items()
    ])
    df_daily = df_daily.sort_values("date")

    daily_path = os.path.join(os.path.dirname(PORTFOLIO_PATH), f"daily_portfolio_{symbol}.csv")
    df_daily.to_csv(daily_path, index=False)
    print(f"[INFO] Daily portfolio saved: {daily_path}")
    return df_daily

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    symbols = SYMBOL if isinstance(SYMBOL, list) else [SYMBOL]
    for symbol in symbols:
        reconstruct_portfolio(symbol)
        generate_daily_portfolio(symbol)