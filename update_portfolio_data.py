import os
import csv
import pandas as pd
import pytz
from datetime import timedelta
from broker import api
from portfolio import _trade_log_file, PORTFOLIO_PATH
from config import SYMBOL, TIMEZONE  # Read symbols and timezone from config

def fetch_filled_trades(symbol):
    """Fetch all filled trades for a symbol from Alpaca and sort by execution time"""
    trades = api.list_orders(status="filled", symbols=[symbol], limit=1000, nested=True)
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
        try:
            bar = api.get_bars(symbol, "1Min", start=ts.isoformat(), end=(ts + timedelta(minutes=1)).isoformat())
            last_price = float(bar[-1].c) if bar else price
        except Exception:
            last_price = price

        value = cash + shares * last_price

        rows.append([
            ts.isoformat(),  # UTC timestamp
            symbol,
            action,
            f"{price:.2f}",
            f"{cash:.2f}",
            f"{shares:.2f}",
            f"{value:.2f}",
            ts_local.isoformat(),
            ts_local.strftime("%Y-%m-%d %H:%M")
        ])

    # Save trade log CSV
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "symbol", "action", "price", "cash", "shares", "value", "timestamp_local", "timestamp_str"])
        writer.writerows(rows)

    print(f"[INFO] Trade log saved for {symbol}: {log_path}")
    return rows

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
        try:
            bar = api.get_bars(symbol, "1D", start=str(day), end=str(day + timedelta(days=1)))
            close_price = float(bar[-1].c) if bar else price
        except Exception:
            close_price = price

        # Keep last trade per day
        daily_values[day] = cash + shares * close_price

    # Convert to DataFrame
    df_daily = pd.DataFrame([
        {"date": pd.Timestamp(d).tz_localize(tz), "value": v}
        for d, v in daily_values.items()
    ])
    df_daily = df_daily.sort_values("date")

    # Save daily portfolio CSV
    daily_path = os.path.join(os.path.dirname(PORTFOLIO_PATH), f"daily_portfolio_{symbol}.csv")
    df_daily.to_csv(daily_path, index=False)
    print(f"[INFO] Daily portfolio saved: {daily_path}")
    return df_daily

if __name__ == "__main__":
    symbols = SYMBOL if isinstance(SYMBOL, list) else [SYMBOL]
    for symbol in symbols:
        reconstruct_portfolio(symbol)
        generate_daily_portfolio(symbol)