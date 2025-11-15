# update_portfolio_data.py
import os
import csv
import pandas as pd
import pytz
from datetime import datetime
from broker import api_market
from config import SYMBOL, TIMEZONE, PORTFOLIO_PATH

# -----------------------------
# Helpers: file paths
# -----------------------------
def trade_log_path(symbol):
    base = os.path.dirname(PORTFOLIO_PATH)
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"trades_{symbol}.csv")

def daily_portfolio_path():
    base = os.path.dirname(PORTFOLIO_PATH)
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "daily_portfolio.csv")

# -----------------------------
# Fetch filled trades for a symbol
# -----------------------------
def fetch_filled_trades(symbol):
    """Return list of filled order objects (sorted by filled_at)."""
    orders = api_market.list_orders(status="filled", symbols=[symbol], limit=1000, nested=True)
    # filter out those with no filled_at just in case
    trades = [o for o in orders if getattr(o, "filled_at", None) is not None]
    trades.sort(key=lambda x: x.filled_at)
    return trades

# -----------------------------
# Build a flat list of trade dicts for all symbols
# -----------------------------
def gather_all_trades(symbols):
    all_trades = []
    for s in symbols:
        trades = fetch_filled_trades(s)
        for t in trades:
            ts = t.filled_at
            # ensure timezone-aware UTC
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=pytz.UTC)
            price = getattr(t, "filled_avg_price", None) or getattr(t, "filled_avg", None) or None
            qty = float(getattr(t, "filled_qty", 0) or 0)
            if price is None:
                # fallback to order price fields (best effort)
                price = getattr(t, "price", None) or 0.0
            all_trades.append({
                "symbol": s,
                "timestamp": ts,
                "action": t.side.lower(),
                "qty": qty,
                "price": float(price),
                "raw": t
            })
    # global chronological order
    all_trades.sort(key=lambda x: x["timestamp"])
    return all_trades

# -----------------------------
# Infer initial cash using final (live) cash and net trade cash flows
# -----------------------------
def infer_initial_cash(all_trades):
    """
    final_cash = initial_cash + net_change
    net_change = sum(sell_values) - sum(buy_values)
    so initial_cash = final_cash - net_change
    """
    buy_total = 0.0
    sell_total = 0.0
    for t in all_trades:
        val = t["qty"] * t["price"]
        if t["action"] == "buy":
            buy_total += val
        else:
            sell_total += val
    net_change = sell_total - buy_total
    # fetch live Alpaca cash (current)
    acct = api_market.get_account()
    final_cash = float(acct.cash)
    initial_cash = final_cash - net_change
    return float(initial_cash), float(final_cash), float(net_change)

# -----------------------------
# Replay trades chronologically to produce per-symbol CSVs and a global daily series
# -----------------------------
def replay_and_emit(all_trades, initial_cash):
    tz = pytz.timezone(TIMEZONE)

    # state
    cash = float(initial_cash)
    symbols_state = {}  # symbol -> {"shares": float, "last_price": float}
    # prepare per-symbol rows
    per_symbol_rows = {}

    # for daily portfolio (use timestamp.date() key)
    daily_values = {}

    for t in all_trades:
        sym = t["symbol"]
        action = t["action"]
        qty = float(t["qty"])
        price = float(t["price"])
        ts = t["timestamp"]
        ts_local = ts.astimezone(tz)

        if sym not in symbols_state:
            symbols_state[sym] = {"shares": 0.0, "last_price": 0.0}
            per_symbol_rows[sym] = []

        # apply trade
        if action == "buy":
            cash -= qty * price
            symbols_state[sym]["shares"] += qty
        elif action == "sell":
            cash += qty * price
            symbols_state[sym]["shares"] -= qty
        else:
            # unknown action — skip
            continue

        symbols_state[sym]["last_price"] = price

        # compute total portfolio value across all symbols at this timestamp
        total_value = cash + sum(state["shares"] * state["last_price"] for state in symbols_state.values())

        # record row for symbol CSV
        per_symbol_rows[sym].append({
            "timestamp": ts.isoformat(),
            "symbol": sym,
            "action": action,
            "qty": qty,
            "price": price,
            "cash": round(cash, 6),
            "shares": round(symbols_state[sym]["shares"], 8),
            "value": round(total_value, 6),
            "timestamp_local": ts_local.isoformat(),
            "timestamp_str": ts_local.strftime("%Y-%m-%d %H:%M:%S")
        })

        # record daily value (last value on that date)
        daily_values[ts.date()] = total_value

    # write per-symbol CSVs
    for sym, rows in per_symbol_rows.items():
        path = trade_log_path(sym)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp","symbol","action","qty","price","cash","shares","value","timestamp_local","timestamp_str"])
            for r in rows:
                writer.writerow([
                    r["timestamp"], r["symbol"], r["action"], r["qty"], f"{r['price']:.6f}",
                    f"{r['cash']:.6f}", f"{r['shares']:.8g}", f"{r['value']:.6f}",
                    r["timestamp_local"], r["timestamp_str"]
                ])
        print(f"✅ Wrote trade log: {path} ({len(rows)} rows)")

    # build daily DataFrame and save global daily_portfolio.csv
    df_daily = pd.DataFrame([{"date": pd.Timestamp(d).tz_localize(tz), "value": v} for d, v in daily_values.items()])
    df_daily = df_daily.sort_values("date").reset_index(drop=True)
    daily_path = daily_portfolio_path()
    df_daily.to_csv(daily_path, index=False)
    print(f"✅ Wrote daily portfolio: {daily_path} ({len(df_daily)} days)")

    return per_symbol_rows, df_daily

# -----------------------------
# Top-level
# -----------------------------
if __name__ == "__main__":
    symbols = SYMBOL if isinstance(SYMBOL, list) else [SYMBOL]
    print("🔎 Gathering trades for symbols:", symbols)

    all_trades = gather_all_trades(symbols)
    if not all_trades:
        print("[WARN] No filled trades found for the configured symbols.")
        # still create empty daily file to avoid dashboard errors
        pd.DataFrame(columns=["date","value"]).to_csv(daily_portfolio_path(), index=False)
        raise SystemExit(0)

    print(f"ℹ️ {len(all_trades)} total trades collected across symbols.")

    initial_cash, final_cash, net_change = infer_initial_cash(all_trades)
    print(f"ℹ️ Inferred initial cash: {initial_cash:.2f}  (final_cash={final_cash:.2f}, net_change={net_change:.2f})")

    per_symbol_rows, df_daily = replay_and_emit(all_trades, initial_cash)

    print("✅ Reconstruction completed.")