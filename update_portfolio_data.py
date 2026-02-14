# update_portfolio_data.py
import os
import csv
import pandas as pd
import pytz
from datetime import datetime

from trader import api as api_market   # ✅ use updated Alpaca client
from config.config import SYMBOL, TIMEZONE, PORTFOLIO_PATH, SPY_SYMBOL


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
    """
    Return list of filled orders *for this symbol only*.
    """
    try:
        orders = api_market.list_orders(status="filled", limit=1000, nested=True)
    except Exception as e:
        print(f"[ERROR] Cannot fetch orders: {e}")
        return []

    filtered = []
    for o in orders:
        if getattr(o, "symbol", None) != symbol:
            continue
        if getattr(o, "filled_at", None) is None:
            continue
        filtered.append(o)

    filtered.sort(key=lambda x: x.filled_at)
    return filtered


# -----------------------------
# Build trade list across all symbols
# -----------------------------
def gather_all_trades(symbols):
    all_trades = []

    for s in symbols:
        trades = fetch_filled_trades(s)
        for t in trades:
            ts = t.filled_at
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=pytz.UTC)

            price = (
                getattr(t, "filled_avg_price", None)
                or getattr(t, "filled_avg", None)
                or getattr(t, "price", None)
                or 0.0
            )
            qty = float(getattr(t, "filled_qty", 0) or 0)

            all_trades.append({
                "symbol": s,
                "timestamp": ts,
                "action": t.side.lower(),
                "qty": qty,
                "price": float(price),
                "raw": t
            })

    all_trades.sort(key=lambda x: x["timestamp"])
    return all_trades


# -----------------------------
# Infer initial cash
# -----------------------------
def infer_initial_cash(all_trades):
    buy_total = 0.0
    sell_total = 0.0

    for t in all_trades:
        val = t["qty"] * t["price"]
        if t["action"] == "buy":
            buy_total += val
        else:
            sell_total += val

    net_change = sell_total - buy_total

    acct = api_market.get_account()
    final_cash = float(acct.cash)

    initial_cash = final_cash - net_change
    return initial_cash, final_cash, net_change


# -----------------------------
# Replay trades and generate CSVs
# -----------------------------
def replay_and_emit(all_trades, initial_cash):
    tz = pytz.timezone(TIMEZONE)

    cash = float(initial_cash)
    symbols_state = {}
    per_symbol_rows = {}
    daily_values = {}

    for t in all_trades:
        sym = t["symbol"]
        action = t["action"]
        qty = float(t["qty"])
        price = float(t["price"])
        ts = t["timestamp"]
        tsl = ts.astimezone(tz)

        if sym not in symbols_state:
            symbols_state[sym] = {"shares": 0.0, "last_price": 0.0}
            per_symbol_rows[sym] = []

        # process trade
        if action == "buy":
            cash -= qty * price
            symbols_state[sym]["shares"] += qty
        elif action == "sell":
            cash += qty * price
            symbols_state[sym]["shares"] -= qty

        symbols_state[sym]["last_price"] = price

        total_value = cash + sum(
            s["shares"] * s["last_price"] for s in symbols_state.values()
        )

        per_symbol_rows[sym].append({
            "timestamp": ts.isoformat(),
            "symbol": sym,
            "action": action,
            "qty": qty,
            "price": round(price, 6),
            "cash": round(cash, 6),
            "shares": round(symbols_state[sym]["shares"], 8),
            "value": round(total_value, 6),
            "timestamp_local": tsl.isoformat(),
            "timestamp_str": tsl.strftime("%Y-%m-%d %H:%M:%S")
        })

        daily_values[ts.date()] = total_value

    # write per-symbol logs
    for sym, rows in per_symbol_rows.items():
        path = trade_log_path(sym)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp","symbol","action","qty","price","cash","shares",
                        "value","timestamp_local","timestamp_str"])
            for r in rows:
                w.writerow([
                    r["timestamp"], r["symbol"], r["action"], r["qty"], r["price"],
                    r["cash"], r["shares"], r["value"],
                    r["timestamp_local"], r["timestamp_str"]
                ])

        print(f"✅ Wrote trade log: {path} ({len(rows)} rows)")

    # write daily portfolio
    df_daily = pd.DataFrame([
        {"date": pd.Timestamp(d).tz_localize(tz), "value": v}
        for d, v in daily_values.items()
    ])
    df_daily = df_daily.sort_values("date")
    df_daily.to_csv(daily_portfolio_path(), index=False)

    print(f"✅ Wrote daily portfolio CSV.")
    return per_symbol_rows, df_daily


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    symbols = SYMBOL if isinstance(SYMBOL, list) else [SYMBOL]
    symbols = [s.upper() for s in symbols]

    # ✅ Always include SPY so reconstruction matches dashboard reality
    spy = SPY_SYMBOL.upper()
    if spy not in symbols:
        symbols.append(spy)

    print("🔎 Gathering trades for symbols:", symbols)

    all_trades = gather_all_trades(symbols)

    if not all_trades:
        print("[WARN] No filled trades found.")
        pd.DataFrame(columns=["date","value"]).to_csv(daily_portfolio_path(), index=False)
        raise SystemExit(0)

    print(f"ℹ️ {len(all_trades)} trades collected.")

    initial_cash, final_cash, net_change = infer_initial_cash(all_trades)
    print(f"ℹ️ initial_cash={initial_cash:.2f}, final_cash={final_cash:.2f}, net_change={net_change:.2f}")

    replay_and_emit(all_trades, initial_cash)
    print("✅ Reconstruction complete.")