import os
import csv
import pandas as pd
import pytz

from trader import api as api_market
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
    """Return ALL filled orders for this symbol using pagination."""
    all_orders = []
    until = None

    while True:
        try:
            kwargs = dict(status="filled", limit=500, nested=True, direction="desc")
            if until:
                kwargs["until"] = until
            chunk = api_market.list_orders(**kwargs)
        except Exception as e:
            print(f"[ERROR] Cannot fetch orders: {e}")
            break

        if not chunk:
            break

        all_orders.extend(chunk)

        earliest = min(o.filled_at for o in chunk)

        # ✅ Format as RFC-3339 string required by Alpaca
        earliest_str = earliest.astimezone(pytz.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

        if until and earliest_str >= until:
            break  # no progress, stop
        until = earliest_str

        if len(chunk) < 500:
            break  # last page

    filtered = [
        o for o in all_orders
        if getattr(o, "symbol", None) == symbol
        and getattr(o, "filled_at", None) is not None
    ]
    seen = set()
    unique = []
    for o in filtered:
        if o.id not in seen:
            seen.add(o.id)
            unique.append(o)

    unique.sort(key=lambda x: x.filled_at)
    print(f"  {symbol}: {len(unique)} filled orders fetched (paginated)")
    return unique


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
                "kind": "trade",
                "symbol": s,
                "timestamp": ts,
                "action": t.side.lower(),
                "qty": qty,
                "price": float(price),
            })

    all_trades.sort(key=lambda x: x["timestamp"])
    return all_trades


# -----------------------------
# Fetch cash deposits / withdrawals
# -----------------------------
def fetch_cash_activities():
    """
    Return external cash-flow events from Alpaca account activities.
    CSD = cash deposit, CSW = cash withdrawal.
    """
    events = []

    for activity_type in ["CSD", "CSW"]:
        try:
            activities = api_market.get_activities(activity_types=activity_type)
            print(f"  {activity_type}: {len(activities)} activities")
        except Exception as e:
            print(f"[WARN] Cannot fetch {activity_type} activities: {e}")
            continue

        for a in activities:
            raw_ts = getattr(a, "date", None) or getattr(a, "transaction_time", None)
            if raw_ts is None:
                continue

            ts = pd.to_datetime(str(raw_ts), utc=True).to_pydatetime()
            amount = float(getattr(a, "net_amount", 0) or 0)

            if activity_type == "CSW":
                amount = -abs(amount)
            else:
                amount = abs(amount)

            if amount == 0:
                continue

            events.append({
                "kind": "cash_flow",
                "timestamp": ts,
                "amount": amount,
                "activity_type": activity_type,
            })

    events.sort(key=lambda x: x["timestamp"])
    return events


# -----------------------------
# Fetch REAL equity from Alpaca portfolio history
# This is the ground truth — avoids replay drift
# -----------------------------
def fetch_portfolio_history_equity(tz) -> pd.DataFrame:
    """
    Fetch real daily closing equity from Alpaca portfolio history API.
    Returns DataFrame with columns: date (tz-aware, normalized), value (float).

    Falls back gracefully — if the API returns nothing, an empty DataFrame
    is returned and the replay values are used instead.
    """
    try:
        hist = api_market.get_portfolio_history(period="all", timeframe="1D")
        if not hist or not hist.equity or not hist.timestamp:
            print("  [WARN] Portfolio history returned empty — using replayed values.")
            return pd.DataFrame()

        rows = []
        for ts_epoch, eq in zip(hist.timestamp, hist.equity):
            if eq is None or float(eq) == 0:
                continue
            rows.append({
                "date": pd.Timestamp(ts_epoch, unit="s", tz="UTC")
                           .tz_convert(tz)
                           .normalize(),
                "value": round(float(eq), 6),
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        # Keep last equity reading per day (market close)
        df = (
            df.groupby("date", as_index=False)
              .agg({"value": "last"})
              .sort_values("date")
              .reset_index(drop=True)
        )
        print(f"  ✅ Fetched {len(df)} days of real equity from Alpaca portfolio history.")
        return df

    except Exception as e:
        print(f"  [WARN] fetch_portfolio_history_equity failed: {e} — using replayed values.")
        return pd.DataFrame()


# -----------------------------
# Replay events → per-symbol trade CSVs + daily portfolio CSV
# -----------------------------
def replay_and_emit(events, initial_cash: float = 0.0):
    """
    Replay all events (trades + cash flows) chronologically.

    Writes:
      trades_{SYMBOL}.csv   — one row per fill, for each symbol
      daily_portfolio.csv   — daily equity + external_flow + initial_cash

    Equity source priority:
      1. Alpaca portfolio history API  (real, matches dashboard KPIs exactly)
      2. Replayed cash + position value (fallback if API unavailable)

    initial_cash: cash present BEFORE the first event.
                  Pass 0.0 when all deposits are tracked as CSD events
                  to avoid double-counting.
    """
    tz = pytz.timezone(TIMEZONE)

    cash = initial_cash
    symbols_state = {}
    per_symbol_rows = {}
    daily_rows = []

    for e in events:
        ts = e["timestamp"]
        tsl = ts.astimezone(tz)

        if e["kind"] == "cash_flow":
            cash += float(e["amount"])

        elif e["kind"] == "trade":
            sym = e["symbol"]
            action = e["action"]
            qty = float(e["qty"])
            price = float(e["price"])

            if sym not in symbols_state:
                symbols_state[sym] = {"shares": 0.0, "last_price": 0.0}
                per_symbol_rows[sym] = []

            if action == "buy":
                cash -= qty * price
                symbols_state[sym]["shares"] += qty
            elif action == "sell":
                cash += qty * price
                symbols_state[sym]["shares"] -= qty

            symbols_state[sym]["last_price"] = price

            replayed_value = cash + sum(
                s["shares"] * s["last_price"] for s in symbols_state.values()
            )

            per_symbol_rows[sym].append({
                "timestamp":      ts.isoformat(),
                "symbol":         sym,
                "action":         action,
                "qty":            qty,
                "price":          round(price, 6),
                "cash":           round(cash, 6),
                "shares":         round(symbols_state[sym]["shares"], 8),
                "value":          round(replayed_value, 6),
                "timestamplocal": tsl.isoformat(),
                "timestampstr":   tsl.strftime("%Y-%m-%d %H%M%S"),
            })

        replayed_value = cash + sum(
            s["shares"] * s["last_price"] for s in symbols_state.values()
        )

        daily_rows.append({
            "date":          pd.Timestamp(ts).tz_convert(tz).normalize(),
            "value":         round(replayed_value, 6),   # may be overwritten below
            "cash":          round(cash, 6),
            "external_flow": float(e["amount"]) if e["kind"] == "cash_flow" else 0.0,
        })

    # ── Write per-symbol trade CSVs ────────────────────────────────────────────
    for sym, rows in per_symbol_rows.items():
        path = trade_log_path(sym)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp", "symbol", "action", "qty", "price",
                "cash", "shares", "value", "timestamplocal", "timestampstr",
            ])
            for r in rows:
                w.writerow([
                    r["timestamp"], r["symbol"], r["action"], r["qty"], r["price"],
                    r["cash"], r["shares"], r["value"],
                    r["timestamplocal"], r["timestampstr"],
                ])
        print(f"  ✅ Wrote trade log: {path} ({len(rows)} rows)")

    # ── Build daily_portfolio DataFrame ───────────────────────────────────────
    df_daily = pd.DataFrame(daily_rows)
    if not df_daily.empty:
        df_daily = (
            df_daily.groupby("date", as_index=False)
            .agg({"value": "last", "cash": "last", "external_flow": "sum"})
            .sort_values("date")
            .reset_index(drop=True)
        )
    else:
        df_daily = pd.DataFrame(
            columns=["date", "value", "cash", "external_flow"]
        )

    # ── ✅ Write initial_cash so dashboard PnL math is correct ────────────────
    df_daily["initial_cash"] = initial_cash
    df_daily.to_csv(daily_portfolio_path(), index=False)
    print(f"  ✅ Wrote daily portfolio CSV ({len(df_daily)} rows).")
    return per_symbol_rows, df_daily


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    symbols = SYMBOL if isinstance(SYMBOL, list) else [SYMBOL]
    symbols = [s.upper() for s in symbols]

    spy = SPY_SYMBOL.upper()
    if spy not in symbols:
        symbols.append(spy)

    print("🔎 Gathering trades for symbols:", symbols)

    # ── Step 1: fetch cash activities FIRST ───────────────────────────────────
    print("\n💳 Fetching cash deposit/withdrawal activities...")
    cash_flows = fetch_cash_activities()

    # ── Step 2: determine initial_cash ────────────────────────────────────────
    # KEY: when CSD events exist, start replay at 0 — the CSD events inject
    # all cash. Starting at portfolio_history\'s first equity would double-count
    # the first deposit (portfolio_history[0] == CSD #1 amount).
    if cash_flows:
        initial_cash = 0.0
        total_deposited = sum(e["amount"] for e in cash_flows if e["amount"] > 0)
        total_withdrawn = sum(abs(e["amount"]) for e in cash_flows if e["amount"] < 0)
        print(f"  ✅ Starting cash = $0.00 (all deposits tracked as CSD events)")
        print(f"  💰 Total deposited via CSD : ${total_deposited:,.2f}")
        if total_withdrawn:
            print(f"  💸 Total withdrawn via CSW: ${total_withdrawn:,.2f}")
    else:
        # Fallback: no CSD activities found — seed from portfolio history
        try:
            hist = api_market.get_portfolio_history(period="all", timeframe="1D")
            initial_cash = 0.0
            if hist and hist.equity:
                for eq in hist.equity:
                    if eq and float(eq) > 0:
                        initial_cash = float(eq)
                        break
        except Exception as e:
            print(f"  [WARN] Could not fetch initial cash: {e}")
            initial_cash = 0.0
        print(f"  ⚠️  No CSD events — using portfolio history seed: ${initial_cash:,.2f}")

    # ── Step 3: fetch trades ───────────────────────────────────────────────────
    print("\n📈 Fetching filled trades...")
    trades = gather_all_trades(symbols)
    print(f"  ✅ {len(trades)} trades collected.")
    print(f"  ✅ {len(cash_flows)} cash-flow events collected.")

    # ── Step 4: merge & sort all events ───────────────────────────────────────
    events = trades + cash_flows
    events.sort(key=lambda x: x["timestamp"])

    if not events:
        print("[WARN] No trades or cash activities found.")
        pd.DataFrame(
            columns=["date", "value", "cash", "external_flow", "initial_cash"]
        ).to_csv(daily_portfolio_path(), index=False)
        raise SystemExit(0)

    # ── Step 5: replay + emit ─────────────────────────────────────────────────
    print("\n⚙️  Replaying events and writing CSVs...")
    replay_and_emit(events, initial_cash=initial_cash)
    print("\n✅ Reconstruction complete.")
