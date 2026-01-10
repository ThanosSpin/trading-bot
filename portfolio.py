# portfolio.py
import os
import json
import csv
import pandas as pd
from datetime import datetime
import pytz

from config import PORTFOLIO_PATH, TIMEZONE
from broker import api_market


# ============================================================
# File helpers
# ============================================================

def get_portfolio_file(symbol):
    """Returns path to portfolio_<symbol>.json inside the same folder as PORTFOLIO_PATH."""
    base = os.path.dirname(PORTFOLIO_PATH)
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"portfolio_{symbol}.json")


def get_trade_log_file(symbol):
    """Returns path to trades_<symbol>.csv inside the same folder as PORTFOLIO_PATH."""
    base = os.path.dirname(PORTFOLIO_PATH)
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"trades_{symbol}.csv")


def get_daily_portfolio_file():
    """Returns path to daily_portfolio.csv inside the same folder as PORTFOLIO_PATH."""
    base = os.path.dirname(PORTFOLIO_PATH)
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "daily_portfolio.csv")


# ============================================================
# Live Portfolio
# ============================================================

def get_live_portfolio(symbol):
    """Fetch live Alpaca portfolio for a symbol."""
    symbol = symbol.upper()

    try:
        account = api_market.get_account()
        positions = api_market.list_positions()
        cash = float(account.cash)
    except Exception:
        return {"cash": 0.0, "shares": 0.0, "last_price": 0.0, "avg_price": 0.0}

    shares = 0.0
    last_price = 0.0
    avg_price = 0.0

    for p in positions:
        if str(getattr(p, "symbol", "")).upper() == symbol:
            try:
                shares = float(getattr(p, "qty", 0.0) or 0.0)
            except Exception:
                shares = 0.0

            try:
                last_price = float(getattr(p, "current_price", 0.0) or 0.0)
            except Exception:
                last_price = 0.0

            # Prefer Alpaca avg entry price (this is the key fix)
            try:
                avg_price = float(getattr(p, "avg_entry_price", 0.0) or 0.0)
            except Exception:
                avg_price = 0.0

            break

    return {"cash": cash, "shares": shares, "last_price": last_price, "avg_price": avg_price}


# ============================================================
# PortfolioManager Class
# ============================================================

class PortfolioManager:
    def __init__(self, symbol):
        self.symbol = symbol.upper()
        self.file = get_portfolio_file(symbol)
        self.data = self._load()

    # ------------------------
    # Loading / Saving
    # ------------------------

    def _load(self):
        """Load JSON portfolio; fallback to live if missing."""
        if os.path.exists(self.file):
            with open(self.file, "r") as f:
                data = json.load(f)

            # Backwards-compatible defaults
            data.setdefault("cash", 0.0)
            data.setdefault("shares", 0.0)
            data.setdefault("last_price", 0.0)

            # If missing, initialize from last_price (or 0)
            lp = float(data.get("last_price", 0.0) or 0.0)
            sh = float(data.get("shares", 0.0) or 0.0)

            data.setdefault("avg_price", lp if sh > 0 else 0.0)
            data.setdefault("max_price", lp if sh > 0 else 0.0)

            return data

        # If no local file → create initial state
        live = get_live_portfolio(self.symbol)
        sh = float(live.get("shares", 0.0) or 0.0)
        lp = float(live.get("last_price", 0.0) or 0.0)
        ap = float(live.get("avg_price", 0.0) or 0.0)

        data = {
            "cash": live["cash"],
            "shares": live.get("shares", 0.0),
            "last_price": live.get("last_price", 0.0),
            "avg_price": live.get("last_price", 0.0) if live.get("shares", 0.0) > 0 else 0.0,
            "max_price": live.get("last_price", 0.0) if live.get("shares", 0.0) > 0 else 0.0,
        }
        return data

    # ------------------------
    # Live synchronization
    # ------------------------

    def refresh_live(self):
        """Refresh LIVE Alpaca values (cash & shares). Keeps avg_price/max_price consistent."""
        try:
            live = get_live_portfolio(self.symbol)

            cash = float(live.get("cash", 0.0) or 0.0)
            sh   = float(live.get("shares", 0.0) or 0.0)
            lp   = float(live.get("last_price", 0.0) or 0.0)
            ap   = float(live.get("avg_price", 0.0) or 0.0)

            self.data["cash"] = cash
            self.data["shares"] = sh
            self.data["last_price"] = lp

            # Backfill missing keys for old portfolio files
            self.data.setdefault("avg_price", 0.0)
            self.data.setdefault("max_price", 0.0)

            if sh > 0 and lp > 0:
                # ✅ Prefer Alpaca avg entry if available (fixes STOP/TP/TRAIL logic)
                if ap > 0:
                    self.data["avg_price"] = ap
                else:
                    # fallback only if we truly don't have it
                    if float(self.data.get("avg_price", 0.0)) <= 0:
                        self.data["avg_price"] = lp

                # Update trailing max while in position
                mp = float(self.data.get("max_price", 0.0) or 0.0)
                if mp <= 0:
                    mp = lp
                self.data["max_price"] = max(mp, lp)

            else:
                # flat: reset to avoid stale trailing data
                self.data["shares"] = 0.0
                self.data["avg_price"] = 0.0
                self.data["max_price"] = 0.0
            
            # print(f"[LIVE] {self.symbol} shares={sh} last={lp} avg={self.data['avg_price']} max={self.data['max_price']}")

            self.save()
        except Exception:
            pass

    # ------------------------
    # Logging
    # ------------------------

    def log(self, action, price, qty):
        """Write trade entry into trades_<symbol>.csv."""
        log_path = get_trade_log_file(self.symbol)
        file_exists = os.path.isfile(log_path)
        timestamp = datetime.utcnow().isoformat()

        value = self.value()

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "timestamp", "symbol", "action", "qty",
                    "price", "cash", "shares", "value"
                ])
            writer.writerow([
                timestamp,
                self.symbol,
                action,
                f"{float(qty):g}",
                f"{float(price):.2f}" if price is not None else "N/A",
                f"{self.data['cash']:.2f}",
                f"{self.data['shares']:.8g}",
                f"{value:.2f}",
            ])

    # ------------------------
    # Trading updates
    # ------------------------

    def _apply(self, action, price, qty):
        """Internal balance update + logging + save."""
        price = float(price)
        qty = float(qty)

        if action == "buy":
            prev_shares = float(self.data.get("shares", 0.0))
            prev_avg = float(self.data.get("avg_price", 0.0))
            new_shares = prev_shares + qty

            # weighted avg entry
            if new_shares > 0:
                self.data["avg_price"] = (prev_shares * prev_avg + qty * price) / new_shares

            self.data["shares"] = new_shares
            self.data["cash"] -= qty * price

            # initialize/raise max_price
            mp = float(self.data.get("max_price", 0.0))
            self.data["max_price"] = max(mp, price)

        elif action == "sell":
            qty = min(qty, float(self.data.get("shares", 0.0)))
            self.data["shares"] -= qty
            self.data["cash"] += qty * price

            # reset when flat
            if float(self.data.get("shares", 0.0)) <= 0:
                self.data["shares"] = 0.0
                self.data["avg_price"] = 0.0
                self.data["max_price"] = 0.0

        self.data["last_price"] = price

        self.log(action, price, qty)
        self.save()

    # PUBLIC API -------------------------------------------------------

    def buy(self, qty, price):
        self._apply("buy", price, qty)

    def sell(self, qty, price):
        qty = min(qty, self.data["shares"])
        self._apply("sell", price, qty)

    def sell_all(self, price):
        shares = self.data["shares"]
        if shares > 0:
            self._apply("sell", price, shares)

    def buy_full(self, price):
        price = float(price)
        qty = self.data["cash"] / price
        if qty > 0:
            self._apply("buy", price, qty)

    # ------------------------
    # Value calculation
    # ------------------------

    def value(self):
        return float(self.data.get("cash", 0)) + \
               float(self.data.get("shares", 0)) * \
               float(self.data.get("last_price", 0))


# ============================================================
# Daily portfolio aggregation
# ============================================================

def save_daily_portfolio_csv(trades_list, initial_cash=1000.0):
    """
    trades_list: list of all trades from all symbols in chronological order
    Saves daily aggregated portfolio value.
    """
    tz = pytz.timezone(TIMEZONE)
    cash = initial_cash
    portfolio_state = {}
    daily_values = {}

    for t in trades_list:
        sym = t["symbol"]
        action = t["action"]
        qty = t["qty"]
        price = t["price"]
        timestamp = t["timestamp"]

        if sym not in portfolio_state:
            portfolio_state[sym] = {"shares": 0.0, "last_price": 0.0}

        if action == "buy":
            portfolio_state[sym]["shares"] += qty
            cash -= qty * price
        elif action == "sell":
            portfolio_state[sym]["shares"] -= qty
            cash += qty * price

        portfolio_state[sym]["last_price"] = price

        total_value = cash + sum(
            p["shares"] * p["last_price"] for p in portfolio_state.values()
        )
        daily_values[timestamp.date()] = total_value

    df_daily = pd.DataFrame([
        {"date": pd.Timestamp(d).tz_localize(tz), "value": v}
        for d, v in daily_values.items()
    ])

    df_daily.sort_values("date", inplace=True)
    df_daily.to_csv(get_daily_portfolio_file(), index=False)

    print(f"✅ Daily portfolio CSV saved: {get_daily_portfolio_file()}")
    return df_daily


# ============================================================
# Detect Deposits / Withdrawals from Alpaca Equity Curve
# ============================================================

def detect_capital_change(save_path="data/deposits_auto.csv", min_change=50):
    """
    Detects deposits or withdrawals by comparing Alpaca equity history.
    Saves results to deposits_auto.csv for dashboard markers.

    Arguments:
    - min_change: ignore tiny noise (equity fluctuates by a few cents).

    Output CSV columns:
    date, change, type, equity
    """

    # ------------------------------
    # 1. Fetch Alpaca equity history
    # ------------------------------
    try:
        hist = api_market.get_portfolio_history(period="1A", timeframe="1D")
        equity = pd.Series(hist.equity, index=pd.to_datetime(hist.timestamp, unit='s'))
        equity = equity.sort_index()
    except Exception as e:
        print(f"[ERROR] detect_capital_change: {e}")
        return None

    if equity.empty:
        print("[WARN] detect_capital_change: No equity history available.")
        return None

    # ------------------------------
    # 2. Compute daily changes
    # ------------------------------
    df = pd.DataFrame({
        "equity": equity,
        "change": equity.diff()
    })

    # Ignore very small fluctuations
    df["change_clean"] = df["change"].where(df["change"].abs() >= min_change)

    # Detect deposits (>0) and withdrawals (<0)
    df["type"] = df["change_clean"].apply(
        lambda x: "deposit" if x and x > 0 else ("withdrawal" if x and x < 0 else None)
    )

    df = df.dropna(subset=["type"]).copy()

    if df.empty:
        print("[INFO] detect_capital_change: No deposits or withdrawals detected.")
        return None

    df.reset_index(inplace=True)
    df.rename(columns={"index": "date"}, inplace=True)

    # ------------------------------
    # 3. Ensure data folder exists
    # ------------------------------
    base = os.path.dirname(save_path)
    if base and not os.path.exists(base):
        os.makedirs(base, exist_ok=True)

    # ------------------------------
    # 4. Save deposits/withdrawals
    # ------------------------------
    df_out = df[["date", "change_clean", "type", "equity"]].copy()
    df_out.rename(columns={"change_clean": "change"}, inplace=True)

    df_out.to_csv(save_path, index=False)
    print(f"💾 Saved deposit/withdrawal history → {save_path}")

    return df_out