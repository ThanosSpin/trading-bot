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
    try:
        account = api_market.get_account()
        positions = api_market.list_positions()
        cash = float(account.cash)
    except Exception:
        return {"cash": 0.0, "shares": 0.0, "last_price": 0.0}

    shares = 0.0
    last_price = 0.0

    for p in positions:
        if p.symbol == symbol:
            shares = float(p.qty)
            last_price = float(p.current_price)

    return {"cash": cash, "shares": shares, "last_price": last_price}


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
                return json.load(f)

        # If no local file → create initial state
        live = get_live_portfolio(self.symbol)
        data = {
            "cash": live["cash"],
            "shares": live.get("shares", 0.0),
            "last_price": live.get("last_price", 0.0),
        }

        return data

    def save(self):
        with open(self.file, "w") as f:
            json.dump(self.data, f, indent=4)

    # ------------------------
    # Live synchronization
    # ------------------------

    def refresh_live(self):
        """Refresh LIVE Alpaca values (cash & shares)."""
        try:
            live = get_live_portfolio(self.symbol)
            self.data["cash"] = live["cash"]
            self.data["shares"] = live["shares"]
            self.data["last_price"] = live["last_price"]
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
            self.data["shares"] += qty
            self.data["cash"] -= qty * price

        elif action == "sell":
            qty = min(qty, self.data["shares"])  # safety
            self.data["shares"] -= qty
            self.data["cash"] += qty * price

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