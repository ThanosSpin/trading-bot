# dashboard.py
import os
import streamlit as st
import pandas as pd
import pytz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
from streamlit_autorefresh import st_autorefresh

from config import TIMEZONE, SYMBOL
from portfolio import (
    get_trade_log_file,
    get_daily_portfolio_file,
    get_live_portfolio
)
from trader import get_pdt_status


# -------------------------------------------------
# Streamlit setup
# -------------------------------------------------
st.set_page_config(page_title="Trading Bot Dashboard", layout="wide")
st.title("📊 Trading Bot Dashboard")

REFRESH_INTERVAL = 60
st_autorefresh(interval=REFRESH_INTERVAL * 1000, key="global_refresh")
st.caption(f"⏳ Auto-refreshing every {REFRESH_INTERVAL} seconds.")

symbols = SYMBOL if isinstance(SYMBOL, list) else [SYMBOL]
tz = pytz.timezone(TIMEZONE)


# -------------------------------------------------
# PORTFOLIO SUMMARY (live Alpaca)
# -------------------------------------------------
st.header("Portfolio Summary")

for sym in symbols:
    try:
        live = get_live_portfolio(sym)
        cash = live["cash"]
        shares = live["shares"]
        last_price = live["last_price"]
        value = cash + shares * last_price

        st.subheader(f"Live Portfolio ({sym})")
        c1, c2, c3 = st.columns(3)
        c1.metric("Cash", f"${cash:,.2f}")
        c2.metric("Shares", f"{shares:,.2f}")
        c3.metric("Value", f"${value:,.2f}")

    except Exception as e:
        st.error(f"Error fetching live Alpaca data for {sym}: {e}")


# -------------------------------------------------
# PDT STATUS
# -------------------------------------------------
st.header("📊 PDT Account Status")
pdt = get_pdt_status()

if pdt:
    msg = (
        f"Equity: ${pdt['equity']:.2f} | "
        f"Day Trades (5d): {pdt['daytrade_count']} | "
        f"Remaining: {pdt['remaining']} | "
        f"{'⚠️ PDT FLAGGED' if pdt['is_pdt'] else '✅ Not PDT'}"
    )

    if pdt["is_pdt"]:
        st.error(msg)
    elif isinstance(pdt["remaining"], int) and pdt["remaining"] <= 1:
        st.warning(msg)
    else:
        st.success(msg)
else:
    st.info("Unable to fetch PDT status.")


# -------------------------------------------------
# TRADE LOG + ANALYTICS (per symbol)
# -------------------------------------------------
st.header("Trade Logs + Analytics")

for sym in symbols:

    st.subheader(f"Trades for {sym}")
    path = get_trade_log_file(sym)

    if not os.path.exists(path):
        st.info(f"No trade log found for {sym}.")
        continue

    df = pd.read_csv(path)

    # --- Timestamp formatting ---
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["timestamp_local"] = df["timestamp"].dt.tz_convert(tz)
    df["timestamp_str"] = df["timestamp_local"].dt.strftime("%Y-%m-%d %H:%M:%S")

    st.dataframe(df.sort_values("timestamp_local", ascending=False), use_container_width=True)

    # -------------------------------------------------
    # TRADE ANALYTICS
    # -------------------------------------------------
    st.subheader(f"📈 Trade Analytics: {sym}")

    df_trades = df.copy()
    if df_trades.empty:
        st.info("No trades available for analytics.")
        continue

    # Compute buy/sell cashflow
    df_trades["trade_value"] = df_trades["qty"] * df_trades["price"]
    df_trades["pnl"] = df_trades.apply(
        lambda row: -row["trade_value"] if row["action"] == "buy" else row["trade_value"],
        axis=1
    )

    df_trades = df_trades.sort_values("timestamp")
    df_trades["shares_after"] = df_trades["shares"]

    cycle_pnls = []
    cycle_rows = []

    for _, r in df_trades.iterrows():
        cycle_rows.append(r)

        # Closed trade cycle = shares go back to 0
        if r["shares_after"] == 0 and len(cycle_rows) >= 2:
            pnl = sum(x["pnl"] for x in cycle_rows)
            cycle_pnls.append(pnl)
            cycle_rows = []

    if not cycle_pnls:
        st.info("Not enough closed trades to compute analytics yet.")
        continue

    s = pd.Series(cycle_pnls)

    gross_profit = s[s > 0].sum()
    gross_loss = -s[s < 0].sum()
    win_rate = (s > 0).mean() * 100
    avg_win = s[s > 0].mean() if (s > 0).any() else 0
    avg_loss = s[s < 0].mean() if (s < 0).any() else 0
    largest_win = s.max()
    largest_loss = s.min()
    profit_factor = (gross_profit / gross_loss) if gross_loss != 0 else float("inf")

    colA, colB, colC, colD = st.columns(4)
    colA.metric("Win Rate", f"{win_rate:.1f}%")
    colB.metric("Profit Factor", f"{profit_factor:.2f}")
    colC.metric("Avg Win / Avg Loss", f"{avg_win:.2f} / {avg_loss:.2f}")
    colD.metric("Largest Win / Largest Loss", f"{largest_win:.2f} / {largest_loss:.2f}")


# -------------------------------------------------
# GLOBAL DAILY PORTFOLIO PERFORMANCE
# -------------------------------------------------
st.header("📈 Total Portfolio Performance (All Symbols Combined)")

daily_path = get_daily_portfolio_file()

if os.path.exists(daily_path):
    df_daily = pd.read_csv(daily_path)

    if df_daily.empty:
        st.warning("Daily portfolio is empty.")
    else:
        df_daily["date"] = pd.to_datetime(df_daily["date"], utc=True, errors="coerce")
        df_daily = df_daily.dropna(subset=["date"])
        df_daily["date"] = df_daily["date"].dt.tz_convert(tz)

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(df_daily["date"], df_daily["value"], marker="o")

        ax.set_title("Total Portfolio Value Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value ($)")
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.2f}"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.xticks(rotation=45)
        ax.grid(True)
        plt.tight_layout()

        st.pyplot(fig)
else:
    st.info("No daily portfolio data found. Run update_portfolio_data.py first.")