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

# Auto-refresh
REFRESH_INTERVAL = 60
st_autorefresh(interval=REFRESH_INTERVAL * 1000, key="global_refresh")
st.caption(f"⏳ Auto-refreshing every {REFRESH_INTERVAL} seconds.")

# Normalize SYMBOL into list
symbols = SYMBOL if isinstance(SYMBOL, list) else [SYMBOL]
tz = pytz.timezone(TIMEZONE)

# -------------------------------------------------
# PORTFOLIO SUMMARY (per symbol live from Alpaca)
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
# TRADE LOGS & TRADE ANALYTICS
# -------------------------------------------------
st.header("Trade Logs & Analytics")
for sym in symbols:
    st.subheader(f"Trades for {sym}")
    trade_path = get_trade_log_file(sym)

    if not os.path.exists(trade_path):
        st.info(f"No trade log found for {sym}.")
        continue

    df_trades = pd.read_csv(trade_path)
    df_trades["timestamp"] = pd.to_datetime(df_trades["timestamp"], utc=True, errors="coerce")
    df_trades = df_trades.dropna(subset=["timestamp"])
    df_trades["timestamp_local"] = df_trades["timestamp"].dt.tz_convert(tz)
    df_trades["timestamp_str"] = df_trades["timestamp_local"].dt.strftime("%Y-%m-%d %H:%M:%S")
    st.dataframe(df_trades.sort_values("timestamp_local", ascending=False), use_container_width=True)

    # ---------------------------
    # Trade Analytics
    # ---------------------------
    if not df_trades.empty:
        # Compute PnL per trade
        df_trades["trade_value"] = df_trades["qty"] * df_trades["price"]
        df_trades["pnl"] = df_trades.apply(
            lambda row: -row["trade_value"] if row["action"].lower() == "buy" else row["trade_value"], axis=1
        )

        # Closed trade cycles (flat → flat)
        df_trades = df_trades.sort_values("timestamp")
        df_trades["shares_after"] = df_trades["shares"]

        cycle_pnls = []
        cycle_rows = []
        prev_shares = 0

        for _, r in df_trades.iterrows():
            cycle_rows.append(r)
            if r["shares_after"] == prev_shares and len(cycle_rows) >= 2:
                pnl = sum(x["pnl"] for x in cycle_rows)
                cycle_pnls.append(pnl)
                cycle_rows = []

        if cycle_pnls:
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
            colC.metric("Avg Win / Loss", f"{avg_win:.2f} / {avg_loss:.2f}")
            colD.metric("Largest Win / Loss", f"{largest_win:.2f} / {largest_loss:.2f}")
        else:
            st.info("Not enough closed trades to compute analytics yet.")

# -------------------------------------------------
# TOTAL DAILY PORTFOLIO PERFORMANCE
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