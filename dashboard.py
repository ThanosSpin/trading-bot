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

symbols = SYMBOL if isinstance(SYMBOL, list) else [SYMBOL]
tz = pytz.timezone(TIMEZONE)


# -------------------------------------------------
# PORTFOLIO SUMMARY (Live Alpaca)
# -------------------------------------------------
st.header("Portfolio Summary")

for sym in symbols:
    try:
        live = get_live_portfolio(sym)
        cash = live["cash"]
        shares = live["shares"]
        last_price = live["last_price"]
        value = cash + shares * last_price

        st.subheader(f"Live Portfolio — {sym}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Cash", f"${cash:,.2f}")
        col2.metric("Shares", f"{shares:,.2f}")
        col3.metric("Value", f"${value:,.2f}")

    except Exception as e:
        st.error(f"⚠️ Could not fetch live Alpaca portfolio for {sym}: {e}")


# -------------------------------------------------
# PDT STATUS
# -------------------------------------------------
st.header("📊 PDT Account Status")

pdt = get_pdt_status()
if pdt:
    text = (
        f"Equity: ${pdt['equity']:.2f} | "
        f"Day Trades (5d): {pdt['daytrade_count']} | "
        f"Remaining: {pdt['remaining']} | "
        f"{'⚠️ PDT FLAGGED' if pdt['is_pdt'] else '✅ Not PDT'}"
    )
    if pdt["is_pdt"]:
        st.error(text)
    elif isinstance(pdt["remaining"], int) and pdt["remaining"] <= 1:
        st.warning(text)
    else:
        st.success(text)
else:
    st.info("Could not fetch PDT information.")


# -------------------------------------------------
# TRADE LOGS & ANALYTICS
# -------------------------------------------------
st.header("Trades & Analytics")

for sym in symbols:
    st.subheader(f"Trades for {sym}")

    trade_file = get_trade_log_file(sym)
    if not os.path.exists(trade_file):
        st.info(f"No trades logged for {sym}.")
        continue

    df = pd.read_csv(trade_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["timestamp_local"] = df["timestamp"].dt.tz_convert(tz)
    df["timestamp_str"] = df["timestamp_local"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Show trade table
    st.dataframe(df.sort_values("timestamp_local", ascending=False), use_container_width=True)

    # ---------------------------
    # Trade Analytics
    # ---------------------------
    st.subheader(f"📈 Trade Analytics — {sym}")

    df = df.sort_values("timestamp")

    # PnL per trade row
    df["trade_value"] = df["qty"] * df["price"]
    df["pnl"] = df.apply(
        lambda r: -r["trade_value"] if r["action"] == "buy" else r["trade_value"],
        axis=1
    )

    # Compute closed trade cycles: flat → flat
    cycle_pnls = []
    cycle = []

    for _, r in df.iterrows():
        cycle.append(r)
        if r["shares"] == 0 and len(cycle) > 1:
            pnl = sum(x["pnl"] for x in cycle)
            cycle_pnls.append(pnl)
            cycle = []

    if cycle_pnls:
        s = pd.Series(cycle_pnls)

        gross_profit = s[s > 0].sum()
        gross_loss = -s[s < 0].sum()
        win_rate = (s > 0).mean() * 100
        avg_win = s[s > 0].mean() if (s > 0).any() else 0
        avg_loss = s[s < 0].mean() if (s < 0).any() else 0
        largest_win = s.max()
        largest_loss = s.min()
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

        colA, colB, colC, colD = st.columns(4)
        colA.metric("Win Rate", f"{win_rate:.1f}%")
        colB.metric("Profit Factor", f"{profit_factor:.2f}")
        colC.metric("Avg Win / Loss", f"{avg_win:.2f} / {avg_loss:.2f}")
        colD.metric("Largest Win / Loss", f"{largest_win:.2f} / {largest_loss:.2f}")

    else:
        st.info("Not enough closed trades to calculate analytics.")


# -------------------------------------------------
# GLOBAL PORTFOLIO PERFORMANCE (Across ALL symbols)
# -------------------------------------------------
st.header("📈 Total Portfolio Performance (All Symbols Combined)")

daily_path = get_daily_portfolio_file()
if os.path.exists(daily_path):
    df_daily = pd.read_csv(daily_path)

    if df_daily.empty:
        st.warning("Daily portfolio file is empty.")
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
    st.info("No aggregate daily portfolio file found.")