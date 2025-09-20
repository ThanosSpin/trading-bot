import os
import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Safe for headless VM
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import pytz

from config import TIMEZONE, SYMBOL
from portfolio import _trade_log_file, get_daily_portfolio_history

# Streamlit setup
st.set_page_config(page_title="Trading Bot Dashboard", layout="wide")
st.title("📊 Trading Bot Dashboard")

# Ensure SYMBOL is a list
symbols = SYMBOL if isinstance(SYMBOL, list) else [SYMBOL]

# ---------------------------
# Portfolio Summary
# ---------------------------
st.header("Portfolio Summary")
from portfolio import get_live_portfolio  # make sure this import exists

for sym in symbols:
    try:
        live_portfolio = get_live_portfolio(sym)
        cash = live_portfolio["cash"]
        shares = live_portfolio["shares"]
        last_price = live_portfolio["last_price"]
        value = cash + shares * last_price

        st.subheader(f"Portfolio Summary: {sym}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Cash ($)", f"{cash:.2f}")
        col2.metric("Shares", f"{shares:.2f}")
        col3.metric("Portfolio Value ($)", f"{value:.2f}")

    except Exception as e:
        st.warning(f"Could not fetch live portfolio for {sym}: {e}")

# ---------------------------
# Trade Logs & Daily Portfolio Performance
# ---------------------------
st.header("Trade Logs & Daily Portfolio Performance")
for sym in symbols:
    trade_log_path = _trade_log_file(sym)
    st.subheader(f"{sym} Trades & Portfolio")

    if os.path.exists(trade_log_path):
        # Load trades
        df_trades = pd.read_csv(trade_log_path)
        df_trades["timestamp"] = pd.to_datetime(df_trades["timestamp"], errors="coerce")
        df_trades = df_trades.dropna(subset=["timestamp"])

        local_tz = pytz.timezone(TIMEZONE)
        # Convert only if tz-naive
        if df_trades["timestamp"].dt.tz is None:
            df_trades["timestamp_local"] = df_trades["timestamp"].dt.tz_localize("UTC").dt.tz_convert(local_tz)
        else:
            df_trades["timestamp_local"] = df_trades["timestamp"].dt.tz_convert(local_tz)
        df_trades["timestamp_str"] = df_trades["timestamp_local"].dt.strftime("%Y-%m-%d %H:%M")

        # Show full trade log table
        st.dataframe(df_trades.sort_values("timestamp_local", ascending=False), use_container_width=True)

        # Plot daily portfolio from CSV
        df_daily = get_daily_portfolio_history(sym)
        if df_daily.empty:
            st.info(f"No daily portfolio data for {sym}.")
            continue

        df_daily["date"] = pd.to_datetime(df_daily["date"])
        # Localize if not already
        if df_daily["date"].dt.tz is None:
            df_daily["date"] = df_daily["date"].dt.tz_localize(local_tz)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df_daily["date"], df_daily["value"], marker="o", linestyle="-")
        ax.set_title(f"Portfolio Value Over Time ({sym})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value ($)")
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.2f}'))

        # X-axis formatting
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.xticks(rotation=45)
        ax.grid(True)
        plt.tight_layout()
        st.pyplot(fig)

    else:
        st.info(f"No trades found for {sym}. Run update_portfolio_data.py first.")