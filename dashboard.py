# dashboard.py
import os
import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Safe for headless VM
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pytz

from config import PORTFOLIO_PATH, TIMEZONE, SYMBOL
from portfolio import get_live_portfolio, _trade_log_file

# Ensure base data folder exists
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Streamlit app setup
st.set_page_config(page_title="Trading Bot Dashboard", layout="wide")
st.title("📊 Trading Bot Dashboard")

# Ensure SYMBOL is a list
symbols = SYMBOL if isinstance(SYMBOL, list) else [SYMBOL]

# ---------------------------
# Live Portfolio Summary
# ---------------------------
st.header("Portfolio Summary")
for sym in symbols:
    portfolio = get_live_portfolio(sym)
    value = portfolio["cash"] + portfolio["shares"] * portfolio["last_price"]

    st.subheader(f"Portfolio Summary: {sym}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Cash ($)", f"{portfolio['cash']:.2f}")
    col2.metric("Shares", f"{portfolio['shares']:.2f}")
    col3.metric("Portfolio Value ($)", f"{value:.2f}")

# ---------------------------
# Trade Logs & Portfolio Performance
# ---------------------------
st.header("Trade Logs & Portfolio Performance")
for sym in symbols:
    trade_log_path = _trade_log_file(sym)
    st.subheader(f"{sym} Trades & Portfolio")

    if os.path.exists(trade_log_path):
        # Read trade log once
        df = pd.read_csv(trade_log_path, parse_dates=["timestamp"])
        local_tz = pytz.timezone(TIMEZONE)
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert(local_tz)
        df["timestamp_str"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")

        # Show trade log table
        st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True)

        # Plot portfolio value
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["timestamp_str"], df["value"], marker="o")
        ax.set_title(f"Portfolio Value Over Time ({sym})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value ($)")
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.2f}'))  # Dollar formatting
        ax.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.info(f"No trades found for {sym}. Run a trade to generate trade log.")