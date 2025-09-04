# dashboard.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import pytz
from config import PORTFOLIO_PATH
from portfolio import get_live_portfolio, _trade_log_file  # ✅ Use live data
from config import TIMEZONE ,SYMBOL

def load_trade_log(symbol):
    log_path = _trade_log_file(symbol)
    if os.path.exists(log_path):
        df = pd.read_csv(log_path, parse_dates=["timestamp"])
        local_tz = pytz.timezone(TIMEZONE)
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert(local_tz)
        return df
    return pd.DataFrame(columns=["timestamp", "action", "price", "value"])

# Streamlit app
st.set_page_config(page_title="Trading Bot Dashboard", layout="wide")
st.title("📊 Trading Bot Dashboard")

symbols = SYMBOL if isinstance(SYMBOL, list) else [SYMBOL]

for sym in symbols:
    # Live Portfolio Summary
    portfolio = get_live_portfolio(sym)
    value = portfolio["cash"] + portfolio["shares"] * portfolio["last_price"]

    st.header(f"Portfolio Summary: {sym}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Cash ($)", f"{portfolio['cash']:.2f}")
    col2.metric("Shares", f"{portfolio['shares']:.2f}")
    col3.metric("Portfolio Value ($)", f"{value:.2f}")

    # Trade Log
    st.subheader(f"Trade Log: {sym}")
    log = load_trade_log(sym)
    if not log.empty:
        log_display = log.copy()
        log_display["timestamp"] = log_display["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        st.dataframe(log_display.sort_values("timestamp", ascending=False), use_container_width=True)
    else:
        st.info(f"No trades logged for {sym} yet.")

    # Dynamic Portfolio Performance Plot
    st.subheader(f"Portfolio Performance: {sym}")
    if not log.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(log["timestamp"], log["value"], marker="o")
        ax.set_title(f"Portfolio Value Over Time ({sym})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value ($)")
        ax.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.info(f"No trades logged for {sym} yet to plot performance.")
