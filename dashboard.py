# dashboard.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import pytz
from config import PORTFOLIO_PATH
from portfolio import get_live_portfolio, _trade_log_file, _performance_chart_file
from config import TIMEZONE ,SYMBOL

def load_trade_log(symbol):
    log_path = _trade_log_file(symbol)
    if os.path.exists(log_path):
        df = pd.read_csv(log_path, parse_dates=["timestamp"])
        local_tz = pytz.timezone(TIMEZONE)
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert(local_tz)
        df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        return df
    return pd.DataFrame(columns=["timestamp", "action", "price", "value"])

# Streamlit app
st.set_page_config(page_title="Trading Bot Dashboard", layout="wide")
st.title("📊 Trading Bot Dashboard")

# Ensure SYMBOL is a list
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
    st.dataframe(log.sort_values("timestamp", ascending=False), use_container_width=True)

    # Portfolio Performance Plot
    plot_path = _performance_chart_file(sym)
    st.subheader(f"Portfolio Performance: {sym}")
    if os.path.exists(plot_path):
        st.image(plot_path, use_container_width=True)
    else:
        st.info(f"Run a trade for {sym} to generate portfolio plot.")