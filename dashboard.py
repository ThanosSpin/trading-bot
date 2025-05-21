# dashboard.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from config import PORTFOLIO_PATH
from portfolio import get_live_portfolio  # ✅ Use live data

TRADE_LOG_PATH = "data/trade_log.csv"
PLOT_PATH = "data/portfolio_performance.png"

def load_trade_log():
    if os.path.exists(TRADE_LOG_PATH):
        return pd.read_csv(TRADE_LOG_PATH, parse_dates=["timestamp"])
    return pd.DataFrame(columns=["timestamp", "action", "price"])

# Streamlit app
st.set_page_config(page_title="Trading Bot Dashboard", layout="wide")
st.title("📊 Trading Bot Dashboard")

# ✅ Live Portfolio Summary
st.header("Portfolio Summary")
portfolio = get_live_portfolio()
value = portfolio["cash"] + portfolio["shares"] * portfolio["last_price"]

col1, col2, col3 = st.columns(3)
col1.metric("Cash ($)", f"{portfolio['cash']:.2f}")
col2.metric("Shares", f"{portfolio['shares']:.2f}")
col3.metric("Portfolio Value ($)", f"{value:.2f}")

# Trade Log
st.header("Trade Log")
log = load_trade_log()
st.dataframe(log.sort_values("timestamp", ascending=False), use_container_width=True)

# Portfolio Plot
if os.path.exists(PLOT_PATH):
    st.header("Portfolio Value Over Time")
    st.image(PLOT_PATH, use_container_width=True)
else:
    st.info("Run a trade to generate portfolio plot.")