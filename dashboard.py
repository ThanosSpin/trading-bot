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

# Trade Log per symbol
st.header("Trade Logs")
for sym in symbols:
    trade_log_path = _trade_log_file(sym)
    st.subheader(f"{sym} Trades")
    if os.path.exists(trade_log_path):
        df = pd.read_csv(trade_log_path, parse_dates=["timestamp"])
        local_tz = pytz.timezone(TIMEZONE)
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert(local_tz)
        df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True)
    else:
        st.info(f"No trades found for {sym}. Run a trade to generate trade log.")

for sym in symbols:
    trade_log_path = _trade_log_file(sym)
    if os.path.exists(trade_log_path):
        df = pd.read_csv(trade_log_path, parse_dates=["timestamp"])
        local_tz = pytz.timezone(TIMEZONE)
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert(local_tz)
        df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

        st.subheader(f"Portfolio Performance: {sym}")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["timestamp"], df["value"], marker="o")
        ax.set_title(f"Portfolio Value Over Time ({sym})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value ($)")
        ax.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.info(f"Run a trade for {sym} to generate portfolio plot.")