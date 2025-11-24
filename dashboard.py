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
from market import debug_market
from typing import Optional
import joblib

from config import TIMEZONE, SYMBOL, MODEL_DIR
from portfolio import (
    get_trade_log_file,
    get_daily_portfolio_file,
    get_live_portfolio
)
from trader import get_pdt_status
from model_xgb import compute_signals
from data_loader import fetch_historical_data, fetch_intraday_history

# -------------------------------------------------
# Streamlit setup
# -------------------------------------------------
st.set_page_config(page_title="Trading Bot Dashboard", layout="wide")
st.title("📊 Trading Bot Dashboard")

# -------------------------------------------------
# MARKET STATUS BOX (Big + Color Coded)
# -------------------------------------------------
st.header("🕒 Market Status")

# Fetch detailed diagnostics
m = debug_market(return_dict=True)

alpaca_flag = m.get("alpaca_is_open")
within_hours = m.get("within_hours")
is_day = m.get("is_trading_day")
open_time = m.get("market_open")
close_time = m.get("market_close")
ny_now = m.get("ny_time")

# Build visual message
if not is_day:
    status_color = "red"
    status_text = "❌ Market Closed — Not a Trading Day"
elif alpaca_flag and within_hours:
    status_color = "green"
    status_text = "✅ Market OPEN"
elif not alpaca_flag and within_hours:
    status_color = "yellow"
    status_text = "⚠️ Market Should Be OPEN — Alpaca Clock Reports CLOSED"
elif alpaca_flag and not within_hours:
    status_color = "yellow"
    status_text = "⚠️ Alpaca Says OPEN — But Market Hours Window is CLOSED"
else:
    status_color = "red"
    status_text = "❌ Market CLOSED"

# Draw Streamlit Box
st.markdown(
    f"""
    <div style="
        padding: 20px;
        border-radius: 12px;
        background-color: {status_color};
        color: white;
        font-size: 22px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    ">
        {status_text}
    </div>
    """,
    unsafe_allow_html=True,
)

# Detailed breakdown
st.subheader("Market Diagnostics")
c1, c2, c3 = st.columns(3)
c1.metric("NY Time", str(ny_now))
c2.metric("Market Opens", str(open_time))
c3.metric("Market Closes", str(close_time))

c4, c5, c6 = st.columns(3)
c4.metric("Trading Day", "Yes" if is_day else "No")
c5.metric("Within Hours", "Yes" if within_hours else "No")
c6.metric("Alpaca Clock", "OPEN" if alpaca_flag else "CLOSED")

st.caption("🔍 Decision = what main.py will use for trading.")

# Auto-refresh
REFRESH_INTERVAL = 60
st_autorefresh(interval=REFRESH_INTERVAL * 100, key="global_refresh")
st.caption(f"⏳ Auto-refreshing every {REFRESH_INTERVAL} seconds.")

# Normalize SYMBOL into list
symbols = SYMBOL if isinstance(SYMBOL, list) else [SYMBOL]
tz = pytz.timezone(TIMEZONE)

# -------------------------------------------------
# Helper: safely extract Close column
# -------------------------------------------------
def _get_close_series(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    if "Close" in df.columns:
        s = df["Close"]
        return s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s
    if isinstance(df.columns, pd.MultiIndex):
        try:
            s = df["Close"]
            return s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s
        except KeyError:
            pass
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            s = df[col]
            return s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s
    return None

# -------------------------------------------------
# Helper: load model info
# -------------------------------------------------
def load_model_info(symbol: str, mode: str) -> Optional[dict]:
    path = os.path.join(MODEL_DIR, f"{symbol}_{mode}_xgb.pkl")
    if not os.path.exists(path):
        return None
    try:
        data = joblib.load(path)
    except Exception as e:
        st.caption(f"{symbol} {mode} model load error — {e}")
        return None
    return {"metrics": data.get("metrics", {}), "trained_at": data.get("trained_at")}

# -------------------------------------------------
# PORTFOLIO SUMMARY
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
# MODEL SIGNALS
# -------------------------------------------------
st.header("📡 Model Signals & Price Charts")

for sym in symbols:
    st.subheader(f"Signals for {sym}")
    try:
        sig = compute_signals(sym, lookback_minutes=300, intraday_weight=0.65, resample_to="5min")
    except Exception as e:
        st.warning(f"{sym}: error computing signals — {e}")
        continue

    if not sig or sig.get("final_prob") is None:
        st.info(f"{sym}: No valid prediction available.")
        continue

    daily_p = sig.get("daily_prob")
    intra_p = sig.get("intraday_prob")
    final_p = sig.get("final_prob")
    weight = sig.get("intraday_weight", 0.65)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Final prob_up", f"{final_p:.3f}")
    col2.metric("Daily model", f"{daily_p:.3f}" if daily_p is not None else "N/A")
    col3.metric("Intraday model", f"{intra_p:.3f}" if intra_p is not None else "N/A")
    col4.metric("Intraday weight", f"{weight:.2f}")
    st.progress(max(0.0, min(final_p, 1.0)))

    # -------------------------------------------------
    # MODEL VALIDATION + FRESHNESS
    # -------------------------------------------------
    with st.expander(f"📘 {sym} Model Validation & Freshness"):

        info_daily = load_model_info(sym, "daily")
        info_intra = load_model_info(sym, "intraday")

        c_md1, c_md2 = st.columns(2)

        def show_model_block(container, label, info):
            container.markdown(f"### **{label} Model**")
            if not info:
                container.caption("No saved model found.")
                return None

            metrics = info.get("metrics", {})
            trained_at = info.get("trained_at")

            if trained_at:
                container.caption(f"Trained at: `{trained_at}`")

            age_days = None
            if trained_at:
                try:
                    t = pd.to_datetime(trained_at)
                    age_days = (pd.Timestamp.utcnow() - t).days
                    if age_days > 90:
                        status = "❌ **STALE — Retrain ASAP (>90 days)**"
                        color = "red"
                    elif age_days > 30:
                        status = "⚠️ **Aging — Retrain Recommended (>30 days)**"
                        color = "orange"
                    else:
                        status = "🟢 Fresh ✓"
                        color = "green"

                    container.markdown(
                        f"""
                        <div style="
                            padding:10px;
                            border-radius:8px;
                            background-color:{color};
                            color:white;
                            font-weight:bold;">
                            {status} (Age: {age_days} days)
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                except:
                    pass

            for key in ["accuracy", "logloss", "roc_auc", "precision", "recall", "f1"]:
                container.write(f"- **{key}**: `{metrics.get(key)}`")

            cm = metrics.get("confusion_matrix")
            if cm:
                container.write(f"- **Confusion Matrix**: `{cm}`")

            return {"accuracy": metrics.get("accuracy"), "logloss": metrics.get("logloss"), "age": age_days}

        daily_stats = show_model_block(c_md1, "Daily", info_daily)
        intra_stats = show_model_block(c_md2, "Intraday", info_intra)

        if "model_compare" not in st.session_state:
            st.session_state["model_compare"] = {}
        st.session_state["model_compare"][sym] = {"daily": daily_stats, "intraday": intra_stats}

# -------------------------------------------------
# GLOBAL MODEL COMPARISON — NOW HIDE/SHOW
# -------------------------------------------------
st.header("📊 Model Comparison Across Symbols")

show_compare = st.checkbox("Show model comparison charts", value=False)

model_compare = st.session_state.get("model_compare", {})

if show_compare and model_compare:

    chart_data = []
    for sym, vals in model_compare.items():
        d = vals.get("daily")
        i = vals.get("intraday")
        if d:
            chart_data.append([sym, "Daily", d["accuracy"], d["logloss"]])
        if i:
            chart_data.append([sym, "Intraday", i["accuracy"], i["logloss"]])

    df_chart = pd.DataFrame(chart_data, columns=["Symbol", "Mode", "Accuracy", "Logloss"])

    colA, colB = st.columns(2)

    # Accuracy Chart
    with colA:
        st.subheader("Accuracy Comparison")
        fig, ax = plt.subplots(figsize=(5, 3))
        for mode in ["Daily", "Intraday"]:
            sub = df_chart[df_chart["Mode"] == mode]
            ax.bar(sub["Symbol"] + " (" + mode + ")", sub["Accuracy"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.set_xticklabels(sub["Symbol"], rotation=45)
        ax.grid(True, alpha=0.2)
        st.pyplot(fig)

    # Logloss Chart
    with colB:
        st.subheader("Logloss Comparison")
        fig, ax = plt.subplots(figsize=(5, 3))
        for mode in ["Daily", "Intraday"]:
            sub = df_chart[df_chart["Mode"] == mode]
            ax.bar(sub["Symbol"] + " (" + mode + ")", sub["Logloss"])
        ax.set_ylabel("Logloss")
        ax.set_xticklabels(sub["Symbol"], rotation=45)
        ax.grid(True, alpha=0.2)
        st.pyplot(fig)

    # Price charts…
    for sym in symbols:
        st.subheader(f"📈 Price Charts for {sym}")

        col_price1, col_price2 = st.columns(2)

        with col_price1:
            try:
                df_daily = fetch_historical_data(sym, period="6mo", interval="1d")
                if df_daily is not None and not df_daily.empty:
                    s = _get_close_series(df_daily)
                    if s is not None:
                        fig, ax = plt.subplots(figsize=(5, 3))
                        ax.plot(s.index, s.values, marker="o", linewidth=1)
                        ax.set_title(f"{sym} Daily Close (6 months)")
                        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                    else:
                        st.caption("Daily Close column missing.")
            except Exception as e:
                st.error(f"Daily chart error: {e}")

        with col_price2:
            try:
                df_intra = fetch_intraday_history(sym, lookback_minutes=300)
                if df_intra is not None and not df_intra.empty:
                    s = _get_close_series(df_intra)
                    if s is not None:
                        fig, ax = plt.subplots(figsize=(5, 3))
                        ax.plot(s.index, s.values, linewidth=1)
                        ax.set_title(f"{sym} Intraday Close")
                        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                    else:
                        st.caption("Intraday Close column missing.")
            except Exception as e:
                st.error(f"Intraday chart error: {e}")

# -------------------------------------------------
# TRADE LOGS & TRADE ANALYTICS
# -------------------------------------------------
st.header("💼 Trade Logs & Analytics")

for sym in symbols:
    st.subheader(f"🔎 Trades for {sym}")
    path = get_trade_log_file(sym)

    if not os.path.exists(path):
        st.info(f"No trade log found for {sym}.")
        continue

    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["local_time"] = df["timestamp"].dt.tz_convert(tz)
    df["local_str"] = df["local_time"].dt.strftime("%Y-%m-%d %H:%M:%S")

    st.dataframe(df.sort_values("local_time", ascending=False), use_container_width=True)

    # ----------------------------------
    # VALUE OVER TIME PER SYMBOL
    # ----------------------------------
    if "value" in df.columns:
        try:
            dfp = df.sort_values("local_time").copy()
            dfp["date_only"] = dfp["local_time"].dt.normalize()

            fig, ax = plt.subplots(figsize=(7, 3))
            ax.plot(dfp["date_only"], dfp["value"], marker="o", linewidth=1)
            ax.set_title(f"{sym} Portfolio Value (per trade)")
            ax.set_xlabel("Date")
            ax.set_ylabel("Value ($)")
            ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.2f}"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            plt.xticks(rotation=45)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not plot value chart: {e}")

    # ----------------------------------
    # TRADE ANALYTICS
    # ----------------------------------
    if not df.empty:
        df["trade_value"] = df["qty"] * df["price"]
        df["pnl"] = df.apply(
            lambda r: -r["trade_value"] if r["action"].lower() == "buy" else r["trade_value"],
            axis=1
        )

        df = df.sort_values("timestamp")

        df["shares_after"] = df["shares"]
        cycle_rows = []
        cycle_pnls = []
        prev = 0

        for _, r in df.iterrows():
            cycle_rows.append(r)
            if r["shares_after"] == prev and len(cycle_rows) >= 2:
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
            profit_factor = gross_profit / gross_loss if gross_loss != 0 else float("inf")

            cA, cB, cC, cD = st.columns(4)
            cA.metric("Win Rate", f"{win_rate:.1f}%")
            cB.metric("Profit Factor", f"{profit_factor:.2f}")
            cC.metric("Avg Win / Loss", f"{avg_win:.2f} / {avg_loss:.2f}")
            cD.metric("Largest Win / Loss", f"{largest_win:.2f} / {largest_loss:.2f}")
        else:
            st.info("Not enough closed trades to compute analytics.")

# -------------------------------------------------
# TOTAL DAILY PORTFOLIO PERFORMANCE
# -------------------------------------------------
st.header("📈 Total Portfolio Performance (All Symbols Combined)")

path = get_daily_portfolio_file()

if os.path.exists(path):
    df = pd.read_csv(path)

    if df.empty:
        st.warning("Daily portfolio is empty.")
    else:
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(tz)
        df = df.sort_values("date")

        # ----------------------
        # Metrics
        # ----------------------
        V0 = df["value"].iloc[0]
        VT = df["value"].iloc[-1]

        days = max((df["date"].iloc[-1] - df["date"].iloc[0]).days, 1)

        cumulative_return = VT / V0 - 1
        annualized_return = (VT / V0) ** (365 / days) - 1

        df["ret"] = df["value"].pct_change()
        daily_vol = df["ret"].std()

        sharpe = (
            (annualized_return - 0) / (daily_vol * (365 ** 0.5))
            if daily_vol and daily_vol > 0 else float("nan")
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Cumulative Return", f"{cumulative_return*100:.2f}%")
        c2.metric("Annualized Return", f"{annualized_return*100:.2f}%")
        c3.metric("Sharpe Ratio", f"{sharpe:.2f}")

        # ----------------------
        # Value Chart
        # ----------------------
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(df["date"], df["value"], marker="o")
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
    st.info("No daily portfolio file found. Run update_portfolio_data.py.")