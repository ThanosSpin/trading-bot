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
decision = m.get("decision")

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
st_autorefresh(interval=REFRESH_INTERVAL * 1000, key="global_refresh")
st.caption(f"⏳ Auto-refreshing every {REFRESH_INTERVAL} seconds.")

# Normalize SYMBOL into list
symbols = SYMBOL if isinstance(SYMBOL, list) else [SYMBOL]
tz = pytz.timezone(TIMEZONE)


# -------------------------------------------------
# Small helper: safely get Close series from yfinance / Alpaca DF
# -------------------------------------------------
def _get_close_series(df: pd.DataFrame) -> Optional[pd.Series]:
    """Return a 1D 'Close' price series from normal or MultiIndex DataFrame."""
    if df is None or df.empty:
        return None

    # Direct column
    if "Close" in df.columns:
        s = df["Close"]
        if isinstance(s, pd.DataFrame):
            return s.iloc[:, 0]
        return s

    # MultiIndex: use first-level "Close"
    if isinstance(df.columns, pd.MultiIndex):
        try:
            s = df["Close"]  # sub-dataframe with all close columns
            if isinstance(s, pd.DataFrame):
                return s.iloc[:, 0]
        except KeyError:
            pass

    # Fallback: first numeric column
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            if isinstance(s, pd.DataFrame):
                return s.iloc[:, 0]
            return s

    return None


# -------------------------------------------------
# Helper: load model metrics from saved .pkl
# -------------------------------------------------
def load_model_info(symbol: str, mode: str) -> Optional[dict]:
    """
    Load saved model info (metrics + trained_at) from models/{symbol}_{mode}_xgb.pkl
    Returns:
        {
          "metrics": {...},
          "trained_at": str or None
        }
    or None if missing / error.
    """
    path = os.path.join(MODEL_DIR, f"{symbol}_{mode}_xgb.pkl")
    if not os.path.exists(path):
        return None

    try:
        data = joblib.load(path)
    except Exception as e:
        st.caption(f"{symbol} {mode} model load error — {e}")
        return None

    metrics = data.get("metrics", {}) or {}
    trained_at = data.get("trained_at")
    return {"metrics": metrics, "trained_at": trained_at}


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
# MODEL SIGNALS (Daily + Intraday) + PRICE CHARTS
# -------------------------------------------------
st.header("📡 Model Signals & Price Charts")

for sym in symbols:
    st.subheader(f"Signals for {sym}")

    # -------------------------------
    # Compute signals
    # -------------------------------
    try:
        sig = compute_signals(
            sym,
            lookback_minutes=300,   # richer intraday window
            intraday_weight=0.65,
            resample_to="5min",
        )
    except Exception as e:
        st.warning(f"{sym}: error computing signals — {e}")
        continue

    if not sig or sig.get("final_prob") is None:
        st.info(f"{sym}: No valid prediction available.")
        continue

    daily_p = sig.get("daily_prob")
    intra_p = sig.get("intraday_prob")
    final_p = sig.get("final_prob")

    # If compute_signals has adaptive weighting, use that;
    # otherwise fall back to the base 0.65 you passed in.
    weight = sig.get("intraday_weight", 0.65)
    allow_intraday = sig.get("allow_intraday", True)

    # -------------------------------
    # Summary metrics
    # -------------------------------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Final prob_up", f"{final_p:.3f}")
    col2.metric("Daily model", f"{daily_p:.3f}" if daily_p is not None else "N/A")

    intra_label = "Intraday model"
    if not allow_intraday:
        intra_label += " (market closed)"
    elif intra_p is not None:
        intra_label += " (active)"

    col3.metric(intra_label, f"{intra_p:.3f}" if intra_p is not None else "N/A")
    col4.metric("Intraday weight", f"{weight:.2f}")

    st.progress(max(0.0, min(final_p, 1.0)))  # visual blending bar

    # -------------------------------
    # MODEL VALIDATION STATS
    # -------------------------------
        # -------------------------------
    # MODEL VALIDATION STATS + FRESHNESS CHECK
    # -------------------------------
    with st.expander(f"{sym} Model Validation (Backtest Metrics)", expanded=False):

        info_daily = load_model_info(sym, "daily")
        info_intra = load_model_info(sym, "intraday")

        c_md1, c_md2 = st.columns(2)

        def show_model_block(container, label, info):
            container.markdown(f"### **{label} Model**")

            if not info:
                container.caption("No saved model or metrics found.")
                return None

            metrics = info.get("metrics", {})
            trained_at = info.get("trained_at", None)
            container.caption(f"Trained at: `{trained_at}`")

            # -------------------------
            # FRESHNESS CHECK
            # -------------------------
            age_days = None
            freshness_status = "unknown"

            if trained_at:
                try:
                    t = pd.to_datetime(trained_at)
                    age_days = (pd.Timestamp.utcnow() - t).days

                    if age_days > 90:
                        freshness_status = "❌ **STALE — Retrain ASAP (>90 days)**"
                        color = "red"
                    elif age_days > 30:
                        freshness_status = "⚠️ **Old — Retrain Recommended (>30 days)**"
                        color = "orange"
                    else:
                        freshness_status = "✅ Model Fresh"
                        color = "green"

                    container.markdown(
                        f"""
                        <div style="
                            padding:8px;
                            border-radius:8px;
                            background-color:{color};
                            color:white;
                            font-weight:bold;">
                            {freshness_status}
                            (Age: {age_days} days)
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                except Exception:
                    pass

            # -------------------------
            # METRICS LIST
            # -------------------------
            for key in ["accuracy", "logloss", "roc_auc", "precision", "recall", "f1"]:
                val = metrics.get(key)
                container.write(f"- {key}: `{val}`")

            cm = metrics.get("confusion_matrix")
            if cm:
                container.write(f"- Confusion Matrix: `{cm}`")

            return {
                "accuracy": metrics.get("accuracy"),
                "logloss": metrics.get("logloss"),
                "age": age_days,
            }

        # Render Daily + Intraday
        daily_stats = show_model_block(c_md1, "Daily", info_daily)
        intra_stats = show_model_block(c_md2, "Intraday", info_intra)

        # Store for global comparison (append into session if needed)
        if "model_compare" not in st.session_state:
            st.session_state["model_compare"] = {}

        st.session_state["model_compare"][sym] = {
            "daily": daily_stats,
            "intraday": intra_stats,
        }

    # -------------------------------
    # Price charts: Daily + Intraday
    # -------------------------------
    c_price1, c_price2 = st.columns(2)

    # Daily price (6 months)
    with c_price1:
        try:
            df_daily = fetch_historical_data(sym, period="6mo", interval="1d")
            if df_daily is not None and not df_daily.empty:
                close_series = _get_close_series(df_daily)
                if close_series is not None:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.plot(close_series.index, close_series.values, marker="o", linewidth=1)
                    ax.set_title(f"{sym} Daily Close (6M)")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Price")
                    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                    plt.xticks(rotation=45)
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.caption(f"{sym}: could not resolve daily Close series.")
            else:
                st.caption(f"{sym}: no daily data available.")
        except Exception as e:
            st.caption(f"{sym}: daily chart error — {e}")

    # Intraday price (last 300 bars @1m, resampled as-is)
    with c_price2:
        try:
            df_intra = fetch_intraday_history(sym, lookback_minutes=300, interval="1m")
            if df_intra is not None and not df_intra.empty:
                close_series = _get_close_series(df_intra)
                if close_series is not None:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.plot(close_series.index, close_series.values, linewidth=1)
                    ax.set_title(f"{sym} Intraday Close (last ~{len(close_series)} min)")
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Price")
                    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
                    plt.xticks(rotation=45)
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.caption(f"{sym}: could not resolve intraday Close series.")
            else:
                st.caption(f"{sym}: no intraday data available.")
        except Exception as e:
            st.caption(f"{sym}: intraday chart error — {e}")

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
    # Equity / value over time chart for this symbol
    # ---------------------------
    if "value" in df_trades.columns:
        try:
            df_plot = df_trades.sort_values("timestamp_local").copy()

            # collapse to date-only on x-axis
            df_plot["date_only"] = df_plot["timestamp_local"].dt.normalize()

            fig, ax = plt.subplots(figsize=(7, 3))
            ax.plot(df_plot["date_only"], df_plot["value"], marker="o", linewidth=1)
            ax.set_title(f"{sym} Portfolio Value Over Time (per-trade)")
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
            st.caption(f"{sym}: could not plot per-symbol value chart — {e}")

    # ---------------------------
    # Trade Analytics
    # ---------------------------
    if not df_trades.empty:
        df_trades["trade_value"] = df_trades["qty"] * df_trades["price"]
        df_trades["pnl"] = df_trades.apply(
            lambda row: -row["trade_value"] if row["action"].lower() == "buy" else row["trade_value"], axis=1
        )

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
# TOTAL DAILY PORTFOLIO PERFORMANCE + METRICS
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

        df_daily = df_daily.sort_values("date")

        # -----------------------------------
        # Metrics
        # -----------------------------------
        V0 = df_daily["value"].iloc[0]
        VT = df_daily["value"].iloc[-1]

        days = (df_daily["date"].iloc[-1] - df_daily["date"].iloc[0]).days
        days = max(days, 1)  # prevent divide by zero

        cumulative_return = (VT / V0) - 1
        annualized_return = (VT / V0) ** (365 / days) - 1

        # Daily returns
        df_daily["return"] = df_daily["value"].pct_change()
        daily_volatility = df_daily["return"].std()

        sharpe = (
            (annualized_return - 0.00) / (daily_volatility * (365 ** 0.5))
            if daily_volatility and daily_volatility > 0
            else float("nan")
        )

        # Display metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Cumulative Return", f"{cumulative_return*100:.2f}%")
        m2.metric("Annualized Return", f"{annualized_return*100:.2f}%")
        m3.metric("Sharpe Ratio", f"{sharpe:.2f}")

        # -----------------------------------
        # Chart
        # -----------------------------------
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