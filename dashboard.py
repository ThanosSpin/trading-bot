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
import plotly.graph_objects as go

from config import TIMEZONE, SYMBOL, MODEL_DIR, SPY_SYMBOL, INTRADAY_WEIGHT
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
# Refresh full dashboard (and therefore prob_up) every 10 minutes
REFRESH_INTERVAL = 600  # seconds
st_autorefresh(interval=REFRESH_INTERVAL * 1000, key="global_refresh")
st.caption(f"⏳ Auto-refreshing every {REFRESH_INTERVAL // 60} minutes.")

# Normalize SYMBOL into list
symbols = SYMBOL if isinstance(SYMBOL, list) else [SYMBOL]
symbols = [s.upper() for s in symbols]

# Optional toggle to include SPY in dashboard
include_spy = st.checkbox(f"Include {SPY_SYMBOL} in dashboard", value=False)

def _has_model(sym: str) -> bool:
    return (
        os.path.exists(os.path.join(MODEL_DIR, f"{sym}_daily_xgb.pkl")) or
        os.path.exists(os.path.join(MODEL_DIR, f"{sym}_intraday_xgb.pkl"))
    )

def _has_position(sym: str) -> bool:
    try:
        lp = get_live_portfolio(sym)
        return float(lp.get("shares", 0.0)) > 0
    except Exception:
        return False

spy = SPY_SYMBOL.upper()

# Auto-include SPY if relevant, or via toggle
if include_spy or spy in symbols or _has_model(spy) or _has_position(spy):
    if spy not in symbols:
        symbols.append(spy)
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
# Signal history helpers (robust)
# -------------------------------------------------
def _signal_history_paths(sym: str):
    sym = sym.upper()
    return [
        os.path.join("logs", f"signals_{sym}.csv"),
        os.path.join("data", f"signals_{sym}.csv"),
        os.path.join(os.getcwd(), "logs", f"signals_{sym}.csv"),
        os.path.join(os.getcwd(), "data", f"signals_{sym}.csv"),
    ]

def load_signal_history(sym: str) -> Optional[pd.DataFrame]:
    """
    Load signals history from whichever location exists.
    Returns df or None.
    """
    for p in _signal_history_paths(sym):
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                # normalize timestamp
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
                    df = df.dropna(subset=["timestamp"])
                return df
            except Exception as e:
                st.warning(f"{sym}: Failed reading signal history ({p}): {e}")
                return None
    return None

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
        sig = compute_signals(sym, lookback_minutes=2400, intraday_weight=INTRADAY_WEIGHT, resample_to="15min")
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

    # -----------------------------
    # Regime badge + quick intraday diagnostics
    # -----------------------------
    model_used = sig.get("intraday_model_used") or "intraday"

    # Badge mapping
    mu = str(model_used).lower()
    if "mom" in mu:
        regime_text = "📈 Momentum intraday"
        regime_color = "#1f77b4"
    elif "mr" in mu:
        regime_text = "↩️ Mean-reversion intraday"
        regime_color = "#ff7f0e"
    else:
        regime_text = f"🧠 Intraday (legacy): {model_used}"
        regime_color = "#6c757d"

    st.markdown(
        f"""
        <div style="
            display:inline-block;
            padding:6px 10px;
            border-radius:999px;
            background:{regime_color};
            color:white;
            font-weight:600;
            font-size:13px;
            margin-top:6px;
            margin-bottom:6px;
        ">
            {regime_text}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Optional: show vol/mom inline (super useful when debugging regime switches)
    vol = sig.get("intraday_vol")
    mom = sig.get("intraday_mom")
    if vol is not None or mom is not None:
        vol_s = "N/A" if vol is None else f"{float(vol):.5f}"
        mom_s = "N/A" if mom is None else f"{float(mom)*100:.2f}%"
        st.caption(f"Intraday diagnostics → vol={vol_s} | mom(≈2h)={mom_s}")

    # -----------------------------
    # Tiny improvement: show dp/ip divergence + which intraday model was used
    # -----------------------------
    model_used = sig.get("intraday_model_used") or sig.get("model") or "intraday"
    div = None
    if daily_p is not None and intra_p is not None:
        try:
            div = float(intra_p - daily_p)
        except Exception:
            div = None

    if div is not None:
        st.caption(f"Δ (ip - dp) = {div:+.3f} | intraday model: {model_used}")
    else:
        st.caption(f"intraday model: {model_used}")

    pretty_model = {
        "intraday_mom": "📈 Momentum intraday",
        "intraday_mr": "↩️ Mean-reversion intraday",
        "intraday": "🧠 Legacy intraday",
    }.get(model_used, model_used)

    # -----------------------------
    # Store points for divergence chart (session-level, bounded, dedup per refresh)
    # -----------------------------
    if "divergence_points" not in st.session_state:
        st.session_state["divergence_points"] = []

    # Use refresh key so we don't append duplicates on Streamlit reruns
    refresh_key = f"{sym}:{st.session_state.get('global_refresh', 0)}"
    if "divergence_seen" not in st.session_state:
        st.session_state["divergence_seen"] = set()

    if refresh_key not in st.session_state["divergence_seen"]:
        st.session_state["divergence_seen"].add(refresh_key)

        st.session_state["divergence_points"].append({
            "time": pd.Timestamp.utcnow(),
            "symbol": sym,
            "dp": float(daily_p) if daily_p is not None else None,
            "ip": float(intra_p) if intra_p is not None else None,
            "divergence": float(div) if div is not None else None,
            "weight": float(weight) if weight is not None else None,
            "model": pretty_model,
        })

        # keep last N points overall (prevents memory growth)
        MAX_POINTS = 500
        if len(st.session_state["divergence_points"]) > MAX_POINTS:
            st.session_state["divergence_points"] = st.session_state["divergence_points"][-MAX_POINTS:]

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
# DIVERGENCE: dp vs ip
# -------------------------------------------------
st.subheader("📉 Daily vs Intraday Divergence (ip - dp)")

pts = st.session_state.get("divergence_points", [])
if not pts:
    st.info("No divergence points yet.")
else:
    dfd = pd.DataFrame(pts)

    # safety: ensure columns exist
    for col in ["time", "symbol", "dp", "ip", "divergence", "weight", "model"]:
        if col not in dfd.columns:
            dfd[col] = None

    dfd["time"] = pd.to_datetime(dfd["time"], utc=True, errors="coerce")
    dfd = dfd.dropna(subset=["time", "symbol"]).sort_values("time")

    # Optional: keep last N points for plotting
    dfd = dfd.tail(300)

    # show latest snapshot table
    latest = (
        dfd.sort_values("time")
           .groupby("symbol", as_index=False)
           .tail(1)
           .copy()
    )

    latest = latest[["symbol", "dp", "ip", "divergence", "weight", "model"]].sort_values("symbol")
    st.dataframe(latest, use_container_width=True)

    # plot divergence over time
    fig = go.Figure()
    for sym in sorted(dfd["symbol"].dropna().unique()):
        sub = dfd[dfd["symbol"] == sym].copy()
        # skip symbols with no divergence values yet
        sub = sub.dropna(subset=["divergence"])
        if sub.empty:
            continue

        fig.add_trace(go.Scatter(
            x=sub["time"],
            y=sub["divergence"],
            mode="lines+markers",
            name=sym,
            hovertemplate=(
                "<b>%{x|%Y-%m-%d %H:%M:%S} UTC</b><br>"
                "ip - dp: %{y:.3f}<extra></extra>"
            ),
        ))

    # zero line
    fig.add_hline(y=0, line_width=1, line_dash="dash")

    fig.update_layout(
        height=360,
        template="plotly_white",
        xaxis_title="Time (UTC)",
        yaxis_title="ip - dp",
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

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
    # ----------------------------------
    # VALUE OVER TIME PER SYMBOL (Plotly)
    # ----------------------------------
    if "value" in df.columns:
        try:
            dfp = df.sort_values("local_time").copy()
            # Interactive Plotly line + markers
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=dfp["local_time"],
                    y=dfp["value"],
                    mode="lines+markers",
                    name=f"{sym} Equity",
                    hovertemplate=(
                        "<b>%{x|%Y-%m-%d %H:%M:%S}</b><br>"
                        "Value: $%{y:,.2f}<extra></extra>"
                    ),
                )
            )

            fig.update_layout(
                title=f"{sym} Portfolio Value (per trade)",
                xaxis_title="Time",
                yaxis_title="Value ($)",
                hovermode="x unified",
                height=350,
                template="plotly_white",
            )

            # optional: show y as dollars with thousands separator
            fig.update_yaxes(tickprefix="$", separatethousands=True)

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.warning(f"Could not plot value chart: {e}")

    # ----------------------------------
    # TRADE ANALYTICS (bullet-proof: derive exec_qty from shares_before/after)
    # ----------------------------------
    if not df.empty:
        df = df.copy()

        # ---- coerce types safely ----
        df["action"] = df["action"].astype(str).str.lower().str.strip()
        df["price"] = pd.to_numeric(df.get("price"), errors="coerce")
        df["timestamp"] = pd.to_datetime(df.get("timestamp"), utc=True, errors="coerce")

        # shares column in your CSV is "shares AFTER this trade"
        df["shares"] = pd.to_numeric(df.get("shares"), errors="coerce")

        # optional new columns if you add them later
        if "shares_before" in df.columns:
            df["shares_before"] = pd.to_numeric(df["shares_before"], errors="coerce")
        if "shares_after" in df.columns:
            df["shares_after"] = pd.to_numeric(df["shares_after"], errors="coerce")

        df = df.dropna(subset=["timestamp", "action", "price", "shares"])

        # ---- build shares_before / shares_after ----
        df = df.sort_values("timestamp").copy()

        if "shares_after" not in df.columns:
            df["shares_after"] = df["shares"]

        if "shares_before" not in df.columns:
            df["shares_before"] = df["shares_after"].shift(1).fillna(0.0)

        # ---- derive executed quantity from share deltas ----
        # This fixes logs where qty is wrong (e.g., SPY sell qty=0).
        def _exec_qty(r):
            sb = float(r["shares_before"])
            sa = float(r["shares_after"])
            if r["action"] == "buy":
                return max(0.0, sa - sb)
            if r["action"] == "sell":
                return max(0.0, sb - sa)
            return 0.0

        df["exec_qty"] = df.apply(_exec_qty, axis=1)

        # drop rows that don’t change position (bad logs / duplicates / no fills)
        df = df[df["exec_qty"] > 0].copy()

        if df.empty:
            st.info("No filled trades detected (position never changed).")
        else:
            # ---- cashflow from executed qty ----
            # BUY consumes cash (negative), SELL returns cash (positive)
            df["cashflow"] = df.apply(
                lambda r: -(r["exec_qty"] * r["price"]) if r["action"] == "buy"
                else +(r["exec_qty"] * r["price"]) if r["action"] == "sell"
                else 0.0,
                axis=1,
            )

            # ---- cycle detection: flat -> in position -> flat ----
            EPS = 1e-9
            cycle_pnls = []
            in_cycle = False
            running = 0.0

            for _, r in df.iterrows():
                sb = float(r["shares_before"])
                sa = float(r["shares_after"])
                cf = float(r["cashflow"])

                was_flat = abs(sb) <= EPS
                now_flat = abs(sa) <= EPS

                # start cycle
                if (not in_cycle) and was_flat and (not now_flat):
                    in_cycle = True
                    running = 0.0

                if in_cycle:
                    running += cf

                # end cycle (position fully closed)
                if in_cycle and (not was_flat) and now_flat:
                    cycle_pnls.append(running)
                    in_cycle = False
                    running = 0.0

            if not cycle_pnls:
                st.info("Not enough closed trades (need flat → position → flat).")
            else:
                s = pd.Series(cycle_pnls, dtype=float)

                gross_profit = float(s[s > 0].sum())
                gross_loss = float(-s[s < 0].sum())  # positive number

                win_rate = float((s > 0).mean() * 100.0)
                avg_win = float(s[s > 0].mean()) if (s > 0).any() else 0.0
                avg_loss = float(s[s < 0].mean()) if (s < 0).any() else 0.0
                largest_win = float(s.max())
                largest_loss = float(s.min())

                # Profit factor: handle no-loss case cleanly
                if gross_loss > 0:
                    profit_factor = gross_profit / gross_loss
                    pf_str = f"{profit_factor:.2f}"
                else:
                    profit_factor = float("inf")
                    pf_str = "∞"

                cA, cB, cC, cD = st.columns(4)
                cA.metric("Win Rate", f"{win_rate:.1f}%")
                cB.metric("Profit Factor", pf_str)
                cC.metric("Avg Win / Loss", f"{avg_win:.2f} / {avg_loss:.2f}")
                cD.metric("Largest Win / Loss", f"{largest_win:.2f} / {largest_loss:.2f}")

                with st.expander("🔍 Closed-trade cycle PnLs (debug)"):
                    st.dataframe(pd.DataFrame({"cycle_pnl": s}))

# -------------------------------------------------
# Price vs Model PERFORMANCE
# -------------------------------------------------
st.header("📈 Price vs Model Probability")

for sym in symbols:
    path = f"logs/signals_{sym}.csv"
    if not os.path.exists(path):
        st.info(f"No signal history for {sym}")
        continue

    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").tail(300)

    fig = go.Figure()

    # ---- PRICE (left axis)
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["price"],
        name="Price",
        line=dict(color="black", width=2),
        yaxis="y1",
        hovertemplate="Price: %{y:.2f}<extra></extra>",
    ))

    # ---- FINAL PROBABILITY (right axis)
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["final_prob"],
        name="Final Probability",
        line=dict(color="blue", width=2),
        yaxis="y2",
        hovertemplate="Prob: %{y:.3f}<extra></extra>",
    ))

    # Optional: intraday vs daily
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["intraday_prob"],
        name="Intraday Prob",
        line=dict(color="orange", dash="dot"),
        yaxis="y2",
        opacity=0.6,
    ))

    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["daily_prob"],
        name="Daily Prob",
        line=dict(color="green", dash="dash"),
        yaxis="y2",
        opacity=0.6,
    ))

    fig.update_layout(
        title=f"{sym} — Price vs Probability",
        height=400,
        hovermode="x unified",
        template="plotly_white",
        yaxis=dict(title="Price"),
        yaxis2=dict(
            title="Probability",
            overlaying="y",
            side="right",
            range=[0, 1],
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# TOTAL DAILY PORTFOLIO PERFORMANCE (DUAL EQUITY CURVES)
# -------------------------------------------------
st.header("📈 Total Portfolio Performance (All Symbols Combined)")

show_bot_only = st.checkbox("Show Bot-Only Equity (PnL curve)", value=True)
show_total_equity = st.checkbox("Show Total Equity (includes deposits/withdrawals)", value=True)
show_markers = st.checkbox("Show deposit/withdrawal markers", value=True)

# ---- Load daily portfolio PnL file ----
portfolio_path = get_daily_portfolio_file()
deposit_path = os.path.join(os.path.dirname(portfolio_path), "deposits.csv")

df_dep = None
if os.path.exists(deposit_path):
    df_dep = pd.read_csv(deposit_path)
    df_dep["date"] = pd.to_datetime(df_dep["date"], utc=True)
else:
    st.info("ℹ️ No deposits.csv found — only showing PnL performance.")

if os.path.exists(portfolio_path):
    df = pd.read_csv(portfolio_path)

    if df.empty:
        st.warning("Daily portfolio is empty.")
    else:
        # ---- Normalize ----
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(tz)
        df = df.sort_values("date")

        # PnL-only curve
        df["pnl_value"] = df["value"]
        V0 = df["pnl_value"].iloc[0]
        VT = df["pnl_value"].iloc[-1]

        days = max((df["date"].iloc[-1] - df["date"].iloc[0]).days, 1)
        cumulative_return = VT / V0 - 1
        annualized_return = (VT / V0) ** (365 / days) - 1

        df["ret"] = df["pnl_value"].pct_change()
        daily_vol = df["ret"].std()
        sharpe = (
            (annualized_return - 0) / (daily_vol * (365 ** 0.5))
            if daily_vol and daily_vol > 0 else float("nan")
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Cumulative Return (PnL-only)", f"{cumulative_return*100:.2f}%")
        c2.metric("Annualized Return (PnL-only)", f"{annualized_return*100:.2f}%")
        c3.metric("Sharpe Ratio (PnL-only)", f"{sharpe:.2f}")

        # Total equity (adds deposits/withdrawals)
        df["total_equity"] = df["pnl_value"]
        if df_dep is not None:
            df_dep = df_dep.sort_values("date")
            for _, row in df_dep.iterrows():
                dep_date = row["date"]
                amount = row["amount"]
                df.loc[df["date"] >= dep_date, "total_equity"] += amount

        # ---------------------------------------
        # 📈 Dual Curve Chart + Markers (Plotly)
        # ---------------------------------------

        fig = go.Figure()

        # --- PnL-only curve ---
        if show_bot_only:
            fig.add_trace(go.Scatter(
                x=df["date"],
                y=df["pnl_value"],
                mode="lines",
                name="PnL-Only Equity",
                line=dict(width=2),
                hovertemplate=(
                    "<b>PnL-Only Equity</b><br>"
                    "Date: %{x|%Y-%m-%d}<br>"
                    "Value: $%{y:,.2f}"
                    "<extra></extra>"
                ),
            ))

        # --- Total equity (with deposits) ---
        if show_total_equity:
            fig.add_trace(go.Scatter(
                x=df["date"],
                y=df["total_equity"],
                mode="lines",
                name="Total Equity (with deposits)",
                line=dict(width=2, dash="dash"),
                hovertemplate=(
                    "<b>Total Equity</b><br>"
                    "Date: %{x|%Y-%m-%d}<br>"
                    "Value: $%{y:,.2f}"
                    "<extra></extra>"
                ),
            ))

        # Determine dynamic Y-axis label
        if show_bot_only and not show_total_equity:
            y_axis_label = "PnL-Only Equity ($)"
        elif show_total_equity and not show_bot_only:
            y_axis_label = "Total Equity ($)"
        elif show_bot_only and show_total_equity:
            y_axis_label = "Value ($)"        # both curves displayed
        else:
            y_axis_label = ""  # no curves selected

        # ---- Deposit / Withdrawal Markers ----
        if df_dep is not None and not df_dep.empty:
            target_tz = df["date"].dt.tz

            has_dep_legend = False
            has_wdr_legend = False

            for _, row in df_dep.iterrows():
                dep_t = row["date"]
                amount = float(row["amount"])

                if target_tz is not None:
                    if dep_t.tzinfo is None:
                        dep_t = dep_t.tz_localize("UTC").tz_convert(target_tz)
                    else:
                        dep_t = dep_t.tz_convert(target_tz)

                nearest_idx = (df["date"] - dep_t).abs().idxmin()
                nearest_date = df.loc[nearest_idx, "date"]
                nearest_value = df.loc[nearest_idx, "total_equity"]

                is_dep = amount > 0
                name = "Deposit" if is_dep else "Withdrawal"

                showlegend = False
                if is_dep and not has_dep_legend:
                    showlegend = True
                    has_dep_legend = True
                elif not is_dep and not has_wdr_legend:
                    showlegend = True
                    has_wdr_legend = True

                fig.add_trace(go.Scatter(
                    x=[nearest_date],
                    y=[nearest_value],
                    mode="markers",
                    marker=dict(
                        size=11,
                        color="green" if is_dep else "red",
                        symbol="triangle-up" if is_dep else "triangle-down",
                        line=dict(width=1, color="black"),
                    ),
                    name=name,
                    showlegend=showlegend,
                    hovertemplate=(
                        f"<b>{name}</b><br>"
                        f"Amount: ${amount:,.2f}<br>"
                        "Date: %{x|%Y-%m-%d}<br>"
                        "Equity: $%{y:,.2f}"
                        "<extra></extra>"
                    )
                ))

        # Layout with dynamic Y-label
        fig.update_layout(
            title="PnL vs Total Account Equity",
            xaxis_title="Date",
            yaxis_title=y_axis_label,
            hovermode="x unified",
            template="plotly_white",
            height=450,
        )

        st.plotly_chart(fig, use_container_width=True)

        # ---------------------------------------
        # 💵 Deposits / Withdrawals Log
        # ---------------------------------------
        if df_dep is not None:
            st.subheader("💵 Deposits / Withdrawals Log")

            # Convert timezone → strip time → format YYYY-MM-DD
            df_dep["display_date"] = (
                df_dep["date"]
                    .dt.tz_convert(tz)
                    .dt.normalize()
                    .dt.strftime("%Y-%m-%d")
            )

            st.dataframe(
                df_dep[["display_date", "amount"]]
                    .rename(columns={"display_date": "Date", "amount": "Amount ($)"}),
                use_container_width=True
            )

else:
    st.info("No daily portfolio file found. Run update_portfolio_data.py.")