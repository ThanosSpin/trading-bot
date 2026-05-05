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


from config.config import TIMEZONE, SYMBOL, MODEL_DIR, SPY_SYMBOL, INTRADAY_WEIGHT
from portfolio import (
    get_trade_log_file,
    get_daily_portfolio_file,
    get_live_portfolio
)
from trader import get_pdt_status
from predictive_model.model_xgb import compute_signals
from predictive_model.data_loader import fetch_historical_data, fetch_intraday_history
from plotly.subplots import make_subplots


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

if st.button("🔄 Clear cache"):
    st.cache_data.clear()
    st.rerun()

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


# Only include SPY when explicitly requested
if include_spy:
    if spy not in symbols:
        symbols.append(spy)
else:
    # Ensure SPY is removed when checkbox is off
    symbols = [s for s in symbols if s != spy]

tz = pytz.timezone(TIMEZONE)

# ── Load deposits from Alpaca account activities (CSD = Cash Deposit) ──────
# Primary source: Alpaca API  →  GET /v2/account/activities/CSD
# Fallback:       deposits.csv (manual file)
from portfolio import get_daily_portfolio_file
portfolio_path = get_daily_portfolio_file()
deposit_path = os.path.join(os.path.dirname(portfolio_path), "deposits.csv")


@st.cache_data(ttl=3600)  # cache for 1 hour — deposits don't change often
def _fetch_deposits_from_alpaca() -> pd.DataFrame:
    """
    Fetch cash deposits (CSD) and withdrawals (CSW) from Alpaca account activities.
    Returns a DataFrame with columns: date (UTC, tz-aware), amount (positive=deposit, negative=withdrawal).
    """
    try:
        import alpaca_trade_api as tradeapi
        from config.config import API_KEY, API_SECRET, BASE_URL
        if not API_KEY or not API_SECRET:
            return pd.DataFrame(columns=["date", "amount", "source"])
        api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL)
        rows = []
        for activity_type in ["CSD", "CSW"]:
            try:
                # ✅ CORRECT
                activities = api.get_activities(activity_types=activity_type)
                for a in activities:
                    amount = float(getattr(a, "net_amount", 0) or 0)
                    date_raw = getattr(a, "date", None) or getattr(a, "transaction_time", None)
                    if date_raw and amount != 0:
                        rows.append({
                            "date": pd.to_datetime(str(date_raw), utc=True),
                            "amount": amount,
                            "source": activity_type,
                        })
            except Exception:
                pass
        if rows:
            df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
            return df
    except ImportError:
        pass
    except Exception:
        pass
    return pd.DataFrame(columns=["date", "amount", "source"])


df_dep = None

# ── Try Alpaca API first ──────────────────────────────────────────────────────
try:
    _alpaca_deps = _fetch_deposits_from_alpaca()
    if not _alpaca_deps.empty:
        df_dep = _alpaca_deps[["date", "amount"]].copy()
        st.caption(f"💳 Loaded {len(df_dep)} deposit/withdrawal entries from Alpaca API.")
except Exception as _e:
    st.caption(f"⚠️ Alpaca deposit fetch failed: {_e} — trying deposits.csv")

# ── Fallback: deposits.csv ────────────────────────────────────────────────────
if df_dep is None or df_dep.empty:
    if os.path.exists(deposit_path):
        _raw = pd.read_csv(deposit_path)
        _raw["date"] = pd.to_datetime(_raw["date"], utc=True)
        df_dep = _raw[["date", "amount"]].copy()
        st.caption(f"💳 Loaded {len(df_dep)} deposit entries from deposits.csv (fallback).")
    else:
        df_dep = pd.DataFrame(columns=["date", "amount"])
        st.caption("ℹ️ No deposits.csv found and Alpaca API returned no deposit data. PnL = Raw Equity.")

# Ensure date is UTC tz-aware (defensive)
if df_dep is not None and not df_dep.empty:
    df_dep["date"] = pd.to_datetime(df_dep["date"], utc=True)
    df_dep = df_dep.sort_values("date").reset_index(drop=True)

# ── Debug: show what deposits were loaded ────────────────────────────────────
with st.expander("🔍 Deposit Debug (expand to verify deposit data)", expanded=False):
    if df_dep is None or df_dep.empty:
        st.error("❌ No deposits loaded — PnL and Raw Equity will be identical!")
        st.info("Add entries to deposits.csv  OR  ensure Alpaca API credentials are correct.")
    else:
        st.success(f"✅ {len(df_dep)} deposit/withdrawal entries loaded:")
        st.dataframe(df_dep, use_container_width=True)
# ─────────────────────────────────────────────────────────────────────────────

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
    Load signals history with robust error handling for schema changes.
    Returns df or None.
    """
    for p in _signal_history_paths(sym):
        if not os.path.exists(p):
            continue

        try:
            # Try normal load first
            df = pd.read_csv(p)

            # Normalize timestamp
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
                df = df.dropna(subset=["timestamp"])

            return df

        except pd.errors.ParserError as e:
            # Handle corrupted/mismatched schema
            st.warning(f"⚠️ {sym}: Signal log has schema mismatch. Attempting recovery...")

            try:
                # Try skipping bad lines
                df = pd.read_csv(p, on_bad_lines='skip')

                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
                    df = df.dropna(subset=["timestamp"])

                st.success(f"✅ Recovered {len(df)} valid rows for {sym}")
                return df

            except Exception as e2:
                st.error(f"❌ Could not recover {sym} signal log: {e2}")

                # Offer to delete corrupted file
                if st.button(f"🗑️ Delete corrupted signal log for {sym}", key=f"delete_signal_{sym}"):
                    try:
                        os.remove(p)
                        st.success(f"Deleted {p}. Will be regenerated on next cycle.")
                        st.rerun()
                    except Exception as e3:
                        st.error(f"Failed to delete: {e3}")

                return None

        except Exception as e:
            st.warning(f"{sym}: Failed reading signal history ({p}): {e}")
            return None

    return None

# -------------------------------------------------
# PORTFOLIO SUMMARY
# -------------------------------------------------
st.header("Portfolio Summary")

try:
    from account_cache import account_cache
    account_cache.invalidate()
    account = account_cache.get_account()

    total_equity = float(account.get("equity", 0.0) or 0.0)
    total_cash = float(account.get("cash", 0.0) or 0.0)

    raw_bp = account.get("buying_power")
    raw_regt_bp = account.get("regt_buying_power")
    raw_day_bp = account.get("daytrading_buying_power")
    raw_non_margin_bp = account.get("non_marginable_buying_power")
    raw_multiplier = account.get("multiplier")

    def _to_float(x, default=0.0):
        try:
            return float(x)
        except (TypeError, ValueError):
            return default

    buying_power = None
    bp_label = "💪 Buying Power"

    if raw_bp not in (None, "", "0", 0):
        buying_power = _to_float(raw_bp, 0.0)
        bp_label = "💪 Buying Power (Alpaca)"
    elif raw_regt_bp not in (None, "", "0", 0):
        buying_power = _to_float(raw_regt_bp, 0.0)
        bp_label = "💪 Reg T Buying Power"
    elif raw_day_bp not in (None, "", "0", 0):
        buying_power = _to_float(raw_day_bp, 0.0)
        bp_label = "💪 Day Trading Buying Power"
    else:
        buying_power = total_cash
        bp_label = "💪 Buying Power (Cash Fallback)"

    k1, k2, k3 = st.columns(3)
    k1.metric("💼 Total Equity", f"${total_equity:,.2f}")
    k2.metric("💵 Cash Available", f"${total_cash:,.2f}")
    k3.metric(bp_label, f"${buying_power:,.2f}")

    with st.expander("Account details", expanded=False):
        st.json({
            "equity": account.get("equity"),
            "cash": account.get("cash"),
            "buying_power": account.get("buying_power"),
            "regt_buying_power": account.get("regt_buying_power"),
            "daytrading_buying_power": account.get("daytrading_buying_power"),
            "non_marginable_buying_power": account.get("non_marginable_buying_power"),
            "multiplier": raw_multiplier,
            "initial_margin": account.get("initial_margin"),
            "maintenance_margin": account.get("maintenance_margin"),
        })

    st.divider()

except Exception as e:
    st.error(f"Error fetching account data: {e}")

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
        import traceback
        st.code(traceback.format_exc())
        sig = None

    if not sig or sig.get("final_prob") is None:
        st.info(f"{sym}: No valid prediction available.")
        sig = None
    else:
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
    st.plotly_chart(fig, use_container_width=True, key="divergence_chart")


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
        fig, ax = plt.subplots(figsize=(8, 4))

        for mode in ["Daily", "Intraday"]:
            sub = df_chart[df_chart["Mode"] == mode]
            bars = ax.bar(sub["Symbol"] + " (" + mode + ")", sub["Accuracy"])

            # Add value labels inside bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height + 0.01,  # Slightly above bar
                    f'{height:.0%}',  # Format as percentage
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    fontweight='bold'
                )

        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")

        ax.grid(True, alpha=0.2)
        ax.tick_params(axis='x', labelsize=8, rotation=45)
        plt.tight_layout()
        st.pyplot(fig)


    # Logloss Chart
    with colB:
        st.subheader("Logloss Comparison")
        fig, ax = plt.subplots(figsize=(8, 4))

        for mode in ["Daily", "Intraday"]:
            sub = df_chart[df_chart["Mode"] == mode]
            bars = ax.bar(sub["Symbol"] + " (" + mode + ")", sub["Logloss"])

            # LogLoss labels (raw value + context)
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height + 0.001,
                    f'{height:.3f}',  # Raw LogLoss (3 decimals)
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    fontweight='bold'
                )

        ax.set_ylabel("Logloss (lower = better)")
        ax.axhline(y=0.693, color='red', linestyle='--', alpha=0.7, label='Random (0.693)')
        ax.legend()
        ax.grid(True, alpha=0.2)
        ax.tick_params(axis='x', labelsize=8, rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

# Price charts
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

    # ─────────────────────────────────────────────────────────────────────────────
    # PER-SYMBOL PnL — deposit-aware
    # ─────────────────────────────────────────────────────────────────────────────
    # Strategy: each symbol gets an equal share of every deposit/withdrawal.
    # Formula: pnl = cumcf + position_value - sym_allocated_deposits
    #
    # Where:
    #   cumcf              = running sum of (sell proceeds - buy costs) for THIS symbol
    #   position_value     = shares_held × last_price
    #   sym_allocated_deps = each deposit × (1 / n_active_symbols_at_that_time)
    #
    # This makes per-symbol PnL start at $0 and show ONLY trading gains,
    # matching the Total Portfolio PnL chart logic.
    # ─────────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────────
    # PER-SYMBOL PnL — deposit-aware, self-contained equity
    # Drop-in replacement for the entire `if "value" in df.columns:` block
    # ─────────────────────────────────────────────────────────────────────────────

    if "value" in df.columns:
        try:
            from plotly.subplots import make_subplots

            # ── 1. build localtime from timestamp ────────────────────────────────
            dfp = df.copy()
            dfp["timestamp"] = pd.to_datetime(dfp["timestamp"], utc=True, errors="coerce")
            dfp["localtime"] = dfp["timestamp"].dt.tz_convert(tz)
            dfp = dfp.sort_values("localtime").dropna(subset=["localtime"])

            # ── 2. numeric columns ───────────────────────────────────────────────
            dfp["price"]  = pd.to_numeric(dfp.get("price"),  errors="coerce").fillna(0)
            dfp["qty"]    = pd.to_numeric(dfp.get("qty"),    errors="coerce").fillna(0)
            dfp["shares"] = pd.to_numeric(dfp.get("shares"), errors="coerce").fillna(0)
            dfp["action"] = dfp["action"].astype(str).str.lower().str.strip()

            # ── 3. cashflow per trade ─────────────────────────────────────────────
            def _cf(r):
                if r["action"] == "sell": return  r["qty"] * r["price"]
                if r["action"] == "buy":  return -r["qty"] * r["price"]
                return 0.0
            dfp["cf"]             = dfp.apply(_cf, axis=1)
            dfp["cumcf"]          = dfp["cf"].cumsum()
            dfp["position_value"] = dfp["shares"] * dfp["price"]

            # ── 4. self-contained per-symbol equity (NO account-wide value col) ──
            # cumcf + position_value is 100% per-symbol, starts at 0 before trades
            dfp["sym_equity"] = dfp["cumcf"] + dfp["position_value"]

            # ── 5. allocate deposits to this symbol ──────────────────────────────
            _spy = (SPY_SYMBOL or "SPY").upper()
            _trading_syms = [s for s in symbols if s.upper() != _spy]
            n_symbols = max(1, len(_trading_syms))
            sym_allocated_deposits = 0.0
            if df_dep is not None and not df_dep.empty:
                ic = 0.0
                if os.path.exists(portfolio_path):
                    try:
                        _dp = pd.read_csv(portfolio_path)
                        if "initial_cash" in _dp.columns:
                            ic = float(pd.to_numeric(_dp["initial_cash"], errors="coerce").fillna(0).iloc[0])
                    except Exception:
                        pass
                total_deps = ic + float(df_dep["amount"].clip(lower=0).sum())
                sym_allocated_deposits = total_deps / n_symbols

            # ── 6. PnL = trading gains only (deposits stripped) ──────────────────
            dfp["pnl_value"]      = dfp["sym_equity"] - sym_allocated_deposits
            # Raw equity = trading result + deposit allocation (starts at ~$1500)
            dfp["sym_raw_equity"] = dfp["sym_equity"] + sym_allocated_deposits

            # ── 7. chart ─────────────────────────────────────────────────────────
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=(
                    "🤖 Bot PnL (trading gains only)",
                    "🗃️ Account Equity (raw, includes deposits)",
                ),
            )
            fig.add_trace(
                go.Scatter(
                    x=dfp["localtime"], y=dfp["pnl_value"],
                    mode="lines+markers", name="PnL",
                    line=dict(color="#00b4d8", width=2),
                    hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>PnL: $%{y:,.2f}<extra></extra>",
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=dfp["localtime"], y=dfp["sym_raw_equity"],
                    mode="lines+markers", name="Raw Equity",
                    line=dict(color="#adb5bd", width=2, dash="dash"),
                    hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>Equity: $%{y:,.2f}<extra></extra>",
                ),
                row=1, col=2,
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

            # deposit markers on equity panel
            if df_dep is not None and not df_dep.empty:
                for _, deprow in df_dep.iterrows():
                    dep_dt = deprow["date"]
                    amount = float(deprow["amount"])
                    if dep_dt.tzinfo is None:
                        dep_dt = dep_dt.tz_localize("UTC").tz_convert(tz)
                    else:
                        dep_dt = dep_dt.tz_convert(tz)
                    fig.add_vline(
                        x=dep_dt.timestamp() * 1000,
                        line_width=1, line_dash="dot",
                        line_color="green" if amount > 0 else "red",
                        annotation_text=f"${amount/n_symbols:,.0f}",
                        annotation_position="top right",
                        row=1, col=2,
                    )

            fig.update_layout(
                title=f"{sym} — PnL vs Equity",
                height=420,
                template="plotly_white",
                hovermode="x unified",
                showlegend=False,
            )
            fig.update_yaxes(tickprefix="$", separatethousands=True, row=1, col=1)
            fig.update_yaxes(tickprefix="$", separatethousands=True, row=1, col=2)

            st.plotly_chart(fig, use_container_width=True, key=f"valuechart_{sym}")

        except Exception as e:
            st.warning(f"Could not plot value chart: {e}")

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
    def _exec_qty(r):
        sb = float(r["shares_before"])
        sa = float(r["shares_after"])
        if r["action"] == "buy":
            return max(0.0, sa - sb)
        if r["action"] == "sell":
            return max(0.0, sb - sa)
        return 0.0


    df["exec_qty"] = df.apply(_exec_qty, axis=1)


    # drop rows that don't change position
    df = df[df["exec_qty"] > 0].copy()


    if df.empty:
        st.info("No filled trades detected (position never changed).")
    else:
        # ---- cashflow from executed qty ----
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
            gross_loss = float(-s[s < 0].sum())


            win_rate = float((s > 0).mean() * 100.0)
            avg_win = float(s[s > 0].mean()) if (s > 0).any() else 0.0
            avg_loss = float(s[s < 0].mean()) if (s < 0).any() else 0.0
            largest_win = float(s.max())
            largest_loss = float(s.min())


            # Profit factor
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
# Price vs Model PERFORMANCE (with trade markers)
# -------------------------------------------------
st.header("📈 Price vs Model Probability (with Buy/Sell Markers)")


for sym in symbols:
    # Use robust loader
    df = load_signal_history(sym)

    if df is None or df.empty:
        st.info(f"No signal history for {sym}")
        continue

    df = df.sort_values("timestamp").tail(300)

    # ---- Load trades (optional) ----
    trade_path = get_trade_log_file(sym)
    df_tr = None
    if os.path.exists(trade_path):
        try:
            df_tr = pd.read_csv(trade_path)
            df_tr["timestamp"] = pd.to_datetime(df_tr["timestamp"], utc=True, errors="coerce")
            df_tr["action"] = df_tr["action"].astype(str).str.lower().str.strip()
            df_tr["price"] = pd.to_numeric(df_tr.get("price"), errors="coerce")
            df_tr["qty"] = pd.to_numeric(df_tr.get("qty"), errors="coerce")
            df_tr = df_tr.dropna(subset=["timestamp", "action", "price"])
            df_tr = df_tr[df_tr["action"].isin(["buy", "sell"])].copy()
        except Exception:
            df_tr = None

    trace_options = ["Price", "Final prob", "Intraday prob", "Daily prob", "BUY", "SELL"]
    default_traces = ["Price", "Final prob", "BUY", "SELL"]

    visible_traces = st.multiselect(
        f"Show traces for {sym}",
        options=trace_options,
        default=default_traces,
        key=f"trace_selector_{sym}",
    )

    fig = go.Figure()

    # ---- PRICE (left axis)
    if "Price" in visible_traces and "price" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["price"],
            name="Price",
            line=dict(color="black", width=2),
            yaxis="y1",
            hovertemplate="Price: %{y:.2f}<extra></extra>",
        ))

    # ---- FINAL PROBABILITY (right axis)
    if ("Final prob" in visible_traces) and ("finalprob" in df.columns or "final_prob" in df.columns):
        prob_col = "finalprob" if "finalprob" in df.columns else "final_prob"
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df[prob_col],
            name="Final Probability",
            line=dict(color="blue", width=2),
            yaxis="y2",
            hovertemplate="Final prob: %{y:.3f}<extra></extra>",
        ))

    # ---- Intraday prob
    if ("Intraday prob" in visible_traces) and ("intradayprob" in df.columns or "intraday_prob" in df.columns):
        ip_col = "intradayprob" if "intradayprob" in df.columns else "intraday_prob"
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df[ip_col],
            name="Intraday Prob",
            line=dict(color="orange", dash="dot"),
            yaxis="y2",
            opacity=0.6,
            hovertemplate="Intraday prob: %{y:.3f}<extra></extra>",
        ))

    # ---- Daily prob
    if ("Daily prob" in visible_traces) and ("dailyprob" in df.columns or "daily_prob" in df.columns):
        dp_col = "dailyprob" if "dailyprob" in df.columns else "daily_prob"
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df[dp_col],
            name="Daily Prob",
            line=dict(color="green", dash="dash"),
            yaxis="y2",
            opacity=0.6,
            hovertemplate="Daily prob: %{y:.3f}<extra></extra>",
        ))

    # ---- BUY/SELL MARKERS (on price axis)
    if df_tr is not None and not df_tr.empty:
        buys = df_tr[df_tr["action"] == "buy"].copy()
        sells = df_tr[df_tr["action"] == "sell"].copy()

        if "BUY" in visible_traces and not buys.empty:
            fig.add_trace(go.Scatter(
                x=buys["timestamp"],
                y=buys["price"],
                mode="markers",
                name="BUY",
                yaxis="y1",
                marker=dict(symbol="triangle-up", size=12, color="green", line=dict(width=1, color="black")),
                customdata=buys[["qty"]].values,
                hovertemplate=(
                    "<b>BUY</b><br>"
                    "Time: %{x|%Y-%m-%d %H:%M:%S} UTC<br>"
                    "Price: %{y:.2f}<br>"
                    "Qty: %{customdata[0]:g}<extra></extra>"
                ),
            ))

        if "SELL" in visible_traces and not sells.empty:
            fig.add_trace(go.Scatter(
                x=sells["timestamp"],
                y=sells["price"],
                mode="markers",
                name="SELL",
                yaxis="y1",
                marker=dict(symbol="triangle-down", size=12, color="red", line=dict(width=1, color="black")),
                customdata=sells[["qty"]].values,
                hovertemplate=(
                    "<b>SELL</b><br>"
                    "Time: %{x|%Y-%m-%d %H:%M:%S} UTC<br>"
                    "Price: %{y:.2f}<br>"
                    "Qty: %{customdata[0]:g}<extra></extra>"
                ),
            ))


    fig.update_layout(
        title=f"{sym} — Price vs Probability",
        height=420,
        hovermode="x unified",
        template="plotly_white",
        yaxis=dict(title="Price"),
        yaxis2=dict(
            title="Probability",
            overlaying="y",
            side="right",
            range=[0, 1],
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )


    st.plotly_chart(fig, use_container_width=True, key=f"price_prob_chart_{sym}")



# -------------------------------------------------
# TOTAL DAILY PORTFOLIO PERFORMANCE
# -------------------------------------------------
st.header("📈 Total Portfolio Performance (All Symbols Combined)")

if os.path.exists(portfolio_path):
    df = pd.read_csv(portfolio_path)

    if df.empty:
        st.warning("Daily portfolio is empty.")
    else:
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(tz)
        df = df.sort_values("date").reset_index(drop=True)

        if df_dep is not None and not df_dep.empty:
            df_dep_chart = df_dep.copy()
            df_dep_chart["date"] = df_dep_chart["date"].dt.tz_convert(tz)
        else:
            df_dep_chart = None

        value_col = next(
            (c for c in ["value", "equity", "total_value", "portfolio_value"] if c in df.columns),
            None
        )
        if value_col is None:
            st.error(f"No value column found. Columns: {df.columns.tolist()}")
            st.stop()

        df["total_equity"] = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0)
        if "external_flow" in df.columns:
            df["external_flow"] = pd.to_numeric(df["external_flow"], errors="coerce").fillna(0.0)
        else:
            df["external_flow"] = 0.0

        if "initial_cash" in df.columns:
            initial_cash = pd.to_numeric(df["initial_cash"], errors="coerce").fillna(0.0).iloc[0]
        else:
            initial_cash = float(df["total_equity"].iloc[0])

        df["cum_external_flow"] = df["external_flow"].cumsum()
        df["total_deposited"] = initial_cash + df["cum_external_flow"]
        df["pnl_value"] = df["total_equity"] - df["total_deposited"]

        # -------------------------------------------------
        # TOTAL PERFORMANCE STATS
        # -------------------------------------------------
        st.subheader("📊 Total Performance Stats")

        df_stats = df.copy()
        df_stats = df_stats.dropna(subset=["date", "total_equity"]).sort_values("date").reset_index(drop=True)

        if len(df_stats) >= 2:
            start_date = df_stats["date"].iloc[0]
            end_date = df_stats["date"].iloc[-1]

            start_equity = float(df_stats["total_equity"].iloc[0])
            end_equity = float(df_stats["total_equity"].iloc[-1])
            total_pnl = float(df_stats["pnl_value"].iloc[-1])

            # Prefer a real positive starting capital
            base_capital = None
            if "total_deposited" in df_stats.columns:
                first_dep = float(pd.to_numeric(df_stats["total_deposited"], errors="coerce").fillna(0.0).iloc[0])
                if first_dep > 0:
                    base_capital = first_dep

            if base_capital is None or base_capital <= 0:
                if "initial_cash" in locals() and initial_cash and float(initial_cash) > 0:
                    base_capital = float(initial_cash)

            if base_capital is None or base_capital <= 0:
                base_capital = max(start_equity - total_pnl, 1e-9)

            total_return = total_pnl / base_capital if base_capital > 0 else 0.0

            elapsed_days = max((end_date - start_date).total_seconds() / 86400.0, 1.0)
            annual_return = (1.0 + total_return) ** (365.25 / elapsed_days) - 1.0 if total_return > -1 else -1.0
            annual_label = "Annualized Return"

            # Build a positive equity curve for drawdown / Sharpe
            df_stats["strategy_equity"] = base_capital + df_stats["pnl_value"]
            df_stats["strategy_equity"] = pd.to_numeric(df_stats["strategy_equity"], errors="coerce").fillna(method="ffill")

            # Prevent divide-by-zero / negative starting peak issues
            df_stats = df_stats[df_stats["strategy_equity"] > 0].copy()

            if len(df_stats) >= 2:
                df_stats["running_peak"] = df_stats["strategy_equity"].cummax()
                df_stats["drawdown"] = df_stats["strategy_equity"] / df_stats["running_peak"] - 1.0
                max_drawdown = float(df_stats["drawdown"].min())
            else:
                max_drawdown = 0.0

            daily_curve = (
                df_stats.set_index("date")[["strategy_equity"]]
                .resample("1D")
                .last()
                .dropna()
                .copy()
            )
            daily_curve["ret"] = daily_curve["strategy_equity"].pct_change()
            daily_rets = daily_curve["ret"].replace([float("inf"), float("-inf")], pd.NA).dropna()

            if len(daily_rets) >= 2 and daily_rets.std() > 0:
                sharpe = float((daily_rets.mean() / daily_rets.std()) * (252 ** 0.5))
                volatility = float(daily_rets.std() * (252 ** 0.5))
            else:
                sharpe = None
                volatility = None

            # Closed-trade stats across all symbols
            all_cycle_pnls = []

            for sym in symbols:
                trade_path = get_trade_log_file(sym)
                if not os.path.exists(trade_path):
                    continue

                try:
                    dft = pd.read_csv(trade_path)
                    dft["timestamp"] = pd.to_datetime(dft["timestamp"], utc=True, errors="coerce")
                    dft["action"] = dft["action"].astype(str).str.lower().str.strip()
                    dft["price"] = pd.to_numeric(dft.get("price"), errors="coerce")
                    dft["shares"] = pd.to_numeric(dft.get("shares"), errors="coerce")

                    dft = dft.dropna(subset=["timestamp", "action", "price", "shares"]).sort_values("timestamp").copy()

                    if "shares_after" not in dft.columns:
                        dft["shares_after"] = dft["shares"]
                    else:
                        dft["shares_after"] = pd.to_numeric(dft["shares_after"], errors="coerce").fillna(dft["shares"])

                    if "shares_before" not in dft.columns:
                        dft["shares_before"] = dft["shares_after"].shift(1).fillna(0.0)
                    else:
                        dft["shares_before"] = pd.to_numeric(dft["shares_before"], errors="coerce").fillna(0.0)

                    def _exec_qty_total(r):
                        sb = float(r["shares_before"])
                        sa = float(r["shares_after"])
                        if r["action"] == "buy":
                            return max(0.0, sa - sb)
                        if r["action"] == "sell":
                            return max(0.0, sb - sa)
                        return 0.0

                    dft["exec_qty"] = dft.apply(_exec_qty_total, axis=1)
                    dft = dft[dft["exec_qty"] > 0].copy()
                    if dft.empty:
                        continue

                    dft["cashflow"] = dft.apply(
                        lambda r: -(r["exec_qty"] * r["price"]) if r["action"] == "buy"
                        else +(r["exec_qty"] * r["price"]) if r["action"] == "sell"
                        else 0.0,
                        axis=1,
                    )

                    EPS = 1e-9
                    in_cycle = False
                    running = 0.0

                    for _, r in dft.iterrows():
                        sb = float(r["shares_before"])
                        sa = float(r["shares_after"])
                        cf = float(r["cashflow"])

                        was_flat = abs(sb) <= EPS
                        now_flat = abs(sa) <= EPS

                        if (not in_cycle) and was_flat and (not now_flat):
                            in_cycle = True
                            running = 0.0

                        if in_cycle:
                            running += cf

                        if in_cycle and (not was_flat) and now_flat:
                            all_cycle_pnls.append(running)
                            in_cycle = False
                            running = 0.0

                except Exception:
                    pass

            if all_cycle_pnls:
                s = pd.Series(all_cycle_pnls, dtype=float)
                win_rate = float((s > 0).mean())
                gross_profit = float(s[s > 0].sum())
                gross_loss = float(-s[s < 0].sum())
                profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
                avg_trade = float(s.mean())
                closed_trades = int(len(s))
            else:
                win_rate = None
                profit_factor = None
                avg_trade = None
                closed_trades = 0

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Equity", f"${end_equity:,.2f}")
            c2.metric("Total PnL", f"${total_pnl:,.2f}")
            c3.metric("Total Return", f"{total_return * 100:.2f}%")
            c4.metric(annual_label, f"{annual_return * 100:.2f}%")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Max Drawdown", f"{max_drawdown * 100:.2f}%")
            c6.metric("Sharpe Ratio", "N/A" if sharpe is None else f"{sharpe:.2f}")
            c7.metric("Win Rate", "N/A" if win_rate is None else f"{win_rate * 100:.1f}%")
            c8.metric("Profit Factor", "∞" if profit_factor == float("inf") else ("N/A" if profit_factor is None else f"{profit_factor:.2f}"))

            c9, c10, c11 = st.columns(3)
            c9.metric("Volatility", "N/A" if volatility is None else f"{volatility * 100:.2f}%")
            c10.metric("Avg Closed Trade", "N/A" if avg_trade is None else f"${avg_trade:,.2f}")
            c11.metric("Closed Trades", f"{closed_trades}")
        else:
            st.info("Not enough portfolio history yet to compute total performance stats.")

        st.write("**External flow summary:**")
        st.dataframe(df[["date", "external_flow", "cum_external_flow", "total_deposited", "pnl_value"]].tail(10))

        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=df["date"],
            y=df["total_equity"],
            mode="lines",
            name="Total Equity (with deposits)",
            line=dict(width=2, color="#636efa"),
            hovertemplate=(
                "<b>Total Equity</b><br>"
                "Date: %{x|%Y-%m-%d}<br>"
                "Value: $%{y:,.2f}"
                "<extra></extra>"
            ),
        ))

        if df_dep_chart is not None and not df_dep_chart.empty:
            target_tz = str(df["date"].dt.tz)
            has_dep_legend = False
            has_wdr_legend = False

            for _, row in df_dep_chart.iterrows():
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

                fig_eq.add_trace(go.Scatter(
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

        fig_eq.update_layout(
            title="💼 Total Account Equity (includes deposits/withdrawals)",
            xaxis_title="Date",
            yaxis_title="Total Equity ($)",
            hovermode="x unified",
            template="plotly_white",
            height=420,
        )
        st.plotly_chart(fig_eq, use_container_width=True, key="portfolio_chart")

        st.subheader("🤖 Bot PnL — Trading Gains Only (deposits stripped)")
        if len(df) < 2:
            st.info("⏳ Not enough data yet to show PnL curve — needs at least 2 days of portfolio history.")
        else:
            fig_pnl = go.Figure()
            fig_pnl.add_trace(go.Scatter(
                x=df["date"],
                y=df["pnl_value"],
                mode="lines",
                name="PnL-Only Equity",
                line=dict(width=2, color="#00b4d8"),
                hovertemplate=(
                    "<b>Bot PnL</b><br>"
                    "Date: %{x|%Y-%m-%d}<br>"
                    "Value: $%{y:,.2f}"
                    "<extra></extra>"
                ),
            ))
            fig_pnl.add_hline(y=0, line_dash="dash", line_color="red", line_width=1)
            fig_pnl.update_layout(
                xaxis_title="Date",
                yaxis_title="PnL-Only Equity ($)",
                hovermode="x unified",
                template="plotly_white",
                height=320,
            )
            st.plotly_chart(fig_pnl, use_container_width=True, key="total_pnl_chart")
else:
    st.info("No daily portfolio file found. Run update_portfolio_data.py.")
