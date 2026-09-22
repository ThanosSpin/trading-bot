# dashboard.py
from portfolio import get_daily_portfolio_file
from plotly.subplots import make_subplots
from predictive_model.data_loader import fetch_historical_data, fetch_intraday_history
from predictive_model.model_xgb import compute_signals
from trader import get_margin_status
from portfolio import (
    get_trade_log_file,
    get_daily_portfolio_file,
    get_live_portfolio,
)
from pathlib import Path
from config import (
    BOT_ENV,
    TIMEZONE,
    SYMBOL,
    MODEL_DIR,
    SPY_SYMBOL,
    INTRADAY_WEIGHT,
    DATA_DIR,
    LOGS_DIR,
    RESOURCE_FEE_PCT,
    HURDLE_PCT,
    PERFORMANCE_FEE_PCT
)
import plotly.graph_objects as go
import joblib
from typing import Optional
from market import debug_market
from streamlit_autorefresh import st_autorefresh
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import os
import streamlit as st
import pandas as pd
import numpy as np
import pytz
import matplotlib

matplotlib.use("Agg")
BASE_DIR = Path(__file__).resolve().parent

# At the top of dashboard.py, near other config-like settings
SHOW_DEBUG_BLOCKS = False  # set True for debugging purposes

# -------------------------------------------------
# Streamlit setup
# -------------------------------------------------
st.set_page_config(
    page_title=f"Trading Bot Dashboard - {BOT_ENV.upper()}",
    layout="wide",
)

st.title(f"📊 Trading Bot Dashboard — {BOT_ENV.upper()}")


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
# Refresh the full dashboard (and therefore prob_up) every 30 minutes by
# default. The interval can be overridden per service without a code change.
try:
    REFRESH_INTERVAL = int(os.getenv("DASHBOARD_REFRESH_SECONDS", "1800"))
except (TypeError, ValueError):
    REFRESH_INTERVAL = 1800

REFRESH_INTERVAL = max(REFRESH_INTERVAL, 60)
st_autorefresh(interval=REFRESH_INTERVAL * 1000, key="global_refresh")
st.caption(f"⏳ Auto-refreshing every {REFRESH_INTERVAL // 60} minutes.")


# Normalize SYMBOL into list
symbols = SYMBOL if isinstance(SYMBOL, list) else [SYMBOL]
symbols = [s.upper() for s in symbols]


# Optional toggle to include SPY in dashboard
include_spy = st.checkbox(f"Include {SPY_SYMBOL} in dashboard", value=False)


def _has_model(sym: str) -> bool:
    return os.path.exists(
        os.path.join(MODEL_DIR, f"{sym}_daily_xgb.pkl")
    ) or os.path.exists(os.path.join(MODEL_DIR, f"{sym}_intraday_xgb.pkl"))


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

# -------------------------------------------------
# INVESTOR CASH FLOWS
# -------------------------------------------------
#
# IMPORTANT:
# deposits_auto.csv contains REAL investor deposits/withdrawals only.
#
# It intentionally excludes broker/system corrections such as Alpaca's
# erroneous +$1,000 / -$1,000 pair.
#
# daily_portfolio.csv.external_flow is separate: that contains ALL broker
# cash movements and is used for cash-flow-adjusted account PnL.
# -------------------------------------------------

portfolio_path = get_daily_portfolio_file()
data_base = os.path.dirname(portfolio_path)

deposit_auto_path = os.path.join(data_base, "deposits_auto.csv")
deposit_manual_path = os.path.join(data_base, "deposits.csv")

df_dep = pd.DataFrame(columns=["date", "amount"])

# Primary source: filtered investor-flow file generated by
# update_portfolio_data.py
if os.path.exists(deposit_auto_path):
    try:
        _raw = pd.read_csv(deposit_auto_path)

        _raw["date"] = pd.to_datetime(
            _raw["date"],
            utc=True,
            errors="coerce",
        )

        _raw["amount"] = pd.to_numeric(
            _raw["amount"],
            errors="coerce",
        )

        _raw = _raw.dropna(subset=["date", "amount"])

        df_dep = _raw[["date", "amount"]].sort_values("date").reset_index(drop=True)

        st.caption(
            f"💳 Loaded {len(df_dep)} investor cash-flow entries "
            f"from deposits_auto.csv."
        )

    except Exception as _e:
        st.caption(f"⚠️ Could not read deposits_auto.csv: {_e}")

# Fallback: manual deposits.csv
elif os.path.exists(deposit_manual_path):
    try:
        _raw = pd.read_csv(deposit_manual_path)

        _raw["date"] = pd.to_datetime(
            _raw["date"],
            utc=True,
            errors="coerce",
        )

        _raw["amount"] = pd.to_numeric(
            _raw["amount"],
            errors="coerce",
        )

        _raw = _raw.dropna(subset=["date", "amount"])

        df_dep = _raw[["date", "amount"]].sort_values("date").reset_index(drop=True)

        st.caption(
            f"💳 Loaded {len(df_dep)} investor cash-flow entries " f"from deposits.csv."
        )

    except Exception as _e:
        st.caption(f"⚠️ Could not read deposits.csv: {_e}")

else:
    st.caption("ℹ️ No investor deposit history found.")


if SHOW_DEBUG_BLOCKS:
    with st.expander(
        "🔍 Investor Cash-Flow Debug",
        expanded=False,
    ):
        st.write("These are investor flows only. " "Broker corrections are excluded.")

        if df_dep.empty:
            st.info("No investor flows loaded.")
        else:
            st.dataframe(
                df_dep,
                use_container_width=True,
            )
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


# ─────────────────────────────────────────────────────────────────────────────

# -------------------------------------------------
# Helper: Calculate Investor Fees
# -------------------------------------------------


def calculate_investor_fees(
    starting_capital: float,
    period_return_pct: float,
    resource_fee_pct: float = 2.0,
    hurdle_pct: float = 5.0,
    performance_fee_pct: float = 15.0,
    period_fraction: float = 1.0,
    resource_fee_base: float = None,
):
    """
    Calculate investor fees for one fee period.

    Fee logic:

        1. Actual period return determines gross profit.

        2. Resource fee is a separate base fee calculated from
           resource_fee_base.

        3. Hurdle is calculated from the performance capital
           (starting_capital).

        4. Performance fee is a fixed percentage of profit
           remaining after:

               gross profit
               - resource fee
               - hurdle profit

        5. Ending capital is the investor's capital after all fees.

    period_return_pct:
        Actual TWR return for the fee period, in percent.

    period_fraction:
        1.0 for a completed annual period.

        Less than 1.0 for the current incomplete period,
        so the annual resource fee and hurdle are prorated.

    resource_fee_base:
        Capital base used only for the resource fee.

        Example:
            completed first period:
                deposits

            current/future period:
                opening capital + new deposits
    """

    # -------------------------------------------------
    # INPUT CLEANUP
    # -------------------------------------------------

    starting_capital = max(
        float(starting_capital),
        0.0,
    )

    period_return_pct = float(period_return_pct)

    period_fraction = min(
        max(
            float(period_fraction),
            0.0,
        ),
        1.0,
    )

    if resource_fee_base is None:
        resource_fee_base = starting_capital

    resource_fee_base = max(
        float(resource_fee_base),
        0.0,
    )

    # -------------------------------------------------
    # EFFECTIVE FEE RATES
    # -------------------------------------------------
    #
    # Completed year:
    #     period_fraction = 1.0
    #
    # Current incomplete year:
    #     annual resource fee and hurdle are prorated.
    # -------------------------------------------------

    effective_resource_fee_pct = resource_fee_pct * period_fraction

    effective_hurdle_pct = hurdle_pct * period_fraction

    # -------------------------------------------------
    # ACTUAL INVESTMENT PERFORMANCE
    # -------------------------------------------------

    gross_profit = starting_capital * (period_return_pct / 100.0)

    gross_value = starting_capital + gross_profit

    # -------------------------------------------------
    # RESOURCE FEE
    # -------------------------------------------------
    #
    # Separate base fee for your work.
    #
    # This uses resource_fee_base, which may be
    # different from starting_capital.
    # -------------------------------------------------

    resource_fee = resource_fee_base * (effective_resource_fee_pct / 100.0)

    # -------------------------------------------------
    # PROFIT AFTER RESOURCE FEE
    # -------------------------------------------------

    net_profit_after_resource = gross_profit - resource_fee

    # -------------------------------------------------
    # HURDLE
    # -------------------------------------------------
    #
    # Hurdle is based on the performance capital.
    # -------------------------------------------------

    hurdle_profit = starting_capital * (effective_hurdle_pct / 100.0)

    # -------------------------------------------------
    # FEEABLE PROFIT
    # -------------------------------------------------
    #
    # Example:
    #
    # Gross Profit       $2,000
    # Resource Fee         $200
    # Hurdle Profit        $500
    #
    # Feeable Profit:
    #     $2,000 - $200 - $500
    #     = $1,300
    # -------------------------------------------------

    feeable_profit = max(
        0.0,
        net_profit_after_resource - hurdle_profit,
    )

    # -------------------------------------------------
    # PERFORMANCE FEE
    # -------------------------------------------------
    #
    # Fixed contractual percentage of feeable profit.
    #
    # Example:
    #
    # $1,300 × 15%
    # = $195
    # -------------------------------------------------

    performance_fee = feeable_profit * (performance_fee_pct / 100.0)

    # -------------------------------------------------
    # TOTAL FEES
    # -------------------------------------------------

    total_fees = resource_fee + performance_fee

    # -------------------------------------------------
    # ENDING INVESTOR CAPITAL
    # -------------------------------------------------

    ending_capital = gross_value - total_fees

    # -------------------------------------------------
    # RETURN RESULT
    # -------------------------------------------------

    return {
        "starting_capital": starting_capital,
        "period_return_pct": period_return_pct,
        "gross_profit": gross_profit,
        "gross_value": gross_value,
        "resource_fee_base": resource_fee_base,
        "resource_fee_pct": effective_resource_fee_pct,
        "resource_fee": resource_fee,
        "net_profit_after_resource": net_profit_after_resource,
        "hurdle_pct": effective_hurdle_pct,
        "hurdle_profit": hurdle_profit,
        "feeable_profit": feeable_profit,
        "performance_fee_pct": performance_fee_pct,
        "performance_fee": performance_fee,
        "total_fees": total_fees,
        "ending_capital": ending_capital,
    }


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

    paths = [
        Path(LOGS_DIR) / f"signals_{sym}.csv",
        Path(DATA_DIR) / f"signals_{sym}.csv",
    ]

    # Legacy Live1 signal history lives in project_root/logs/
    if BOT_ENV == "live":
        legacy_log = BASE_DIR / "logs" / f"signals_{sym}.csv"

        if legacy_log not in paths:
            paths.append(legacy_log)

    return [str(p) for p in paths]


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
                df["timestamp"] = pd.to_datetime(
                    df["timestamp"], utc=True, errors="coerce"
                )
                df = df.dropna(subset=["timestamp"])

            return df

        except pd.errors.ParserError as e:
            # Handle corrupted/mismatched schema
            st.warning(
                f"⚠️ {sym}: Signal log has schema mismatch. Attempting recovery..."
            )

            try:
                # Try skipping bad lines
                df = pd.read_csv(p, on_bad_lines="skip")

                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(
                        df["timestamp"], utc=True, errors="coerce"
                    )
                    df = df.dropna(subset=["timestamp"])

                st.success(f"✅ Recovered {len(df)} valid rows for {sym}")
                return df

            except Exception as e2:
                st.error(f"❌ Could not recover {sym} signal log: {e2}")

                # Offer to delete corrupted file
                if st.button(
                    f"🗑️ Delete corrupted signal log for {sym}",
                    key=f"delete_signal_{sym}",
                ):
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
    k1.metric("💼 Broker Equity", f"${total_equity:,.2f}")
    k2.metric("💵 Cash Available", f"${total_cash:,.2f}")
    k3.metric(bp_label, f"${buying_power:,.2f}")

    with st.expander("Account details", expanded=False):
        st.json(
            {
                "equity": account.get("equity"),
                "cash": account.get("cash"),
                "buying_power": account.get("buying_power"),
                "regt_buying_power": account.get("regt_buying_power"),
                "daytrading_buying_power": account.get("daytrading_buying_power"),
                "non_marginable_buying_power": account.get(
                    "non_marginable_buying_power"
                ),
                "multiplier": raw_multiplier,
                "initial_margin": account.get("initial_margin"),
                "maintenance_margin": account.get("maintenance_margin"),
            }
        )

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
# MARGIN STATUS
# -------------------------------------------------
st.header("📊 Margin / Buying Power Status")
margin = get_margin_status()
if margin:
    equity = margin.get("equity", 0.0)
    bp = margin.get("buying_power", 0.0)
    multiplier = margin.get("multiplier", None)
    blocked = bool(margin.get("trading_blocked", False))

    # Build a human-readable status line
    mult_str = f"{multiplier}x" if multiplier is not None else "N/A"
    msg = (
        f"Equity: ${equity:.2f} | "
        f"Buying Power: ${bp:.2f} | "
        f"Margin Multiplier: {mult_str} | "
        f"{'🚫 Trading BLOCKED' if blocked else '✅ Trading allowed'}"
    )

    if blocked:
        st.error(msg)
    else:
        # Optional: warn if buying power is very low relative to equity
        if equity > 0 and bp < 0.5 * equity:
            st.warning(msg)
        else:
            st.success(msg)
else:
    st.info("Unable to fetch margin status.")


# -------------------------------------------------
# MODEL SIGNALS
# -------------------------------------------------
st.header("📡 Model Signals & Price Charts")


for sym in symbols:
    st.subheader(f"Signals for {sym}")
    try:
        sig = compute_signals(
            sym,
            lookback_minutes=2400,
            intraday_weight=INTRADAY_WEIGHT,
            resample_to="15min",
        )
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
        col3.metric(
            "Intraday model", f"{intra_p:.3f}" if intra_p is not None else "N/A"
        )
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

            st.session_state["divergence_points"].append(
                {
                    "time": pd.Timestamp.utcnow(),
                    "symbol": sym,
                    "dp": float(daily_p) if daily_p is not None else None,
                    "ip": float(intra_p) if intra_p is not None else None,
                    "divergence": float(div) if div is not None else None,
                    "weight": float(weight) if weight is not None else None,
                    "model": pretty_model,
                }
            )

            # keep last N points overall (prevents memory growth)
            MAX_POINTS = 500
            if len(st.session_state["divergence_points"]) > MAX_POINTS:
                st.session_state["divergence_points"] = st.session_state[
                    "divergence_points"
                ][-MAX_POINTS:]

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
                            unsafe_allow_html=True,
                        )
                    except:
                        pass

                for key in [
                    "accuracy",
                    "logloss",
                    "roc_auc",
                    "precision",
                    "recall",
                    "f1",
                ]:
                    container.write(f"- **{key}**: `{metrics.get(key)}`")

                cm = metrics.get("confusion_matrix")
                if cm:
                    container.write(f"- **Confusion Matrix**: `{cm}`")

                return {
                    "accuracy": metrics.get("accuracy"),
                    "logloss": metrics.get("logloss"),
                    "age": age_days,
                }

            daily_stats = show_model_block(c_md1, "Daily", info_daily)
            intra_stats = show_model_block(c_md2, "Intraday", info_intra)

            if "model_compare" not in st.session_state:
                st.session_state["model_compare"] = {}
            st.session_state["model_compare"][sym] = {
                "daily": daily_stats,
                "intraday": intra_stats,
            }


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
    latest = dfd.sort_values("time").groupby("symbol", as_index=False).tail(1).copy()

    latest = latest[
        ["symbol", "dp", "ip", "divergence", "weight", "model"]
    ].sort_values("symbol")
    st.dataframe(latest, use_container_width=True)

    # plot divergence over time
    fig = go.Figure()
    for sym in sorted(dfd["symbol"].dropna().unique()):
        sub = dfd[dfd["symbol"] == sym].copy()
        # skip symbols with no divergence values yet
        sub = sub.dropna(subset=["divergence"])
        if sub.empty:
            continue

        fig.add_trace(
            go.Scatter(
                x=sub["time"],
                y=sub["divergence"],
                mode="lines+markers",
                name=sym,
                hovertemplate=(
                    "<b>%{x|%Y-%m-%d %H:%M:%S} UTC</b><br>"
                    "ip - dp: %{y:.3f}<extra></extra>"
                ),
            )
        )

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

    df_chart = pd.DataFrame(
        chart_data, columns=["Symbol", "Mode", "Accuracy", "Logloss"]
    )

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
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,  # Slightly above bar
                    f"{height:.0%}",  # Format as percentage
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")

        ax.grid(True, alpha=0.2)
        ax.tick_params(axis="x", labelsize=8, rotation=45)
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
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.001,
                    f"{height:.3f}",  # Raw LogLoss (3 decimals)
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

        ax.set_ylabel("Logloss (lower = better)")
        ax.axhline(
            y=0.693, color="red", linestyle="--", alpha=0.7, label="Random (0.693)"
        )
        ax.legend()
        ax.grid(True, alpha=0.2)
        ax.tick_params(axis="x", labelsize=8, rotation=45)
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

    st.dataframe(
        df.sort_values("local_time", ascending=False), use_container_width=True
    )

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
            dfp["timestamp"] = pd.to_datetime(
                dfp["timestamp"], utc=True, errors="coerce"
            )
            dfp["localtime"] = dfp["timestamp"].dt.tz_convert(tz)
            dfp = dfp.sort_values("localtime").dropna(subset=["localtime"])

            # ── 2. numeric columns ───────────────────────────────────────────────
            dfp["price"] = pd.to_numeric(dfp.get("price"), errors="coerce").fillna(0)
            dfp["qty"] = pd.to_numeric(dfp.get("qty"), errors="coerce").fillna(0)
            dfp["shares"] = pd.to_numeric(dfp.get("shares"), errors="coerce").fillna(0)
            dfp["action"] = dfp["action"].astype(str).str.lower().str.strip()

            # ── 3. cashflow per trade ─────────────────────────────────────────────
            def _cf(r):
                if r["action"] == "sell":
                    return r["qty"] * r["price"]
                if r["action"] == "buy":
                    return -r["qty"] * r["price"]
                return 0.0

            dfp["cf"] = dfp.apply(_cf, axis=1)
            dfp["cumcf"] = dfp["cf"].cumsum()
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
                            ic = float(
                                pd.to_numeric(_dp["initial_cash"], errors="coerce")
                                .fillna(0)
                                .iloc[0]
                            )
                    except Exception:
                        pass
                total_deps = ic + float(df_dep["amount"].clip(lower=0).sum())
                sym_allocated_deposits = total_deps / n_symbols

            # ── 6. PnL = trading gains only (deposits stripped) ──────────────────
            dfp["pnl_value"] = dfp["sym_equity"] - sym_allocated_deposits
            # Raw equity = trading result + deposit allocation (starts at ~$1500)
            dfp["sym_raw_equity"] = dfp["sym_equity"] + sym_allocated_deposits

            # ── 7. chart ─────────────────────────────────────────────────────────
            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=(
                    "🤖 Bot PnL (trading gains only)",
                    "🗃️ Account Equity (raw, includes deposits)",
                ),
            )
            fig.add_trace(
                go.Scatter(
                    x=dfp["localtime"],
                    y=dfp["pnl_value"],
                    mode="lines+markers",
                    name="PnL",
                    line=dict(color="#00b4d8", width=2),
                    hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>PnL: $%{y:,.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=dfp["localtime"],
                    y=dfp["sym_raw_equity"],
                    mode="lines+markers",
                    name="Raw Equity",
                    line=dict(color="#adb5bd", width=2, dash="dash"),
                    hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>Equity: $%{y:,.2f}<extra></extra>",
                ),
                row=1,
                col=2,
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
                        line_width=1,
                        line_dash="dot",
                        line_color="green" if amount > 0 else "red",
                        annotation_text=f"${amount/n_symbols:,.0f}",
                        annotation_position="top right",
                        row=1,
                        col=2,
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
            lambda r: (
                -(r["exec_qty"] * r["price"])
                if r["action"] == "buy"
                else +(r["exec_qty"] * r["price"]) if r["action"] == "sell" else 0.0
            ),
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

            if SHOW_DEBUG_BLOCKS:
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
            df_tr["timestamp"] = pd.to_datetime(
                df_tr["timestamp"], utc=True, errors="coerce"
            )
            df_tr["action"] = df_tr["action"].astype(str).str.lower().str.strip()
            df_tr["price"] = pd.to_numeric(df_tr.get("price"), errors="coerce")
            df_tr["qty"] = pd.to_numeric(df_tr.get("qty"), errors="coerce")
            df_tr = df_tr.dropna(subset=["timestamp", "action", "price"])
            df_tr = df_tr[df_tr["action"].isin(["buy", "sell"])].copy()
        except Exception:
            df_tr = None

    trace_options = [
        "Price",
        "Final prob",
        "Intraday prob",
        "Daily prob",
        "BUY",
        "SELL",
    ]
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
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["price"],
                name="Price",
                line=dict(color="black", width=2),
                yaxis="y1",
                hovertemplate="Price: %{y:.2f}<extra></extra>",
            )
        )

    # ---- FINAL PROBABILITY (right axis)
    if ("Final prob" in visible_traces) and (
        "finalprob" in df.columns or "final_prob" in df.columns
    ):
        prob_col = "finalprob" if "finalprob" in df.columns else "final_prob"
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df[prob_col],
                name="Final Probability",
                line=dict(color="blue", width=2),
                yaxis="y2",
                hovertemplate="Final prob: %{y:.3f}<extra></extra>",
            )
        )

    # ---- Intraday prob
    if ("Intraday prob" in visible_traces) and (
        "intradayprob" in df.columns or "intraday_prob" in df.columns
    ):
        ip_col = "intradayprob" if "intradayprob" in df.columns else "intraday_prob"
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df[ip_col],
                name="Intraday Prob",
                line=dict(color="orange", dash="dot"),
                yaxis="y2",
                opacity=0.6,
                hovertemplate="Intraday prob: %{y:.3f}<extra></extra>",
            )
        )

    # ---- Daily prob
    if ("Daily prob" in visible_traces) and (
        "dailyprob" in df.columns or "daily_prob" in df.columns
    ):
        dp_col = "dailyprob" if "dailyprob" in df.columns else "daily_prob"
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df[dp_col],
                name="Daily Prob",
                line=dict(color="green", dash="dash"),
                yaxis="y2",
                opacity=0.6,
                hovertemplate="Daily prob: %{y:.3f}<extra></extra>",
            )
        )

    # ---- BUY/SELL MARKERS (on price axis)
    if df_tr is not None and not df_tr.empty:
        buys = df_tr[df_tr["action"] == "buy"].copy()
        sells = df_tr[df_tr["action"] == "sell"].copy()

        if "BUY" in visible_traces and not buys.empty:
            fig.add_trace(
                go.Scatter(
                    x=buys["timestamp"],
                    y=buys["price"],
                    mode="markers",
                    name="BUY",
                    yaxis="y1",
                    marker=dict(
                        symbol="triangle-up",
                        size=12,
                        color="green",
                        line=dict(width=1, color="black"),
                    ),
                    customdata=buys[["qty"]].values,
                    hovertemplate=(
                        "<b>BUY</b><br>"
                        "Time: %{x|%Y-%m-%d %H:%M:%S} UTC<br>"
                        "Price: %{y:.2f}<br>"
                        "Qty: %{customdata[0]:g}<extra></extra>"
                    ),
                )
            )

        if "SELL" in visible_traces and not sells.empty:
            fig.add_trace(
                go.Scatter(
                    x=sells["timestamp"],
                    y=sells["price"],
                    mode="markers",
                    name="SELL",
                    yaxis="y1",
                    marker=dict(
                        symbol="triangle-down",
                        size=12,
                        color="red",
                        line=dict(width=1, color="black"),
                    ),
                    customdata=sells[["qty"]].values,
                    hovertemplate=(
                        "<b>SELL</b><br>"
                        "Time: %{x|%Y-%m-%d %H:%M:%S} UTC<br>"
                        "Price: %{y:.2f}<br>"
                        "Qty: %{customdata[0]:g}<extra></extra>"
                    ),
                )
            )

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
            (
                c
                for c in ["value", "equity", "total_value", "portfolio_value"]
                if c in df.columns
            ),
            None,
        )
        if value_col is None:
            st.error(f"No value column found. Columns: {df.columns.tolist()}")
            st.stop()

        df["total_equity"] = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0)

        # -------------------------------------------------
        # ACCOUNT CASH-FLOW ACCOUNTING
        # -------------------------------------------------
        #
        # daily_portfolio.external_flow:
        #   ALL account cash movements, including broker corrections.
        #   Used for PnL adjustment.
        #
        # df_dep / deposits_auto.csv:
        #   REAL investor deposits/withdrawals only.
        #   Used for contributed-capital reporting.
        # -------------------------------------------------

        df = df.sort_values("date").reset_index(drop=True).copy()

        df["date"] = pd.to_datetime(
            df["date"],
            utc=True,
            errors="coerce",
        )

        df = df.dropna(subset=["date", "total_equity"]).reset_index(drop=True)

        # -----------------------------------------
        # 1. PERFORMANCE CASH FLOWS
        # -----------------------------------------
        # Never overwrite this from deposits_auto.csv.
        # update_portfolio_data.py already wrote the complete
        # broker cash-flow history here.

        if "external_flow" in df.columns:
            df["external_flow"] = pd.to_numeric(
                df["external_flow"],
                errors="coerce",
            ).fillna(0.0)
        else:
            df["external_flow"] = 0.0

        # -----------------------------------------
        # 2. Recover capital present before first flow
        # -----------------------------------------

        first_equity = float(df["total_equity"].iloc[0])

        first_day_flow = float(df["external_flow"].iloc[0])

        # Alpaca portfolio-history equity for the first recorded
        # day already includes that day's external flow.
        initial_capital = first_equity - first_day_flow

        # Defensive fallback
        if not np.isfinite(initial_capital):
            initial_capital = first_equity

        if initial_capital < 0:
            initial_capital = 0.0

        # -----------------------------------------
        # 3. Cash-flow-adjusted account PnL
        # -----------------------------------------

        df["cum_external_flow"] = df["external_flow"].cumsum()

        df["net_cash_flow"] = initial_capital + df["cum_external_flow"]

        df["pnl_value"] = df["total_equity"] - df["net_cash_flow"]

        # -----------------------------------------
        # 3b. TIME-WEIGHTED RETURN (TWR)
        # -----------------------------------------
        #
        # TWR removes external cash flows from percentage performance.
        #
        # Withdrawal:
        #   reduces account equity
        #   but is added back mathematically through external_flow
        #   so it does NOT count as an investment loss.
        #
        # Deposit:
        #   increases account equity
        #   but is removed from performance.
        #
        # For each observed period:
        #
        #   return_t =
        #       (ending_equity - external_flow)
        #       / previous_equity
        #       - 1
        #
        # The first portfolio row has no prior observation,
        # so its return is defined as zero.
        # -----------------------------------------

        df["twr_period_return"] = 0.0

        prev_equity = df["total_equity"].shift(1)

        valid_prev_equity = (
            pd.to_numeric(
                prev_equity,
                errors="coerce",
            )
            > 0
        )

        df.loc[
            valid_prev_equity,
            "twr_period_return",
        ] = (
            df.loc[
                valid_prev_equity,
                "total_equity",
            ]
            - df.loc[
                valid_prev_equity,
                "external_flow",
            ]
        ) / prev_equity.loc[valid_prev_equity] - 1.0

        df["twr_period_return"] = (
            pd.to_numeric(
                df["twr_period_return"],
                errors="coerce",
            )
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0.0)
        )

        df["twr_growth"] = (1.0 + df["twr_period_return"]).cumprod()

        df["twr_cumulative_return"] = df["twr_growth"] - 1.0

        # -----------------------------------------
        # 4. REAL INVESTOR CAPITAL
        # -----------------------------------------

        df["investor_deposit_flow"] = 0.0
        df["investor_withdrawal_flow"] = 0.0

        if df_dep is not None and not df_dep.empty:

            investor_flows = df_dep.copy()

            investor_flows["date"] = pd.to_datetime(
                investor_flows["date"],
                utc=True,
                errors="coerce",
            )

            investor_flows["amount"] = pd.to_numeric(
                investor_flows["amount"],
                errors="coerce",
            ).fillna(0.0)

            investor_flows = (
                investor_flows.dropna(subset=["date"])
                .sort_values("date")
                .reset_index(drop=True)
            )

            # Use calendar dates for investor-capital reporting.
            investor_flows["day"] = investor_flows["date"].dt.tz_convert(tz).dt.date

            df["investor_day"] = df["date"].dt.tz_convert(tz).dt.date

            investor_flows["deposit"] = investor_flows["amount"].clip(lower=0.0)

            investor_flows["withdrawal"] = -investor_flows["amount"].clip(upper=0.0)

            investor_deposits = investor_flows.groupby("day")["deposit"].sum()

            investor_withdrawals = investor_flows.groupby("day")["withdrawal"].sum()

            df["investor_deposit_flow"] = (
                df["investor_day"].map(investor_deposits).fillna(0.0)
            )

            df["investor_withdrawal_flow"] = (
                df["investor_day"].map(investor_withdrawals).fillna(0.0)
            )

        df["cum_investor_deposits"] = df["investor_deposit_flow"].cumsum()

        df["cum_investor_withdrawals"] = df["investor_withdrawal_flow"].cumsum()

        # Capital that was already present before our first tracked
        # deposit is legitimate investor capital too.
        df["total_deposited"] = initial_capital + df["cum_investor_deposits"]

        df["total_withdrawn"] = df["cum_investor_withdrawals"]

        df["investor_net_capital"] = df["total_deposited"] - df["total_withdrawn"]

        # -------------------------------------------------
        # TOTAL PERFORMANCE STATS
        # -------------------------------------------------
        st.subheader("📊 Total Performance Stats")
        st.subheader("💰 Account Performance (cash-flow adjusted)")

        df_stats = df.copy()
        df_stats = (
            df_stats.dropna(subset=["date", "total_equity"])
            .sort_values("date")
            .reset_index(drop=True)
        )

        if len(df_stats) >= 2:
            # -----------------------------------------
            # Window
            # -----------------------------------------

            start_date = df_stats["date"].iloc[0]
            end_date = df_stats["date"].iloc[-1]

            elapsed_days = max(
                (end_date - start_date).total_seconds() / 86400.0,
                1.0,
            )

            start_equity = float(df_stats["total_equity"].iloc[0])

            end_equity = float(df_stats["total_equity"].iloc[-1])

            total_pnl = float(df_stats["pnl_value"].iloc[-1])

            # -----------------------------------------
            # Capital bases
            # -----------------------------------------

            starting_capital = float(initial_capital)

            net_invested_capital = float(df_stats["investor_net_capital"].iloc[-1])

            total_deposited_now = float(df_stats["total_deposited"].iloc[-1])

            total_withdrawn_now = float(df_stats["total_withdrawn"].iloc[-1])

            # -----------------------------------------
            # Performance returns
            # -----------------------------------------
            #
            # TWR = strategy percentage performance with deposits /
            # withdrawals removed from the return calculation.
            #
            # Capital ROI is retained as a separate informational metric:
            #
            #     cumulative trading PnL / current net invested capital
            #
            # Do NOT use Capital ROI as the performance-fee calculation.
            # -----------------------------------------

            account_twr = float(
                pd.to_numeric(
                    df_stats["twr_cumulative_return"],
                    errors="coerce",
                )
                .fillna(0.0)
                .iloc[-1]
            )

            annual_account_return = (
                (1.0 + account_twr) ** (365.25 / elapsed_days) - 1.0
                if account_twr > -1.0
                else -1.0
            )

            capital_roi_base = max(
                net_invested_capital,
                1e-9,
            )

            capital_roi = total_pnl / capital_roi_base

            # -----------------------------------------
            # Cash-flow-neutral TWR performance curve
            # -----------------------------------------

            df_stats["performance_index"] = (
                pd.to_numeric(
                    df_stats["twr_growth"],
                    errors="coerce",
                )
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                .ffill()
                .fillna(1.0)
            )

            df_stats = df_stats[df_stats["performance_index"] > 0].copy()

            # -----------------------------------------
            # Drawdown
            # -----------------------------------------

            if len(df_stats) >= 2:
                df_stats["running_peak"] = df_stats["performance_index"].cummax()

                df_stats["drawdown"] = (
                    df_stats["performance_index"] / df_stats["running_peak"] - 1.0
                )

                max_drawdown = float(df_stats["drawdown"].min())
            else:
                max_drawdown = 0.0

            # -----------------------------------------
            # Daily TWR returns
            # -----------------------------------------

            daily_curve = (
                df_stats.set_index("date")[["performance_index"]]
                .resample("1D")
                .last()
                .dropna()
                .copy()
            )

            daily_curve["ret"] = daily_curve["performance_index"].pct_change()

            daily_rets = (
                daily_curve["ret"]
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                .dropna()
            )

            # -----------------------------------------
            # Sharpe / volatility
            # -----------------------------------------

            if len(daily_rets) >= 2 and daily_rets.std() > 0:
                sharpe = float((daily_rets.mean() / daily_rets.std()) * (252**0.5))

                volatility = float(daily_rets.std() * (252**0.5))
            else:
                sharpe = None
                volatility = None

            # Rolling 30-day Sharpe (optional, but you liked it)
            rolling_sharpe_30 = None
            if len(daily_rets) >= 30:
                roll_mean = daily_curve["ret"].rolling(window=30).mean()
                roll_std = daily_curve["ret"].rolling(window=30).std()
                roll_sharpe_daily = roll_mean / roll_std
                roll_sharpe_annual = roll_sharpe_daily * (252**0.5)
                roll_sharpe_annual = roll_sharpe_annual.replace(
                    [float("inf"), float("-inf")], pd.NA
                ).dropna()
                if not roll_sharpe_annual.empty:
                    rolling_sharpe_30 = float(roll_sharpe_annual.iloc[-1])

            # ---------- CLOSED-TRADE STATS (unchanged) ----------
            all_cycle_pnls = []
            for sym in symbols:
                trade_path = get_trade_log_file(sym)
                if not os.path.exists(trade_path):
                    continue
                try:
                    dft = pd.read_csv(trade_path)
                    dft["timestamp"] = pd.to_datetime(
                        dft["timestamp"], utc=True, errors="coerce"
                    )
                    dft["action"] = dft["action"].astype(str).str.lower().str.strip()
                    dft["price"] = pd.to_numeric(dft.get("price"), errors="coerce")
                    dft["shares"] = pd.to_numeric(dft.get("shares"), errors="coerce")

                    dft = (
                        dft.dropna(subset=["timestamp", "action", "price", "shares"])
                        .sort_values("timestamp")
                        .copy()
                    )

                    if "shares_after" not in dft.columns:
                        dft["shares_after"] = dft["shares"]
                    else:
                        dft["shares_after"] = pd.to_numeric(
                            dft["shares_after"], errors="coerce"
                        ).fillna(dft["shares"])

                    if "shares_before" not in dft.columns:
                        dft["shares_before"] = dft["shares_after"].shift(1).fillna(0.0)
                    else:
                        dft["shares_before"] = pd.to_numeric(
                            dft["shares_before"], errors="coerce"
                        ).fillna(0.0)

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
                        lambda r: (
                            -(r["exec_qty"] * r["price"])
                            if r["action"] == "buy"
                            else (
                                +(r["exec_qty"] * r["price"])
                                if r["action"] == "sell"
                                else 0.0
                            )
                        ),
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
                profit_factor = (
                    (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
                )
                avg_trade = float(s.mean())
                closed_trades = int(len(s))
            else:
                win_rate = None
                profit_factor = None
                avg_trade = None
                closed_trades = 0

            # -------------------------------------------------
            # ACCOUNT PERFORMANCE METRICS
            # -------------------------------------------------

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Account Equity",
                f"${end_equity:,.2f}",
            )

            c2.metric(
                "Total PnL",
                f"${total_pnl:,.2f}",
            )

            c3.metric(
                "TWR",
                f"{account_twr * 100:.2f}%",
            )

            c4.metric(
                "Annualized TWR",
                f"{annual_account_return * 100:.2f}%",
            )

            c5, c6, c7, c8 = st.columns(4)

            c5.metric(
                "Capital ROI",
                f"{capital_roi * 100:.2f}%",
            )

            c6.metric(
                "Net Invested Capital",
                f"${net_invested_capital:,.2f}",
            )

            c7.metric(
                "Sharpe Ratio",
                "N/A" if sharpe is None else f"{sharpe:.2f}",
            )

            c8.metric(
                "Max Drawdown",
                f"{max_drawdown * 100:.2f}%",
            )

            c9, c10, c11, c12 = st.columns(4)

            c9.metric(
                "Volatility",
                "N/A" if volatility is None else f"{volatility * 100:.2f}%",
            )

            c10.metric(
                "Win Rate",
                "N/A" if win_rate is None else f"{win_rate * 100:.1f}%",
            )

            c11.metric(
                "Profit Factor",
                (
                    "∞"
                    if profit_factor == float("inf")
                    else ("N/A" if profit_factor is None else f"{profit_factor:.2f}")
                ),
            )

            c12.metric(
                "Closed Trades",
                f"{closed_trades}",
            )

            st.caption(
                "TWR measures strategy percentage performance with deposits and "
                "withdrawals neutralized. Capital ROI is cumulative trading PnL "
                "divided by current net invested capital."
            )

            # -------------------------------------------------
            # INVESTOR / PERFORMANCE FEE KPIs
            # -------------------------------------------------

            with st.expander(
                "📈 Investor & Performance Fee KPIs",
                expanded=False,
            ):
                # -----------------------------------------
                # Current investor-capital snapshot
                # -----------------------------------------

                c_inv1, c_inv2, c_inv3, c_inv4 = st.columns(4)

                c_inv1.metric(
                    "Starting Capital",
                    f"${starting_capital:,.2f}",
                )

                c_inv2.metric(
                    "Total Investor Deposits",
                    f"${total_deposited_now - starting_capital:,.2f}",
                )

                c_inv3.metric(
                    "Total Withdrawals",
                    f"${total_withdrawn_now:,.2f}",
                )

                c_inv4.metric(
                    "Net Invested Capital",
                    f"${net_invested_capital:,.2f}",
                )

                c_inv5, c_inv6, c_inv7, c_inv8 = st.columns(4)

                c_inv5.metric(
                    "Cumulative Trading PnL",
                    f"${total_pnl:,.2f}",
                )

                c_inv6.metric(
                    "TWR",
                    f"{account_twr * 100:.2f}%",
                )

                c_inv7.metric(
                    "Annualized TWR",
                    f"{annual_account_return * 100:.2f}%",
                )

                c_inv8.metric(
                    "Capital ROI",
                    f"{capital_roi * 100:.2f}%",
                )

                st.caption(
                    "Withdrawals reduce account equity and net invested capital, "
                    "but do not reduce trading PnL or TWR."
                )

                st.divider()

                # -------------------------------------------------
                # ANNUAL INVESTOR FEE BREAKDOWN
                # -------------------------------------------------

                st.divider()

                st.markdown("### 💵 Annual Investor Fee Breakdown")

                st.caption(
                    f"Fee model: {RESOURCE_FEE_PCT:.2f}% annual resource fee, "
                    f"{HURDLE_PCT:.2f}% hurdle, "
                    f"{PERFORMANCE_FEE_PCT:.2f}% performance fee on profit "
                    f"above the hurdle after the resource fee."
                )

                st.caption(
                    "Completed fee years use the full annual fee terms. "
                    "The current incomplete fee year is shown as an estimate-to-date "
                    "with the annual resource fee and hurdle prorated by elapsed time."
                )

                max_years_back = 20
                fee_rows = []

                launch_date = start_date.normalize()
                final_date = end_date.normalize()

                for k in range(max_years_back):

                    period_start = launch_date + pd.DateOffset(years=k)

                    scheduled_period_end = launch_date + pd.DateOffset(years=k + 1)

                    if period_start > final_date:
                        break

                    # Latest available observation for current incomplete period.
                    effective_end = min(
                        scheduled_period_end,
                        final_date,
                    )

                    is_complete = final_date >= scheduled_period_end

                    period_mask = (df_stats["date"] >= period_start) & (
                        df_stats["date"] <= effective_end
                    )

                    df_period = df_stats.loc[period_mask].sort_values("date").copy()

                    if df_period.empty:
                        continue

                    # -------------------------------------------------
                    # INVESTOR CAPITAL BASE FOR FEE PERIOD
                    # -------------------------------------------------
                    #
                    # We cannot use only the opening capital because the
                    # account may begin the fee year at $0 and receive
                    # deposits during the year.
                    #
                    # Fee capital is therefore:
                    #
                    #   opening investor capital
                    # + time-weighted deposits
                    # - time-weighted withdrawals
                    #
                    # A flow early in the period receives almost full
                    # weight. A flow near the end receives little weight.
                    # -------------------------------------------------

                    before_period = df[df["date"] < period_start].sort_values("date")

                    if not before_period.empty:
                        opening_investor_capital = float(
                            pd.to_numeric(
                                before_period["investor_net_capital"],
                                errors="coerce",
                            )
                            .fillna(0.0)
                            .iloc[-1]
                        )
                    else:
                        opening_investor_capital = 0.0

                    opening_investor_capital = max(
                        opening_investor_capital,
                        0.0,
                    )

                    # -------------------------------------------------
                    # Investor deposits / withdrawals inside period
                    # -------------------------------------------------

                    period_deposits = 0.0
                    period_withdrawals = 0.0
                    weighted_net_flows = 0.0

                    # Use the actual observed fee-period dates.
                    capital_period_start = df_period["date"].iloc[0]

                    capital_period_end = df_period["date"].iloc[-1]

                    period_seconds = max(
                        (capital_period_end - capital_period_start).total_seconds(),
                        1.0,
                    )

                    if df_dep is not None and not df_dep.empty:
                        investor_flows_period = df_dep.copy()

                        investor_flows_period["date"] = pd.to_datetime(
                            investor_flows_period["date"],
                            utc=True,
                            errors="coerce",
                        )

                        investor_flows_period["amount"] = pd.to_numeric(
                            investor_flows_period["amount"],
                            errors="coerce",
                        ).fillna(0.0)

                        flow_mask = (
                            investor_flows_period["date"] >= capital_period_start
                        ) & (investor_flows_period["date"] <= capital_period_end)

                        investor_flows_period = (
                            investor_flows_period.loc[flow_mask]
                            .sort_values("date")
                            .copy()
                        )

                        if not investor_flows_period.empty:

                            period_deposits = float(
                                investor_flows_period["amount"].clip(lower=0.0).sum()
                            )

                            period_withdrawals = float(
                                -investor_flows_period["amount"].clip(upper=0.0).sum()
                            )

                            for _, flow_row in investor_flows_period.iterrows():
                                flow_date = flow_row["date"]

                                flow_amount = float(flow_row["amount"])

                                # Portion of the fee period for which
                                # this money was actually invested.
                                remaining_seconds = max(
                                    (capital_period_end - flow_date).total_seconds(),
                                    0.0,
                                )

                                weight = min(
                                    max(
                                        remaining_seconds / period_seconds,
                                        0.0,
                                    ),
                                    1.0,
                                )

                                weighted_net_flows += flow_amount * weight

                    # -------------------------------------------------
                    # Effective fee capital
                    # -------------------------------------------------

                    period_fee_capital = max(
                        opening_investor_capital + weighted_net_flows,
                        0.0,
                    )

                    # -------------------------------------------------
                    # ACTUAL fee-period TWR
                    # -------------------------------------------------
                    #
                    # Do NOT use annualized TWR for fee calculation.
                    #
                    # The fee should use the actual percentage earned
                    # during this fee period.
                    # -------------------------------------------------

                    period_returns = (
                        pd.to_numeric(
                            df_period["twr_period_return"],
                            errors="coerce",
                        )
                        .replace(
                            [np.inf, -np.inf],
                            np.nan,
                        )
                        .dropna()
                    )

                    if not period_returns.empty:
                        period_twr = float((1.0 + period_returns).prod() - 1.0)
                    else:
                        period_twr = 0.0

                    period_return_pct = period_twr * 100.0

                    # -------------------------------------------------
                    # Period duration / proration
                    # -------------------------------------------------

                    actual_start = df_period["date"].iloc[0]

                    actual_end = df_period["date"].iloc[-1]

                    elapsed_period_days = max(
                        (actual_end - actual_start).total_seconds() / 86400.0,
                        1.0,
                    )

                    if is_complete:
                        period_fraction = 1.0
                    else:
                        period_fraction = min(
                            elapsed_period_days / 365.25,
                            1.0,
                        )

                    # -------------------------------------------------
                    # RESOURCE FEE BASE
                    # -------------------------------------------------
                    #
                    # Complete fee period:
                    #   charge resource fee on net deposits made in that period.
                    #
                    # Current incomplete period:
                    #   charge resource fee on opening capital carried forward
                    #   from the previous completed period.
                    # -------------------------------------------------

                    if is_complete:
                        # Completed period:
                        # resource fee based on net capital that funded that period
                        resource_fee_base = max(
                            period_deposits - period_withdrawals,
                            0.0,
                        )

                    else:
                        # Current / incomplete period:
                        # opening capital carried from the previous period
                        # + any new deposits made during this period
                        # - any withdrawals made during this period
                        resource_fee_base = max(
                            opening_investor_capital
                            + period_deposits
                            - period_withdrawals,
                            0.0,
                        )

                    # -------------------------------------------------
                    # Apply your fee_model.py rules
                    # -------------------------------------------------

                    fee_result = calculate_investor_fees(
                        starting_capital=period_fee_capital,
                        period_return_pct=period_return_pct,
                        resource_fee_pct=RESOURCE_FEE_PCT,
                        hurdle_pct=HURDLE_PCT,
                        performance_fee_pct=PERFORMANCE_FEE_PCT,
                        period_fraction=period_fraction,
                        resource_fee_base=resource_fee_base,
                    )
                    # -------------------------------------------------
                    # Actual investor cash flows during fee period
                    # shown separately for transparency.
                    #
                    # They do NOT determine investment performance.
                    # -------------------------------------------------

                    period_deposits = 0.0
                    period_withdrawals = 0.0

                    if df_dep is not None and not df_dep.empty:
                        dep_dates = pd.to_datetime(
                            df_dep["date"],
                            utc=True,
                            errors="coerce",
                        )

                        investor_flow_mask = (dep_dates >= period_start) & (
                            dep_dates <= effective_end
                        )

                        period_flows = pd.to_numeric(
                            df_dep.loc[
                                investor_flow_mask,
                                "amount",
                            ],
                            errors="coerce",
                        ).fillna(0.0)

                        period_deposits = float(period_flows.clip(lower=0.0).sum())

                        period_withdrawals = float(-period_flows.clip(upper=0.0).sum())

                    # -------------------------------------------------
                    # Account trading PnL during period
                    # -------------------------------------------------

                    before_period_pnl = df[df["date"] < period_start].sort_values(
                        "date"
                    )

                    if not before_period_pnl.empty:
                        cumulative_pnl_start = float(
                            before_period_pnl["pnl_value"].iloc[-1]
                        )
                    else:
                        cumulative_pnl_start = 0.0

                    cumulative_pnl_end = float(df_period["pnl_value"].iloc[-1])

                    actual_period_trading_pnl = (
                        cumulative_pnl_end - cumulative_pnl_start
                    )

                    # -------------------------------------------------
                    # Display row
                    # -------------------------------------------------

                    status = "Complete" if is_complete else "Current estimate"

                    fee_rows.append(
                        {
                            "Fee Period": (
                                f"{period_start:%Y-%m-%d}"
                                f" → "
                                f"{effective_end:%Y-%m-%d}"
                            ),
                            "Status": status,
                            "Opening Capital": f"${opening_investor_capital:,.2f}",
                            "Fee Capital Base": f"${period_fee_capital:,.2f}",
                            "Deposits": f"${period_deposits:,.2f}",
                            "Withdrawals": f"${period_withdrawals:,.2f}",
                            "Actual Return": (
                                f"{fee_result['period_return_pct']:.2f}%"
                            ),
                            "Gross Profit": (f"${fee_result['gross_profit']:,.2f}"),
                            "Resource Fee": (f"${fee_result['resource_fee']:,.2f}"),
                            "Resource Fee Base": (
                                f"${fee_result['resource_fee_base']:,.2f}"
                            ),
                            "Hurdle": (f"{fee_result['hurdle_pct']:.2f}%"),
                            "Hurdle Profit": (f"${fee_result['hurdle_profit']:,.2f}"),
                            "Feeable Profit": (f"${fee_result['feeable_profit']:,.2f}"),
                            "Performance Fee": (
                                f"${fee_result['performance_fee']:,.2f}"
                            ),
                            "Total Fees": (f"${fee_result['total_fees']:,.2f}"),
                            "Ending Capital": (f"${fee_result['ending_capital']:,.2f}"),
                            "Actual Account PnL": (
                                f"${actual_period_trading_pnl:,.2f}"
                            ),
                            "Deposits": (f"${period_deposits:,.2f}"),
                            "Withdrawals": (f"${period_withdrawals:,.2f}"),
                        }
                    )

                if fee_rows:
                    df_fees = pd.DataFrame(fee_rows)

                    st.dataframe(
                        df_fees,
                        use_container_width=True,
                        hide_index=True,
                    )

                else:
                    st.info(
                        "Not enough history yet to calculate " "an investor fee period."
                    )

                if fee_rows:

                    st.markdown("### Current Fee Period")

                    # Reuse the most recently calculated fee result.
                    r = fee_result

                    st.markdown(f"""
                **Starting Capital:** ${r['starting_capital']:,.2f}

                **Actual Return:** {r['period_return_pct']:.2f}%

                - Gross Profit: **${r['gross_profit']:,.2f}**
                - Resource Fee: **${r['resource_fee']:,.2f}**
                - Hurdle Profit: **${r['hurdle_profit']:,.2f}**
                - Feeable Profit: **${r['feeable_profit']:,.2f}**
                - Performance Fee: **${r['performance_fee']:,.2f}**
                - Total Fees: **${r['total_fees']:,.2f}**
                - Ending Capital for Investor: **${r['ending_capital']:,.2f}**
                """)

            # ---------- Broker vs Account Equity ----------
            broker_equity = None
            try:
                broker_equity = float(
                    account_cache.get_account().get("equity", 0.0) or 0.0
                )
            except Exception:
                broker_equity = None

            if broker_equity is not None:
                st.markdown(
                    f"""
                    <div style="font-size: 0.85rem; color: var(--text-muted, rgba(255,255,255,0.65)); line-height: 1.5; margin-top: 0.25rem;">
                        <strong>Broker Equity</strong> ${broker_equity:,.2f}<br>
                        <strong>Account Equity</strong> ${end_equity:,.2f}<br>
                        Account Equity is calculated from the local daily portfolio file.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # -------------------------------------------------
            # Strategy Performance
            # Uses the SAME cash-flow-adjusted accounting calculated above.
            # Do not independently reconstruct initial capital / deposits here.
            # -------------------------------------------------

            with st.expander(
                "📈 Strategy Performance (cash-flow adjusted)",
                expanded=False,
            ):

                # -------------------------------------------------
                # Core performance numbers
                # -------------------------------------------------

                strategy_pnl = float(total_pnl)
                strategy_twr = float(account_twr)
                strategy_annual_twr = float(annual_account_return)

                # Capital actually supplied by the investor:
                # starting capital + deposits - withdrawals.
                strategy_net_capital = float(net_invested_capital)

                strategy_capital_roi = (
                    strategy_pnl / strategy_net_capital
                    if strategy_net_capital > 0
                    else None
                )

                # -------------------------------------------------
                # Main metrics
                # -------------------------------------------------

                c1s, c2s, c3s, c4s = st.columns(4)

                c1s.metric(
                    "Net Invested Capital",
                    f"${strategy_net_capital:,.2f}",
                )

                c2s.metric(
                    "Total PnL",
                    f"${strategy_pnl:,.2f}",
                )

                c3s.metric(
                    "TWR",
                    f"{strategy_twr * 100:.2f}%",
                )

                c4s.metric(
                    "Annualized TWR",
                    f"{strategy_annual_twr * 100:.2f}%",
                )

                c5s, c6s, c7s, c8s = st.columns(4)

                c5s.metric(
                    "Capital ROI",
                    (
                        "N/A"
                        if strategy_capital_roi is None
                        else f"{strategy_capital_roi * 100:.2f}%"
                    ),
                )

                c6s.metric(
                    "Sharpe",
                    ("N/A" if sharpe is None else f"{sharpe:.4f}"),
                )

                c7s.metric(
                    "Volatility",
                    ("N/A" if volatility is None else f"{volatility * 100:.2f}%"),
                )

                c8s.metric(
                    "Max Drawdown",
                    f"{max_drawdown * 100:.2f}%",
                )

                # -------------------------------------------------
                # Additional capital information
                # -------------------------------------------------

                st.markdown(f"""
                    **Investor capital**

                    - Starting capital: **${starting_capital:,.2f}**
                    - Investor deposits: **${total_deposited_now - starting_capital:,.2f}**
                    - Investor withdrawals: **${total_withdrawn_now:,.2f}**
                    - Net invested capital: **${strategy_net_capital:,.2f}**
                    - Current account equity: **${end_equity:,.2f}**
                    - Cash-flow-adjusted trading PnL: **${strategy_pnl:,.2f}**
                    """)

                st.caption(
                    "TWR, Sharpe, volatility and drawdown are calculated from the "
                    "cash-flow-neutral performance series. Deposits and withdrawals "
                    "therefore do not count as trading gains or losses."
                )

        else:
            st.info(
                "Not enough portfolio history yet to compute total performance stats."
            )

        st.write("**Cash-flow / performance summary:**")

        st.dataframe(
            df[
                [
                    "date",
                    "total_equity",
                    "external_flow",
                    "cum_external_flow",
                    "investor_deposit_flow",
                    "investor_withdrawal_flow",
                    "investor_net_capital",
                    "pnl_value",
                    "twr_cumulative_return",
                ]
            ].tail(10),
            use_container_width=True,
        )

        fig_eq = go.Figure()
        fig_eq.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["total_equity"],
                mode="lines",
                name="Account Equity",
                line=dict(width=2, color="#636efa"),
                hovertemplate=(
                    "<b>Account Equity</b><br>"
                    "Date: %{x|%Y-%m-%d}<br>"
                    "Value: $%{y:,.2f}"
                    "<extra></extra>"
                ),
            )
        )

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

                fig_eq.add_trace(
                    go.Scatter(
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
                        ),
                    )
                )

        fig_eq.update_layout(
            title="💼 Account Equity",
            xaxis_title="Date",
            yaxis_title="Account Equity ($)",
            hovermode="x unified",
            template="plotly_white",
            height=420,
        )
        st.plotly_chart(fig_eq, use_container_width=True, key="portfolio_chart")

        if SHOW_DEBUG_BLOCKS:
            with st.expander("🔍 PnL debug around deposits"):
                # Select the key columns
                debug_cols = [
                    "date",
                    "total_equity",
                    "external_flow",
                    "net_cash_flow",
                    "pnl_value",
                ]

                df_debug = df.copy()
                df_debug = df_debug.sort_values("date")

                # Option 1: show the last 50 rows
                st.write("Last 50 rows of portfolio data:")
                st.dataframe(df_debug[debug_cols].tail(50), use_container_width=True)

                # Option 2 (better): filter manually around the spike date
                # Example: between 2026-03-01 and 2026-04-15
                # df_window = df_debug[(df_debug["date"] >= "2026-03-01") & (df_debug["date"] <= "2026-04-15")]
                # st.dataframe(df_window[debug_cols], use_container_width=True)

        st.subheader("🤖 Bot PnL — Trading Gains Only (deposits stripped)")
        if len(df) < 2:
            st.info(
                "⏳ Not enough data yet to show PnL curve — needs at least 2 days of portfolio history."
            )
        else:
            df_pnl = df.sort_values("date").copy()
            # Ensure pnl_value is numeric and forward-filled
            df_pnl["pnl_value"] = pd.to_numeric(
                df_pnl["pnl_value"], errors="coerce"
            ).fillna(method="ffill")

            # Re-base so curve starts at 0 (pure PnL from first point)
            first_pnl = float(df_pnl["pnl_value"].iloc[0])
            df_pnl["pnl_rebased"] = df_pnl["pnl_value"] - first_pnl

            fig_pnl = go.Figure()
            fig_pnl.add_trace(
                go.Scatter(
                    x=df_pnl["date"],
                    y=df_pnl["pnl_rebased"],
                    mode="lines",
                    name="PnL-Only Equity",
                    line=dict(width=2, color="#00b4d8"),
                    hovertemplate=(
                        "<b>Bot PnL</b><br>"
                        "Date: %{x|%Y-%m-%d}<br>"
                        "Value: $%{y:,.2f}"
                        "<extra></extra>"
                    ),
                )
            )
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
