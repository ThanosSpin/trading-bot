"""Trade-cycle reconstruction and performance attribution helpers."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


ATTRIBUTION_COLUMNS = [
    "model_used",
    "regime",
    "entry_reason",
    "exit_reason",
    "decision_probability",
    "buy_threshold",
    "sell_threshold",
    "position_value_before",
    "execution_id",
    "client_order_id",
    "source_commit",
    "source_checkout",
]


def _text(value, default="unknown") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    value = str(value).strip()
    return value if value and value.lower() != "nan" else default


def _joined_unique(values: Iterable[object]) -> str:
    result = []
    for value in values:
        item = _text(value)
        if item != "unknown" and item not in result:
            result.append(item)
    return ", ".join(result) if result else "unknown"


def normalize_trade_log(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize old and enriched trade logs into one execution schema."""
    if frame is None or frame.empty:
        return pd.DataFrame()

    df = frame.copy()
    for column in ATTRIBUTION_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan

    df["timestamp"] = pd.to_datetime(df.get("timestamp"), utc=True, errors="coerce")
    df["action"] = df.get("action", "").astype(str).str.lower().str.strip()
    for column in ("price", "qty", "shares", "shares_before", "shares_after"):
        if column not in df.columns:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["timestamp", "price"]).sort_values("timestamp").copy()
    if df.empty:
        return df

    df["shares_after"] = df["shares_after"].fillna(df["shares"])
    inferred_before = df["shares_after"].shift(1).fillna(0.0)
    df["shares_before"] = df["shares_before"].fillna(inferred_before)

    buy_delta = (df["shares_after"] - df["shares_before"]).clip(lower=0.0)
    sell_delta = (df["shares_before"] - df["shares_after"]).clip(lower=0.0)
    logged_qty = df["qty"].fillna(0.0).clip(lower=0.0)
    df["exec_qty"] = np.where(
        df["action"].eq("buy"),
        np.where(buy_delta > 0.0, buy_delta, logged_qty),
        np.where(
            df["action"].eq("sell"),
            np.where(sell_delta > 0.0, sell_delta, logged_qty),
            0.0,
        ),
    )
    return df[df["exec_qty"] > 0.0].copy()


def build_trade_cycles(frame: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    """Return completed flat-to-flat position cycles with attribution metadata."""
    df = normalize_trade_log(frame)
    if df.empty:
        return pd.DataFrame()

    cycles = []
    rows = []
    epsilon = 1e-9

    for _, row in df.iterrows():
        before = float(row["shares_before"] or 0.0)
        after = float(row["shares_after"] or 0.0)
        was_flat = abs(before) <= epsilon
        now_flat = abs(after) <= epsilon

        if not rows and was_flat and not now_flat and row["action"] == "buy":
            rows = [row]
        elif rows:
            rows.append(row)
        else:
            # A historical log may begin in the middle of a position. Skip that
            # incomplete cycle rather than inventing its cost basis.
            continue

        if not (rows and row["action"] == "sell" and now_flat):
            continue

        cycle = pd.DataFrame(rows)
        buys = cycle[cycle["action"].eq("buy")]
        sells = cycle[cycle["action"].eq("sell")]
        invested = float((buys["exec_qty"] * buys["price"]).sum())
        proceeds = float((sells["exec_qty"] * sells["price"]).sum())
        pnl = proceeds - invested
        first = buys.iloc[0]
        last = sells.iloc[-1]

        cycles.append(
            {
                "symbol": _text(symbol or first.get("symbol"), "unknown").upper(),
                "entry_time": first["timestamp"],
                "exit_time": last["timestamp"],
                "holding_hours": (
                    last["timestamp"] - first["timestamp"]
                ).total_seconds()
                / 3600.0,
                "buy_count": int(len(buys)),
                "sell_count": int(len(sells)),
                "invested_capital": invested,
                "sale_proceeds": proceeds,
                "pnl_dollars": pnl,
                "return_pct": pnl / invested if invested > 0.0 else np.nan,
                "model_used": _text(first.get("model_used")),
                "regime": _text(first.get("regime")),
                "entry_reason": _text(first.get("entry_reason")),
                "entry_reasons": _joined_unique(buys["entry_reason"]),
                "exit_reason": _text(last.get("exit_reason")),
                "exit_reasons": _joined_unique(sells["exit_reason"]),
                "decision_probability": pd.to_numeric(
                    pd.Series([first.get("decision_probability")]), errors="coerce"
                ).iloc[0],
                "buy_threshold": pd.to_numeric(
                    pd.Series([first.get("buy_threshold")]), errors="coerce"
                ).iloc[0],
                "sell_threshold": pd.to_numeric(
                    pd.Series([first.get("sell_threshold")]), errors="coerce"
                ).iloc[0],
                "position_value_before": pd.to_numeric(
                    pd.Series([first.get("position_value_before")]), errors="coerce"
                ).iloc[0],
            }
        )
        rows = []

    return pd.DataFrame(cycles)


def attribution_summary(cycles: pd.DataFrame, group_columns=None) -> pd.DataFrame:
    """Aggregate completed cycles using both dollar and normalized returns."""
    if cycles is None or cycles.empty:
        return pd.DataFrame()
    group_columns = group_columns or [
        "symbol",
        "model_used",
        "regime",
        "entry_reason",
        "exit_reason",
    ]

    def summarize(group: pd.DataFrame) -> pd.Series:
        pnl = pd.to_numeric(group["pnl_dollars"], errors="coerce").dropna()
        returns = pd.to_numeric(group["return_pct"], errors="coerce").dropna()
        dollar_loss = float(-pnl[pnl < 0].sum())
        return_loss = float(-returns[returns < 0].sum())
        return pd.Series(
            {
                "cycles": int(len(group)),
                "win_rate_pct": float((pnl > 0).mean() * 100.0),
                "pnl_dollars": float(pnl.sum()),
                "avg_pnl_dollars": float(pnl.mean()),
                "avg_return_pct": float(returns.mean() * 100.0),
                "dollar_profit_factor": (
                    float(pnl[pnl > 0].sum()) / dollar_loss
                    if dollar_loss > 0.0
                    else np.inf
                ),
                "return_profit_factor": (
                    float(returns[returns > 0].sum()) / return_loss
                    if return_loss > 0.0
                    else np.inf
                ),
            }
        )

    rows = []
    for keys, group in cycles.groupby(group_columns, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, keys))
        row.update(summarize(group).to_dict())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["pnl_dollars", "cycles"], ascending=[False, False]
    )
