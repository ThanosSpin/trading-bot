"""Paper-only earnings-gap opening continuation strategy."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import pytz


NY_TZ = pytz.timezone("America/New_York")


def _record_evaluation(log_dir, timestamp: datetime, result: Dict[str, object]) -> None:
    path = Path(log_dir) / "earnings_opening_signals.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "timestamp",
        "symbol",
        "qualified",
        "reason",
        "gap_return",
        "momentum_5m",
        "opening_return",
        "spy_open_return",
        "relative_open_return",
        "price",
        "vwap",
        "opening_high",
        "volume_pace",
        "checks",
    ]
    row = {name: result.get(name) for name in columns}
    row["timestamp"] = timestamp.isoformat()
    row["checks"] = json.dumps(result.get("checks") or {}, sort_keys=True)
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _regular_session(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    result = frame.copy()
    result.index = pd.to_datetime(result.index, utc=True, errors="coerce")
    result = result[~result.index.isna()].sort_index()
    index_ny = result.index.tz_convert(NY_TZ)
    minute = index_ny.hour * 60 + index_ny.minute
    return result[(minute >= 9 * 60 + 30) & (minute < 16 * 60)]


def _session_rows(frame: pd.DataFrame, session_date) -> pd.DataFrame:
    regular = _regular_session(frame)
    if regular.empty:
        return regular
    dates = regular.index.tz_convert(NY_TZ).date
    return regular[dates == session_date]


def _opening_volume_pace(
    history: pd.DataFrame,
    today_rows: pd.DataFrame,
    session_date,
) -> Optional[float]:
    bars_elapsed = len(today_rows)
    if bars_elapsed < 2:
        return None
    regular = _regular_session(history)
    if regular.empty:
        return None
    dates = pd.Series(regular.index.tz_convert(NY_TZ).date, index=regular.index)
    prior_totals = []
    for date_value in sorted(set(dates)):
        if date_value >= session_date:
            continue
        session = regular[dates == date_value].head(bars_elapsed)
        if len(session) == bars_elapsed:
            prior_totals.append(float(session["Volume"].sum()))
    if not prior_totals:
        return None
    baseline = float(pd.Series(prior_totals[-20:]).median())
    if baseline <= 0:
        return None
    return float(today_rows["Volume"].sum()) / baseline


def evaluate_opening_bars(
    symbol: str,
    symbol_history: pd.DataFrame,
    spy_history: pd.DataFrame,
    previous_close: float,
    now_ny: datetime,
    min_gap: float = 0.02,
    min_volume_pace: float = 1.50,
    min_relative_return: float = 0.0,
) -> Dict[str, object]:
    """Evaluate completed five-minute bars without placing an order."""
    symbol = symbol.upper()
    session_date = now_ny.date()
    rows = _session_rows(symbol_history, session_date)
    spy_rows = _session_rows(spy_history, session_date)
    completed_cutoff = pd.Timestamp(now_ny).tz_convert("UTC") - pd.Timedelta(minutes=5)
    rows = rows[rows.index <= completed_cutoff]
    spy_rows = spy_rows[spy_rows.index <= completed_cutoff]
    result: Dict[str, object] = {"qualified": False, "symbol": symbol}

    if len(rows) < 2 or len(spy_rows) < 2:
        return {**result, "reason": "waiting for two completed five-minute bars"}
    if previous_close is None or float(previous_close) <= 0:
        return {**result, "reason": "previous close unavailable"}

    # The first regular-session bar defines the opening range. The latest
    # completed bar must close above it rather than merely trade through it.
    first = rows.iloc[0]
    latest = rows.iloc[-1]
    prior = rows.iloc[-2]
    spy_first = spy_rows.iloc[0]
    spy_latest = spy_rows.iloc[-1]
    typical = (rows["High"] + rows["Low"] + rows["Close"]) / 3.0
    cumulative_volume = rows["Volume"].cumsum()
    if float(cumulative_volume.iloc[-1]) <= 0:
        return {**result, "reason": "opening volume unavailable"}
    vwap = float((typical * rows["Volume"]).cumsum().iloc[-1] / cumulative_volume.iloc[-1])

    gap_return = float(first["Open"]) / float(previous_close) - 1.0
    momentum = float(latest["Close"]) / float(prior["Close"]) - 1.0
    symbol_open_return = float(latest["Close"]) / float(first["Open"]) - 1.0
    spy_open_return = float(spy_latest["Close"]) / float(spy_first["Open"]) - 1.0
    relative_return = symbol_open_return - spy_open_return
    volume_pace = _opening_volume_pace(symbol_history, rows, session_date)

    checks = {
        "gap": gap_return >= float(min_gap),
        "opening_breakout": float(latest["Close"]) > float(first["High"]),
        "positive_momentum": momentum > 0.0,
        "above_vwap": float(latest["Close"]) > vwap,
        "relative_strength": relative_return >= float(min_relative_return),
        "volume_pace": volume_pace is not None
        and volume_pace >= float(min_volume_pace),
    }
    result.update(
        reason="passed" if all(checks.values()) else "failed opening confirmation",
        checks=checks,
        gap_return=gap_return,
        momentum_5m=momentum,
        opening_return=symbol_open_return,
        spy_open_return=spy_open_return,
        relative_open_return=relative_return,
        vwap=vwap,
        price=float(latest["Close"]),
        opening_high=float(first["High"]),
        volume_pace=volume_pace,
        qualified=all(checks.values()),
    )
    return result


def evaluate_opening_exit_bars(
    symbol_history: pd.DataFrame,
    now_ny: datetime,
) -> Dict[str, object]:
    """Protect an event entry when a completed bar loses VWAP/opening low."""
    rows = _session_rows(symbol_history, now_ny.date())
    completed_cutoff = pd.Timestamp(now_ny).tz_convert("UTC") - pd.Timedelta(minutes=5)
    rows = rows[rows.index <= completed_cutoff]
    if rows.empty or float(rows["Volume"].sum()) <= 0:
        return {"exit": False, "reason": "completed opening bars unavailable"}
    typical = (rows["High"] + rows["Low"] + rows["Close"]) / 3.0
    vwap = float((typical * rows["Volume"]).sum() / rows["Volume"].sum())
    price = float(rows.iloc[-1]["Close"])
    opening_low = float(rows.iloc[0]["Low"])
    lost_vwap = price < vwap
    lost_opening_low = price < opening_low
    return {
        "exit": lost_vwap or lost_opening_low,
        "reason": (
            "lost opening low"
            if lost_opening_low
            else ("lost session VWAP" if lost_vwap else "event trend intact")
        ),
        "price": price,
        "vwap": vwap,
        "opening_low": opening_low,
        "lost_vwap": lost_vwap,
        "lost_opening_low": lost_opening_low,
    }


def _event_is_active(symbol: str, now_ny: datetime, event_dates) -> bool:
    dates = event_dates.get(symbol.upper(), [])
    return now_ny.date().isoformat() in {str(value) for value in dates}


def apply_earnings_opening_continuation(
    decisions: Dict[str, dict],
    diagnostics: Dict[str, dict],
    now_ny: Optional[datetime] = None,
    already_bought_symbols=None,
) -> Dict[str, dict]:
    """Replace HOLD with a paper-only attributed event entry when confirmed."""
    # Runtime imports keep the pure bar evaluator independent from broker
    # credentials and make it directly testable.
    from account_cache import account_cache
    from config import (
        EARNINGS_OPENING_CONTINUATION_ENABLED,
        EARNINGS_OPENING_EVENT_DATES,
        EARNINGS_OPENING_INITIAL_POSITION_EQUITY_PCT,
        EARNINGS_OPENING_MIN_GAP,
        EARNINGS_OPENING_MIN_RELATIVE_RETURN,
        EARNINGS_OPENING_MIN_VOLUME_PACE,
        EARNINGS_OPENING_WINDOW_END,
        LOGS_DIR,
        SPY_SYMBOL,
    )
    from portfolio import PortfolioManager
    from predictive_model.data_loader import (
        fetch_intraday_history,
        fetch_previous_close,
    )

    if not EARNINGS_OPENING_CONTINUATION_ENABLED:
        return decisions

    now_ny = now_ny or datetime.now(NY_TZ)
    if now_ny.tzinfo is None:
        now_ny = NY_TZ.localize(now_ny)
    else:
        now_ny = now_ny.astimezone(NY_TZ)
    end_hour, end_minute = [int(value) for value in EARNINGS_OPENING_WINDOW_END.split(":")]
    minute = now_ny.hour * 60 + now_ny.minute
    if minute < 9 * 60 + 40 or minute > end_hour * 60 + end_minute:
        return decisions

    already_bought = {
        str(symbol).upper() for symbol in (already_bought_symbols or [])
    }

    spy_history = fetch_intraday_history(
        SPY_SYMBOL, lookback_minutes=9750, interval="5min"
    )
    for symbol in sorted(EARNINGS_OPENING_EVENT_DATES):
        if not _event_is_active(symbol, now_ny, EARNINGS_OPENING_EVENT_DATES):
            continue
        if symbol.upper() in already_bought:
            print(f"[EARNINGS OPENING] {symbol}: skipped; already bought this session")
            continue
        try:
            pm = PortfolioManager(symbol)
            pm.refresh_live()
            if float(pm.data.get("shares", 0.0) or 0.0) > 0:
                continue
            history = fetch_intraday_history(
                symbol, lookback_minutes=9750, interval="5min"
            )
            result = evaluate_opening_bars(
                symbol,
                history,
                spy_history,
                fetch_previous_close(symbol),
                now_ny,
                min_gap=EARNINGS_OPENING_MIN_GAP,
                min_volume_pace=EARNINGS_OPENING_MIN_VOLUME_PACE,
                min_relative_return=EARNINGS_OPENING_MIN_RELATIVE_RETURN,
            )
        except Exception as exc:
            print(f"[EARNINGS OPENING] {symbol}: evaluation failed ({exc})")
            continue
        _record_evaluation(LOGS_DIR, now_ny, result)
        print(
            f"[EARNINGS OPENING] {symbol}: qualified={result.get('qualified')} "
            f"reason={result.get('reason')} checks={result.get('checks')}"
        )
        diagnostics.setdefault(symbol, {})["earnings_opening"] = result
        if not result.get("qualified"):
            continue

        account = account_cache.get_account() or {}
        cash = float(account.get("cash", pm.data.get("cash", 0.0)) or 0.0)
        equity = float(account.get("equity", cash) or cash)
        price = float(result["price"])
        requested_value = min(cash, equity * float(EARNINGS_OPENING_INITIAL_POSITION_EQUITY_PCT))
        qty = int(requested_value // price) if price > 0 else 0
        if qty <= 0:
            continue
        decisions[symbol] = {
            "action": "buy",
            "qty": qty,
            "explain": (
                f"{symbol}: earnings opening continuation confirmed; "
                f"gap={result['gap_return']:.2%}, momentum_5m={result['momentum_5m']:.2%}, "
                f"relative_open={result['relative_open_return']:.2%}, "
                f"volume_pace={result['volume_pace']:.2f}"
            ),
            "earnings_opening_continuation": True,
            "initial_exposure_pct": float(
                EARNINGS_OPENING_INITIAL_POSITION_EQUITY_PCT
            ),
            "priority_rank": 1,
        }
    return decisions


def apply_earnings_opening_protective_exit(
    decisions: Dict[str, dict],
    event_symbols,
    now_ny: Optional[datetime] = None,
) -> Dict[str, dict]:
    """Apply fast paper exits to positions opened by the event strategy."""
    from config import EARNINGS_OPENING_CONTINUATION_ENABLED
    from portfolio import PortfolioManager
    from predictive_model.data_loader import fetch_intraday_history

    if not EARNINGS_OPENING_CONTINUATION_ENABLED:
        return decisions
    now_ny = now_ny or datetime.now(NY_TZ)
    if now_ny.tzinfo is None:
        now_ny = NY_TZ.localize(now_ny)
    else:
        now_ny = now_ny.astimezone(NY_TZ)

    for symbol in sorted({str(value).upper() for value in event_symbols or []}):
        try:
            pm = PortfolioManager(symbol)
            pm.refresh_live()
            shares = int(float(pm.data.get("shares", 0.0) or 0.0))
            if shares <= 0:
                continue
            history = fetch_intraday_history(
                symbol, lookback_minutes=9750, interval="5min"
            )
            result = evaluate_opening_exit_bars(history, now_ny)
        except Exception as exc:
            print(f"[EARNINGS OPENING EXIT] {symbol}: evaluation failed ({exc})")
            continue
        print(
            f"[EARNINGS OPENING EXIT] {symbol}: exit={result.get('exit')} "
            f"reason={result.get('reason')} price={result.get('price')} "
            f"vwap={result.get('vwap')} opening_low={result.get('opening_low')}"
        )
        if result.get("exit"):
            decisions[symbol] = {
                "action": "sell",
                "qty": shares,
                "explain": (
                    f"{symbol}: earnings opening protective exit - "
                    f"{result['reason']}"
                ),
                "risk_exit": "earnings_opening_failure",
                "exit_reason": "earnings_opening_failure",
            }
    return decisions
