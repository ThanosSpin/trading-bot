#!/usr/bin/env python3
"""Build an incremental, persistent Alpaca 15-minute research dataset."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

import config
from broker import get_trading_api


DEFAULT_SYMBOLS = tuple(
    dict.fromkeys(
        [*config.TRAIN_SYMBOLS, config.SPY_SYMBOL, *config.SECTOR_BENCHMARKS.values()]
    )
)


def _bars_frame(bars) -> pd.DataFrame:
    rows = [
        {
            "timestamp": bar.t,
            "Open": float(bar.o),
            "High": float(bar.h),
            "Low": float(bar.l),
            "Close": float(bar.c),
            "Volume": float(bar.v),
        }
        for bar in bars
    ]
    if not rows:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.set_index("timestamp").sort_index()


def fetch_alpaca_range(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch a range in bounded chunks to avoid API page/limit truncation."""
    api = get_trading_api()
    chunks = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=30), end)
        bars = api.get_bars(
            symbol,
            timeframe="15Min",
            start=cursor.isoformat(),
            end=chunk_end.isoformat(),
            feed="iex",
            adjustment="raw",
            limit=10000,
        )
        frame = _bars_frame(bars)
        if not frame.empty:
            chunks.append(frame)
        cursor = chunk_end
    if not chunks:
        return _bars_frame([])
    frame = pd.concat(chunks).sort_index()
    return frame[~frame.index.duplicated(keep="last")]


def regular_session_only(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    ny = frame.index.tz_convert("America/New_York")
    minute = ny.hour * 60 + ny.minute
    weekdays = ny.dayofweek < 5
    mask = weekdays & (minute >= 9 * 60 + 30) & (minute < 16 * 60)
    return frame.loc[mask]


def update_symbol(
    symbol: str,
    output_dir: Path,
    months: int,
    end: datetime,
) -> dict:
    path = output_dir / f"{symbol}_15min.parquet"
    existing = pd.read_parquet(path) if path.exists() else None
    requested_start = end - timedelta(days=max(31, int(months * 30.4375)))
    if existing is not None and not existing.empty:
        existing.index = pd.to_datetime(existing.index, utc=True)
        start = max(requested_start, existing.index.max().to_pydatetime() + timedelta(seconds=1))
    else:
        start = requested_start

    fetched = fetch_alpaca_range(symbol, start, end) if start < end else _bars_frame([])
    combined = pd.concat([item for item in (existing, fetched) if item is not None])
    combined.index = pd.to_datetime(combined.index, utc=True)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined = regular_session_only(combined)
    combined = combined.loc[combined.index >= pd.Timestamp(requested_start)]
    output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(path)
    return {
        "symbol": symbol,
        "path": str(path),
        "rows": len(combined),
        "start": str(combined.index.min()) if not combined.empty else None,
        "end": str(combined.index.max()) if not combined.empty else None,
        "new_rows": len(fetched),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--output-dir", default="research_data/intraday_15m")
    args = parser.parse_args()
    if args.months < 12:
        parser.error("--months must be at least 12 for unified intraday research")

    end = datetime.now(timezone.utc)
    output_dir = Path(args.output_dir).resolve()
    for raw_symbol in args.symbols:
        symbol = str(raw_symbol).strip().upper()
        result = update_symbol(symbol, output_dir, args.months, end)
        print(
            f"[DATASET] {symbol}: rows={result['rows']} new={result['new_rows']} "
            f"range={result['start']} -> {result['end']} path={result['path']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
