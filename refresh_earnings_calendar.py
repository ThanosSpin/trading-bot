#!/usr/bin/env python3
"""Refresh the durable earnings calendar used by event-day entry routing."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from config import (
    ALPHA_VANTAGE_API_KEY,
    EARNINGS_CALENDAR_CACHE_FILE,
    EARNINGS_CALENDAR_REQUIRE_CONFIRMATION,
    EARNINGS_CALENDAR_SYMBOLS,
    FINNHUB_API_KEY,
)
from earnings_calendar import (
    build_calendar,
    fetch_alpha_vantage_dates,
    fetch_finnhub_events,
    write_calendar_cache,
)


def main() -> int:
    if not FINNHUB_API_KEY:
        raise RuntimeError("FINNHUB_API_KEY is required")
    if EARNINGS_CALENDAR_REQUIRE_CONFIRMATION and not ALPHA_VANTAGE_API_KEY:
        raise RuntimeError(
            "ALPHA_VANTAGE_API_KEY is required while calendar confirmation is enabled"
        )

    start = datetime.now(ZoneInfo("America/New_York")).date()
    end = start + timedelta(days=100)
    primary = fetch_finnhub_events(
        EARNINGS_CALENDAR_SYMBOLS,
        start.isoformat(),
        end.isoformat(),
        FINNHUB_API_KEY,
    )
    secondary = (
        fetch_alpha_vantage_dates(EARNINGS_CALENDAR_SYMBOLS, ALPHA_VANTAGE_API_KEY)
        if ALPHA_VANTAGE_API_KEY
        else {}
    )
    payload = build_calendar(
        EARNINGS_CALENDAR_SYMBOLS,
        primary,
        secondary,
        require_confirmation=EARNINGS_CALENDAR_REQUIRE_CONFIRMATION,
    )
    write_calendar_cache(EARNINGS_CALENDAR_CACHE_FILE, payload)

    print(f"[EARNINGS CALENDAR] Cache: {EARNINGS_CALENDAR_CACHE_FILE}")
    for symbol in EARNINGS_CALENDAR_SYMBOLS:
        event = payload["events"].get(symbol)
        if not event:
            print(f"[EARNINGS CALENDAR] {symbol}: no upcoming event")
            continue
        print(
            f"[EARNINGS CALENDAR] {symbol}: report={event['report_date']} "
            f"hour={event['report_hour'] or 'unknown'} "
            f"event_session={event['trading_event_date']} "
            f"confirmed={bool(event['confirmed_by'])} "
            f"live_eligible={event['live_eligible']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
