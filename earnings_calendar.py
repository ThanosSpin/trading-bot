"""Cached, provider-confirmed earnings dates for event-day trading rules."""

from __future__ import annotations

import csv
import io
import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

FINNHUB_URL = "https://finnhub.io/api/v1/calendar/earnings"
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"


class EarningsCalendarProviderError(RuntimeError):
    """Provider failure whose message deliberately excludes request URLs."""


def _ensure_provider_success(response, provider: str) -> None:
    status = int(getattr(response, "status_code", 0) or 0)
    if 200 <= status < 300:
        return
    if status == 401:
        detail = "authentication failed; verify or regenerate the API key"
    elif status == 403:
        detail = "access denied; verify the API plan permits this endpoint"
    elif status == 429:
        detail = "rate limit exceeded; retry after the provider reset"
    else:
        detail = "request failed"
    raise EarningsCalendarProviderError(
        f"{provider} {detail} (HTTP {status or 'unknown'})"
    )


def _iso_date(value) -> Optional[str]:
    try:
        return date.fromisoformat(str(value)[:10]).isoformat()
    except (TypeError, ValueError):
        return None


def _next_nyse_session(report_date: str) -> Optional[str]:
    import pandas_market_calendars as mcal

    start = date.fromisoformat(report_date) + timedelta(days=1)
    end = start + timedelta(days=10)
    schedule = mcal.get_calendar("NYSE").schedule(start_date=start, end_date=end)
    if schedule.empty:
        return None
    return schedule.index[0].date().isoformat()


def resolve_trading_event_date(report_date: str, report_hour: str) -> Optional[str]:
    """Map provider report timing to the regular session that reacts to it."""
    report_date = _iso_date(report_date)
    if report_date is None:
        return None
    hour = str(report_hour or "").strip().lower()
    if hour == "bmo":
        return report_date
    if hour == "amc":
        return _next_nyse_session(report_date)
    return None


def fetch_finnhub_events(
    symbols: Iterable[str],
    start_date: str,
    end_date: str,
    api_key: str,
    session=None,
) -> Dict[str, dict]:
    if session is None:
        import requests

        session = requests
    api_key = str(api_key or "").strip()
    if not api_key:
        raise EarningsCalendarProviderError("Finnhub API key is missing")
    events = {}
    for symbol in sorted({str(value).upper() for value in symbols}):
        response = session.get(
            FINNHUB_URL,
            params={
                "from": start_date,
                "to": end_date,
                "symbol": symbol,
                "token": api_key,
            },
            timeout=20,
        )
        _ensure_provider_success(response, "Finnhub")
        rows = response.json().get("earningsCalendar") or []
        future_rows = [row for row in rows if _iso_date(row.get("date"))]
        if not future_rows:
            continue
        row = sorted(future_rows, key=lambda item: str(item.get("date")))[0]
        report_date = _iso_date(row.get("date"))
        report_hour = str(row.get("hour") or "").lower()
        events[symbol] = {
            "symbol": symbol,
            "report_date": report_date,
            "report_hour": report_hour,
            "trading_event_date": resolve_trading_event_date(
                report_date, report_hour
            ),
            "primary_source": "finnhub",
        }
    return events


def fetch_alpha_vantage_dates(
    symbols: Iterable[str],
    api_key: str,
    session=None,
) -> Dict[str, str]:
    if session is None:
        import requests

        session = requests
    api_key = str(api_key or "").strip()
    if not api_key:
        raise EarningsCalendarProviderError("Alpha Vantage API key is missing")
    response = session.get(
        ALPHA_VANTAGE_URL,
        params={
            "function": "EARNINGS_CALENDAR",
            "horizon": "3month",
            "apikey": api_key,
        },
        timeout=30,
    )
    _ensure_provider_success(response, "Alpha Vantage")
    wanted = {str(value).upper() for value in symbols}
    matches: Dict[str, str] = {}
    for row in csv.DictReader(io.StringIO(response.text)):
        symbol = str(row.get("symbol") or "").upper()
        report_date = _iso_date(row.get("reportDate"))
        if symbol in wanted and report_date and symbol not in matches:
            matches[symbol] = report_date
    return matches


def build_calendar(
    symbols: Iterable[str],
    finnhub_events: Dict[str, dict],
    alpha_dates: Dict[str, str],
    require_confirmation: bool = True,
) -> dict:
    events = {}
    for symbol in sorted({str(value).upper() for value in symbols}):
        event = dict(finnhub_events.get(symbol) or {})
        if not event:
            continue
        alpha_date = alpha_dates.get(symbol)
        confirmed = bool(alpha_date and alpha_date == event.get("report_date"))
        timing_known = event.get("report_hour") in {"bmo", "amc"}
        event.update(
            confirmed_by="alpha_vantage" if confirmed else None,
            confirmation_report_date=alpha_date,
            live_eligible=bool(
                event.get("trading_event_date")
                and timing_known
                and (confirmed or not require_confirmation)
            ),
        )
        events[symbol] = event
    return {
        "schema_version": 1,
        "refreshed_at": datetime.now(timezone.utc).isoformat(),
        "events": events,
    }


def write_calendar_cache(path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_cached_event_dates(
    path,
    max_age_hours: float,
    now: Optional[datetime] = None,
) -> Dict[str, list]:
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text())
        refreshed_at = datetime.fromisoformat(payload["refreshed_at"])
        if refreshed_at.tzinfo is None:
            refreshed_at = refreshed_at.replace(tzinfo=timezone.utc)
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        age_hours = (now - refreshed_at.astimezone(timezone.utc)).total_seconds() / 3600
        if age_hours < 0 or age_hours > float(max_age_hours):
            return {}
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return {}

    result = {}
    for symbol, event in (payload.get("events") or {}).items():
        event_date = _iso_date((event or {}).get("trading_event_date"))
        if event_date and bool((event or {}).get("live_eligible")):
            result[str(symbol).upper()] = [event_date]
    return result


def resolved_event_dates(manual_dates: Dict[str, list]) -> Dict[str, list]:
    """Merge eligible cached dates with explicit manual overrides."""
    from config import (
        EARNINGS_CALENDAR_CACHE_FILE,
        EARNINGS_CALENDAR_MAX_AGE_HOURS,
    )

    combined = {
        str(symbol).upper(): sorted({_iso_date(value) for value in values if _iso_date(value)})
        for symbol, values in (manual_dates or {}).items()
    }
    cached = load_cached_event_dates(
        EARNINGS_CALENDAR_CACHE_FILE,
        EARNINGS_CALENDAR_MAX_AGE_HOURS,
    )
    for symbol, values in cached.items():
        combined[symbol] = sorted(set(combined.get(symbol, [])) | set(values))
    return combined
