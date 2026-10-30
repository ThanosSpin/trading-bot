import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from earnings_calendar import (
    build_calendar,
    load_cached_event_dates,
    resolve_trading_event_date,
    write_calendar_cache,
)


class EarningsCalendarTests(unittest.TestCase):
    def test_before_open_uses_report_date(self):
        self.assertEqual(
            resolve_trading_event_date("2026-11-05", "bmo"),
            "2026-11-05",
        )

    def test_after_close_uses_next_nyse_session(self):
        with patch(
            "earnings_calendar._next_nyse_session",
            return_value="2026-11-02",
        ):
            self.assertEqual(
                resolve_trading_event_date("2026-10-30", "amc"),
                "2026-11-02",
            )

    def test_unknown_timing_is_not_live_eligible(self):
        payload = build_calendar(
            ["PLTR"],
            {
                "PLTR": {
                    "symbol": "PLTR",
                    "report_date": "2026-11-02",
                    "report_hour": "dmh",
                    "trading_event_date": None,
                    "primary_source": "finnhub",
                }
            },
            {"PLTR": "2026-11-02"},
        )
        self.assertFalse(payload["events"]["PLTR"]["live_eligible"])

    def test_provider_confirmation_is_required(self):
        event = {
            "symbol": "AAPL",
            "report_date": "2026-11-05",
            "report_hour": "amc",
            "trading_event_date": "2026-11-06",
            "primary_source": "finnhub",
        }
        payload = build_calendar(
            ["AAPL"], {"AAPL": event}, {"AAPL": "2026-11-06"}
        )
        self.assertFalse(payload["events"]["AAPL"]["live_eligible"])

    def test_fresh_eligible_cache_is_loaded_and_stale_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "earnings.json"
            payload = {
                "schema_version": 1,
                "refreshed_at": "2026-10-30T08:00:00+00:00",
                "events": {
                    "ABBV": {
                        "trading_event_date": "2026-10-30",
                        "live_eligible": True,
                    },
                    "PLTR": {
                        "trading_event_date": "2026-11-02",
                        "live_eligible": False,
                    },
                },
            }
            write_calendar_cache(path, payload)
            fresh = load_cached_event_dates(
                path,
                36,
                now=datetime(2026, 10, 30, 12, tzinfo=timezone.utc),
            )
            stale = load_cached_event_dates(
                path,
                1,
                now=datetime(2026, 10, 30, 12, tzinfo=timezone.utc),
            )

            self.assertEqual(fresh, {"ABBV": ["2026-10-30"]})
            self.assertEqual(stale, {})


if __name__ == "__main__":
    unittest.main()
