import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path

import pandas as pd

import outcome_tracker


class OutcomeTrackerCloseTests(unittest.TestCase):
    def test_mixed_legacy_and_iso_prediction_timestamps_are_retained(self):
        values = pd.Series(
            [
                "2026-09-17 11:36:41.830837",
                "2026-10-02T15:42:08.264912",
            ]
        )

        parsed = outcome_tracker._parse_prediction_timestamps(values, utc=True)

        self.assertEqual(parsed.notna().sum(), 2)
        self.assertEqual(str(parsed.dt.tz), "UTC")

    def test_unparseable_timestamp_is_not_erased_when_csv_is_saved(self):
        with TemporaryDirectory() as directory:
            log_file = Path(directory) / "predictions_TEST.csv"
            original_timestamp = "legacy-value-that-cannot-be-parsed"
            pd.DataFrame(
                [
                    {
                        "timestamp": original_timestamp,
                        "symbol": "TEST",
                        "mode": "daily",
                        "predicted_prob": 0.6,
                        "price": 100.0,
                    }
                ]
            ).to_csv(log_file, index=False)

            with patch.object(outcome_tracker, "LOGS_DIR", directory):
                updated = outcome_tracker.update_outcomes_for_symbol("TEST")

            saved = pd.read_csv(log_file)
            self.assertEqual(updated, 0)
            self.assertEqual(saved.loc[0, "timestamp"], original_timestamp)

    def test_daily_naive_index_keeps_exchange_session_date_and_bypasses_cache(self):
        daily = pd.DataFrame(
            {"Close": [100.0, 105.0]},
            index=pd.to_datetime(["2026-10-01", "2026-10-02"]),
        )

        with patch.object(
            outcome_tracker, "fetch_historical_data", return_value=daily
        ) as fetch:
            close = outcome_tracker.get_next_day_close(
                "TEST",
                pd.Timestamp("2026-10-01T15:00:00Z"),
                now_utc=pd.Timestamp("2026-10-02T21:00:00Z"),
            )

        self.assertEqual(close, 105.0)
        fetch.assert_called_once_with(
            "TEST", period="60d", interval="1d", use_cache=False
        )

    def test_daily_does_not_use_current_session_before_close_settles(self):
        daily = pd.DataFrame(
            {"Close": [100.0, 103.0]},
            index=pd.to_datetime(["2026-10-01", "2026-10-02"]),
        )

        with patch.object(
            outcome_tracker, "fetch_historical_data", return_value=daily
        ):
            close = outcome_tracker.get_next_day_close(
                "TEST",
                pd.Timestamp("2026-10-01T15:00:00Z"),
                now_utc=pd.Timestamp("2026-10-02T19:59:00Z"),
            )

        self.assertIsNone(close)

    def test_60_minute_outcome_uses_first_completed_15_minute_bar(self):
        bars = pd.DataFrame(
            {"Close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]},
            index=pd.date_range("2026-10-02T14:00:00Z", periods=6, freq="15min"),
        )

        with patch.object(
            outcome_tracker, "fetch_intraday_history", return_value=bars
        ) as fetch:
            close = outcome_tracker.get_intraday_horizon_close(
                "TEST",
                pd.Timestamp("2026-10-02T14:06:00Z"),
                horizon_minutes=60,
                now_utc=pd.Timestamp("2026-10-02T15:16:00Z"),
            )

        # The 15:00 UTC bar closes at 15:15, the first completed close at
        # least 60 minutes after the 14:06 prediction.
        self.assertEqual(close, 104.0)
        self.assertEqual(fetch.call_args.kwargs["interval"], "15min")

    def test_intraday_outcome_does_not_cross_trading_sessions(self):
        bars = pd.DataFrame(
            {"Close": [110.0, 111.0]},
            index=pd.to_datetime(
                ["2026-10-05T13:30:00Z", "2026-10-05T13:45:00Z"]
            ),
        )

        with patch.object(
            outcome_tracker, "fetch_intraday_history", return_value=bars
        ):
            close = outcome_tracker.get_intraday_horizon_close(
                "TEST",
                pd.Timestamp("2026-10-02T19:45:00Z"),
                horizon_minutes=60,
                now_utc=pd.Timestamp("2026-10-05T15:00:00Z"),
            )

        self.assertIsNone(close)

    def test_intraday_outcome_ignores_an_incomplete_bar(self):
        bars = pd.DataFrame(
            {"Close": [101.0]},
            index=pd.to_datetime(["2026-10-02T14:15:00Z"]),
        )

        with patch.object(
            outcome_tracker, "fetch_intraday_history", return_value=bars
        ):
            close = outcome_tracker.get_intraday_horizon_close(
                "TEST",
                pd.Timestamp("2026-10-02T14:06:00Z"),
                horizon_minutes=15,
                now_utc=pd.Timestamp("2026-10-02T14:29:00Z"),
            )

        self.assertIsNone(close)


if __name__ == "__main__":
    unittest.main()
