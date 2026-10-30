import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import pytz

os.environ.setdefault("BOT_ENV", "paper")

from earnings_opening import (
    _record_evaluation,
    apply_earnings_event_day_entry_gate,
    evaluate_opening_bars,
    evaluate_opening_exit_bars,
)


NY_TZ = pytz.timezone("America/New_York")


def _history(symbol=True):
    rows = []
    for date_value in ["2026-10-27", "2026-10-28", "2026-10-29"]:
        for minute in [30, 35]:
            timestamp = pd.Timestamp(
                f"{date_value} 09:{minute:02d}:00", tz="America/New_York"
            ).tz_convert("UTC")
            rows.append(
                {
                    "timestamp": timestamp,
                    "Open": 100.0,
                    "High": 100.2,
                    "Low": 99.9,
                    "Close": 100.1,
                    "Volume": 100.0,
                }
            )

    today = "2026-10-30"
    if symbol:
        values = [
            (30, 103.0, 104.0, 102.8, 103.8, 1000.0),
            (35, 103.8, 105.2, 103.7, 105.0, 1200.0),
        ]
    else:
        values = [
            (30, 100.0, 100.2, 99.9, 100.1, 1000.0),
            (35, 100.1, 100.3, 100.0, 100.2, 1000.0),
        ]
    for minute, open_, high, low, close, volume in values:
        rows.append(
            {
                "timestamp": pd.Timestamp(
                    f"{today} 09:{minute:02d}:00", tz="America/New_York"
                ).tz_convert("UTC"),
                "Open": open_,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
            }
        )
    return pd.DataFrame(rows).set_index("timestamp")


class EarningsOpeningTests(unittest.TestCase):
    def test_event_day_gate_blocks_ordinary_buy_only_for_event_symbol(self):
        decisions = {
            "ABBV": {
                "action": "buy",
                "qty": 13,
                "explain": "ABBV: Rotation BUY after AAPL sale.",
            },
            "AAPL": {
                "action": "sell",
                "qty": 11,
                "explain": "AAPL: Rotated into stronger ABBV signal.",
                "rotation_source": "AAPL",
                "rotation_target": "ABBV",
            },
        }
        now_ny = NY_TZ.localize(
            pd.Timestamp("2026-10-30 13:20:00").to_pydatetime()
        )

        result = apply_earnings_event_day_entry_gate(
            decisions,
            now_ny=now_ny,
            enabled=True,
            event_dates={"ABBV": ["2026-10-30"]},
        )

        self.assertEqual(result["ABBV"]["action"], "hold")
        self.assertEqual(result["ABBV"]["qty"], 0)
        self.assertTrue(result["ABBV"]["event_day_entry_blocked"])
        self.assertEqual(result["AAPL"]["action"], "hold")
        self.assertEqual(result["AAPL"]["qty"], 0)
        self.assertTrue(result["AAPL"]["event_day_rotation_cancelled"])

    def test_event_day_gate_preserves_independent_sell(self):
        decisions = {
            "ABBV": {
                "action": "buy",
                "qty": 13,
                "explain": "ABBV rotation BUY",
                "rotation_source": "AAPL",
                "rotation_target": "ABBV",
            },
            "AAPL": {
                "action": "sell",
                "qty": 11,
                "explain": "AAPL risk stop",
                "risk_exit": "dynamic_risk_stop",
            },
        }
        now_ny = NY_TZ.localize(
            pd.Timestamp("2026-10-30 13:20:00").to_pydatetime()
        )

        result = apply_earnings_event_day_entry_gate(
            decisions,
            now_ny=now_ny,
            enabled=True,
            event_dates={"ABBV": ["2026-10-30"]},
        )

        self.assertEqual(result["ABBV"]["action"], "hold")
        self.assertEqual(result["AAPL"]["action"], "sell")

    def test_event_day_gate_does_not_block_non_event_date(self):
        decisions = {
            "ABBV": {"action": "buy", "qty": 2, "explain": "ordinary BUY"}
        }
        now_ny = NY_TZ.localize(
            pd.Timestamp("2026-10-29 10:15:00").to_pydatetime()
        )

        result = apply_earnings_event_day_entry_gate(
            decisions,
            now_ny=now_ny,
            enabled=True,
            event_dates={"ABBV": ["2026-10-30"]},
        )

        self.assertEqual(result["ABBV"]["action"], "buy")

    def test_qualified_gap_breakout_uses_completed_bars(self):
        result = evaluate_opening_bars(
            "ABBV",
            _history(symbol=True),
            _history(symbol=False),
            previous_close=100.0,
            now_ny=NY_TZ.localize(pd.Timestamp("2026-10-30 09:40:00").to_pydatetime()),
        )

        self.assertTrue(result["qualified"])
        self.assertTrue(all(result["checks"].values()))
        self.assertGreater(result["volume_pace"], 1.5)

    def test_waits_until_second_bar_is_complete(self):
        result = evaluate_opening_bars(
            "ABBV",
            _history(symbol=True),
            _history(symbol=False),
            previous_close=100.0,
            now_ny=NY_TZ.localize(pd.Timestamp("2026-10-30 09:39:00").to_pydatetime()),
        )

        self.assertFalse(result["qualified"])
        self.assertIn("two completed", result["reason"])

    def test_records_separate_paper_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            timestamp = NY_TZ.localize(
                pd.Timestamp("2026-10-30 09:40:00").to_pydatetime()
            )
            _record_evaluation(
                directory,
                timestamp,
                {
                    "symbol": "ABBV",
                    "qualified": True,
                    "reason": "passed",
                    "checks": {"gap": True},
                },
            )

            frame = pd.read_csv(Path(directory) / "earnings_opening_signals.csv")
            self.assertEqual(len(frame), 1)
            self.assertEqual(frame.iloc[0]["symbol"], "ABBV")
            self.assertTrue(frame.iloc[0]["qualified"])

    def test_event_exit_fires_when_completed_bar_loses_vwap(self):
        history = _history(symbol=True)
        reversal = pd.DataFrame(
            [
                {
                    "timestamp": pd.Timestamp(
                        "2026-10-30 09:40:00", tz="America/New_York"
                    ).tz_convert("UTC"),
                    "Open": 105.0,
                    "High": 105.1,
                    "Low": 102.0,
                    "Close": 102.2,
                    "Volume": 2000.0,
                }
            ]
        ).set_index("timestamp")
        history = pd.concat([history, reversal]).sort_index()

        result = evaluate_opening_exit_bars(
            history,
            NY_TZ.localize(
                pd.Timestamp("2026-10-30 09:45:00").to_pydatetime()
            ),
        )

        self.assertTrue(result["exit"])
        self.assertTrue(result["lost_vwap"])


if __name__ == "__main__":
    unittest.main()
