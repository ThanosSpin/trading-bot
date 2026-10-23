import unittest

import pandas as pd

from intraday_model_monitor import (
    evaluate_intraday_mode,
    non_overlapping_intraday_outcomes,
)


class IntradayModelMonitorTests(unittest.TestCase):
    def test_overlapping_prediction_horizons_are_removed(self):
        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2026-10-22T14:00:00Z",
                        "2026-10-22T14:06:00Z",
                        "2026-10-22T14:12:00Z",
                        "2026-10-22T14:18:00Z",
                    ],
                    utc=True,
                ),
                "symbol": ["NVDA"] * 4,
                "mode": ["intraday_mr"] * 4,
                "outcome_horizon": ["15min"] * 4,
            }
        )

        selected = non_overlapping_intraday_outcomes(frame)

        self.assertEqual(len(selected), 2)
        self.assertEqual(
            selected["timestamp"].dt.minute.tolist(),
            [0, 18],
        )

    def test_monitor_requires_multiple_sessions_and_can_recommend_shadow(self):
        rows = []
        for session_offset in range(5):
            session = pd.Timestamp("2026-10-12T14:00:00Z") + pd.Timedelta(
                days=session_offset
            )
            for observation in range(6):
                rows.append(
                    {
                        "timestamp": session + pd.Timedelta(minutes=15 * observation),
                        "symbol": "AAPL",
                        "mode": "intraday_mom",
                        "predicted_prob": 0.8,
                        "actual_outcome": 0,
                        "return_pct": -0.2,
                        "session_date_ny": session.date(),
                    }
                )

        metrics = evaluate_intraday_mode(pd.DataFrame(rows), cost_bps=10.0)

        self.assertTrue(metrics["eligible"])
        self.assertTrue(metrics["shadow_retrain_recommended"])
        self.assertEqual(metrics["outcomes"], 30)
        self.assertEqual(metrics["sessions"], 5)
        self.assertTrue(any("accuracy" in reason for reason in metrics["reasons"]))


if __name__ == "__main__":
    unittest.main()
