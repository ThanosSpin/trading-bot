import unittest

import pandas as pd

from evaluate_guarded_trend_entries import (
    evaluate_horizon,
    guarded_candidates,
    summarize,
)


def _history(momentum=0.002):
    timestamps = pd.date_range("2026-10-28 14:00:00+00:00", periods=14, freq="5min")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "price": [100.0 + index * 0.1 for index in range(len(timestamps))],
            "finalprob": 0.55,
            "buythreshold": 0.57,
            "sessionreturn": 0.01,
            "intradaymom": momentum,
            "vwapdistance": 0.002,
            "vwapslope2": 0.001,
            "relativereturn4": 0.003,
            "relativesessionreturn": 0.004,
            "relativemarketavailable": True,
            "volumetimeratio": 1.0,
        }
    )


class GuardedTrendEntryEvaluationTests(unittest.TestCase):
    def test_rule_requires_all_confirmations(self):
        passing = guarded_candidates(_history(), 0.05, 0.80)
        failing = guarded_candidates(_history(momentum=-0.001), 0.05, 0.80)

        self.assertEqual(len(passing), 14)
        self.assertTrue(failing.empty)

    def test_horizon_samples_are_non_overlapping(self):
        history = _history()
        candidates = guarded_candidates(history, 0.05, 0.80)
        evaluated = evaluate_horizon(history, candidates, 30, 10, 5)

        self.assertEqual(len(evaluated), 2)
        self.assertTrue((evaluated["net_return"] > 0).all())
        self.assertEqual(evaluated.iloc[0]["future_timestamp"], history.iloc[6]["timestamp"])

    def test_summary_collects_until_sample_and_session_minimums(self):
        history = _history()
        candidates = guarded_candidates(history, 0.05, 0.80)
        evaluated = evaluate_horizon(history, candidates, 30, 10, 5)

        summary = summarize(evaluated, minimum_samples=20, minimum_sessions=5)

        self.assertEqual(summary["status"], "COLLECTING")
        self.assertEqual(summary["observations"], 2)


if __name__ == "__main__":
    unittest.main()
