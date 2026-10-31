import unittest

import pandas as pd

from evaluate_guarded_trend_entries import (
    evaluate_horizon,
    guard_funnel,
    guarded_candidates,
    horizon_eligible_count,
    summarize,
)


def _history(momentum=0.002, relative_session=0.004, start="2026-10-28 14:00:00+00:00"):
    timestamps = pd.date_range(start, periods=14, freq="5min")
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
            "relativesessionreturn": relative_session,
            "relativemarketavailable": True,
            "volumetimeratio": 1.0,
        }
    )


class GuardedTrendEntryEvaluationTests(unittest.TestCase):
    def test_rule_requires_all_confirmations(self):
        passing = guarded_candidates(_history(), 0.05, 0.80, relative_rule="strict")
        failing = guarded_candidates(
            _history(momentum=-0.001), 0.05, 0.80, relative_rule="strict"
        )

        self.assertEqual(len(passing), 14)
        self.assertTrue(failing.empty)

    def test_recent_rule_does_not_require_positive_session_relative_return(self):
        history = _history(relative_session=-0.005)

        strict = guarded_candidates(history, 0.05, 0.80, relative_rule="strict")
        recent = guarded_candidates(history, 0.05, 0.80, relative_rule="recent")

        self.assertTrue(strict.empty)
        self.assertEqual(len(recent), 14)

    def test_funnel_identifies_first_guard_that_rejects_rows(self):
        history = _history(momentum=-0.001)

        funnel = guard_funnel(history, 0.05, 0.80, relative_rule="strict")

        self.assertEqual(funnel["complete_rows"], 14)
        self.assertEqual(funnel["individual_pass_counts"]["probability_gap"], 14)
        self.assertEqual(funnel["cumulative_pass_counts"]["positive_session_return"], 14)
        self.assertEqual(funnel["rejected_at_guard"]["positive_intraday_momentum"], 14)
        self.assertEqual(funnel["cumulative_pass_counts"]["positive_intraday_momentum"], 0)
        self.assertEqual(funnel["candidate_rows"], 0)

    def test_recent_funnel_omits_session_relative_guard(self):
        funnel = guard_funnel(
            _history(relative_session=-0.005),
            0.05,
            0.80,
            relative_rule="recent",
        )

        self.assertNotIn(
            "positive_session_relative_return", funnel["cumulative_pass_counts"]
        )
        self.assertEqual(funnel["candidate_rows"], 14)

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

    def test_horizon_excludes_entries_too_close_to_market_close(self):
        history = _history(start="2026-10-28 19:40:00+00:00")
        candidates = guarded_candidates(history, 0.05, 0.80)

        evaluated = evaluate_horizon(history, candidates, 30, 10, 5)

        self.assertTrue(evaluated.empty)
        self.assertEqual(horizon_eligible_count(candidates, 30), 0)


if __name__ == "__main__":
    unittest.main()
