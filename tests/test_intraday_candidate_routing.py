import unittest

from predictive_model.model_xgb import _order_intraday_model_candidates


class IntradayCandidateRoutingTests(unittest.TestCase):
    def test_stale_regime_match_precedes_fresh_opposite_regime(self):
        candidates = _order_intraday_model_candidates(
            ["intraday_mr", "intraday_mom", "intraday"],
            [("intraday_mom", object(), 0.9)],
            [("intraday_mr", object(), 34.0)],
        )

        self.assertEqual(
            [(name, age, stale) for name, _, age, stale in candidates],
            [("intraday_mr", 34.0, True), ("intraday_mom", 0.9, False)],
        )

    def test_fresh_regime_match_remains_first(self):
        candidates = _order_intraday_model_candidates(
            ["intraday_mom", "intraday_mr", "intraday"],
            [("intraday_mr", object(), 2.0), ("intraday_mom", object(), 1.0)],
            [],
        )

        self.assertEqual(
            [name for name, _, _, _ in candidates],
            ["intraday_mom", "intraday_mr"],
        )


if __name__ == "__main__":
    unittest.main()
