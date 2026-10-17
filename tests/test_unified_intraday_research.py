import unittest

import numpy as np
import pandas as pd

from build_intraday_dataset import regular_session_only
from unified_intraday_research import (
    add_causal_regime_feature,
    aggregate_non_overlapping_trades,
    model_factory,
)


class UnifiedIntradayResearchTests(unittest.TestCase):
    @staticmethod
    def _frame(periods=700):
        index = pd.date_range(
            "2026-01-05 09:30",
            periods=periods,
            freq="15min",
            tz="America/New_York",
        )
        close = 100 + np.sin(np.arange(periods) / 10) + np.arange(periods) * 0.01
        return pd.DataFrame(
            {
                "Open": close,
                "High": close + 0.2,
                "Low": close - 0.2,
                "Close": close,
                "Volume": 1000,
            },
            index=index,
        )

    def test_regime_feature_does_not_rewrite_past_when_future_changes(self):
        frame = self._frame()
        changed = frame.copy()
        changed.iloc[-1, changed.columns.get_loc("Close")] *= 1.5
        original_result = add_causal_regime_feature(frame)
        changed_result = add_causal_regime_feature(changed)
        pd.testing.assert_series_equal(
            original_result["regime_momentum"].iloc[:-1],
            changed_result["regime_momentum"].iloc[:-1],
        )

    def test_regular_session_filter(self):
        index = pd.DatetimeIndex(
            [
                "2026-01-05 13:00:00+00:00",
                "2026-01-05 14:30:00+00:00",
                "2026-01-05 20:45:00+00:00",
                "2026-01-05 21:00:00+00:00",
            ]
        )
        frame = pd.DataFrame({"Close": range(4)}, index=index)
        result = regular_session_only(frame)
        self.assertEqual(len(result), 2)
        self.assertEqual(result.index[0], index[1])
        self.assertEqual(result.index[-1], index[2])

    def test_aggregate_charges_cost_per_non_overlapping_trade(self):
        metrics = aggregate_non_overlapping_trades(
            np.array([0.02, -0.01, 0.50]),
            np.array([True, True, False]),
            cost_bps=10.0,
        )
        expected = (1.019 * 0.989) - 1.0
        self.assertEqual(metrics["trade_count"], 2)
        self.assertAlmostEqual(metrics["net_return"], expected)

    def test_model_candidates_are_distinct(self):
        self.assertNotEqual(
            type(model_factory("logistic")).__name__,
            type(model_factory("xgboost")).__name__,
        )

    def test_no_trade_evaluation_is_not_profitable(self):
        metrics = aggregate_non_overlapping_trades(
            np.array([0.10, 0.20]),
            np.array([False, False]),
            cost_bps=10.0,
        )
        self.assertEqual(metrics["trade_count"], 0)
        self.assertEqual(metrics["net_return"], 0.0)
        self.assertEqual(metrics["profit_factor"], 0.0)


if __name__ == "__main__":
    unittest.main()
