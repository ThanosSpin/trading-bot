import unittest

import numpy as np
import pandas as pd

from predictive_model.features import (
    add_intraday_relative_market_features,
    add_intraday_vwap_volume_features,
)


class IntradayVwapVolumeFeatureTests(unittest.TestCase):
    @staticmethod
    def _frame():
        first = pd.date_range(
            "2026-10-01 09:30", periods=6, freq="15min", tz="America/New_York"
        )
        second = pd.date_range(
            "2026-10-02 09:30", periods=6, freq="15min", tz="America/New_York"
        )
        index = first.append(second)
        close = np.array(
            [100, 101, 102, 101, 103, 104, 200, 202, 204, 202, 206, 208],
            dtype=float,
        )
        volume = np.array(
            [100, 200, 300, 400, 500, 600, 200, 400, 600, 800, 1000, 1200],
            dtype=float,
        )
        return pd.DataFrame(
            {
                "Open": close,
                "High": close + 1.0,
                "Low": close - 1.0,
                "Close": close,
                "Volume": volume,
            },
            index=index,
        )

    def test_vwap_resets_at_new_session(self):
        frame = self._frame()
        result = add_intraday_vwap_volume_features(frame)
        second_open = frame.index[6]

        self.assertAlmostEqual(result.loc[second_open, "vwap_distance"], 0.0)
        self.assertAlmostEqual(result.loc[second_open, "vwap_slope_2"], 0.0)
        self.assertAlmostEqual(result.loc[second_open, "vwap_slope_4"], 0.0)
        self.assertEqual(result.loc[second_open, "vwap_cross_up"], 0.0)
        self.assertEqual(result.loc[second_open, "vwap_cross_down"], 0.0)

    def test_time_slot_volume_baseline_uses_prior_session(self):
        frame = self._frame()
        result = add_intraday_vwap_volume_features(frame)

        for row in frame.index[6:]:
            self.assertAlmostEqual(result.loc[row, "volume_time_ratio"], 2.0)
            self.assertAlmostEqual(result.loc[row, "volume_time_log_ratio"], np.log(2.0))
            self.assertAlmostEqual(result.loc[row, "session_volume_pace"], 2.0)

    def test_current_volume_does_not_change_its_own_baseline(self):
        frame = self._frame()
        changed = frame.copy()
        changed.loc[changed.index[-1], "Volume"] = 12000.0

        baseline = add_intraday_vwap_volume_features(frame)
        result = add_intraday_vwap_volume_features(changed)

        self.assertAlmostEqual(baseline.iloc[-1]["volume_time_ratio"], 2.0)
        self.assertAlmostEqual(result.iloc[-1]["volume_time_ratio"], 20.0)

    def test_timezone_is_required(self):
        frame = self._frame()
        frame.index = frame.index.tz_localize(None)
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            add_intraday_vwap_volume_features(frame)

    def test_relative_returns_use_matching_benchmark_bars(self):
        frame = add_intraday_vwap_volume_features(self._frame())
        benchmark = self._frame().copy()
        benchmark["Close"] = np.array(
            [100, 100.5, 101, 101.5, 102, 102.5, 200, 201, 202, 203, 204, 205],
            dtype=float,
        )
        benchmark["Open"] = benchmark["Close"]
        benchmark["High"] = benchmark["Close"] + 1.0
        benchmark["Low"] = benchmark["Close"] - 1.0

        result = add_intraday_relative_market_features(frame, benchmark)
        row = frame.index[-1]
        expected = frame["Close"].pct_change(4).loc[row] - benchmark["Close"].pct_change(4).loc[row]
        self.assertAlmostEqual(result.loc[row, "relative_return_4"], expected)
        self.assertEqual(result.loc[row, "relative_market_available"], 1.0)

    def test_relative_alignment_never_uses_future_benchmark_bar(self):
        frame = add_intraday_vwap_volume_features(self._frame().iloc[:6])
        benchmark = self._frame().iloc[:6].copy()
        benchmark.index = benchmark.index + pd.Timedelta(minutes=1)

        result = add_intraday_relative_market_features(frame, benchmark)

        self.assertEqual(result.iloc[0]["relative_market_available"], 0.0)
        self.assertEqual(result.iloc[1]["relative_market_available"], 1.0)
        # At 09:45, alignment may use 09:31 but never the future 09:46 bar.
        self.assertAlmostEqual(result.iloc[1]["spy_return_1"], 0.0)
        self.assertAlmostEqual(
            result.iloc[2]["spy_return_1"],
            benchmark["Close"].iloc[1] / benchmark["Close"].iloc[0] - 1.0,
        )

    def test_missing_benchmark_is_explicit_and_neutral(self):
        frame = add_intraday_vwap_volume_features(self._frame())
        result = add_intraday_relative_market_features(frame, None)
        self.assertTrue((result["relative_market_available"] == 0.0).all())
        self.assertTrue((result["relative_return_4"] == 0.0).all())


if __name__ == "__main__":
    unittest.main()
