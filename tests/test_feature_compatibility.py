import unittest

import numpy as np
import pandas as pd

from predictive_model.features import add_advanced_features
from predictive_model.time_features import add_intraday_market_features


class FeatureCompatibilityTests(unittest.TestCase):
    @staticmethod
    def _frame(periods=120):
        index = pd.date_range(
            "2026-10-01 09:30",
            periods=periods,
            freq="15min",
            tz="America/New_York",
        )
        close = np.linspace(100.0, 110.0, periods)
        return pd.DataFrame(
            {
                "Open": close - 0.1,
                "High": close + 0.5,
                "Low": close - 0.5,
                "Close": close,
                "Volume": np.arange(periods, dtype=float) + 1000.0,
            },
            index=index,
        )

    def test_legacy_rsi_columns_are_preserved(self):
        result = add_advanced_features(self._frame(), mode="daily")

        self.assertIn("rsi_6", result.columns)
        self.assertIn("rsi_14", result.columns)

    def test_market_features_accept_datetime_index_arithmetic(self):
        frame = self._frame(periods=20)
        result = add_intraday_market_features(frame)

        self.assertIn("volume_time_ratio", result.columns)
        self.assertTrue(result["volume_time_ratio"].notna().any())

    def test_benchmark_features_keep_timezone_aware_asset_index(self):
        frame = self._frame(periods=20)
        benchmark = frame.copy()
        benchmark.index = benchmark.index.tz_convert("UTC")
        benchmark["Close"] = benchmark["Close"] * 0.99

        result = add_intraday_market_features(frame, benchmark_df=benchmark)

        self.assertTrue(result["relative_return_1"].iloc[1:].notna().all())
        self.assertTrue((result["relative_market_available"] == 1).all())


if __name__ == "__main__":
    unittest.main()
