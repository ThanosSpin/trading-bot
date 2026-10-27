import unittest

import numpy as np
import pandas as pd

from predictive_model.features import add_advanced_features
from predictive_model.model_xgb import _validate_exact_feature_schema
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

    def test_market_clock_features_match_for_new_york_and_utc_indexes(self):
        new_york_frame = self._frame(periods=20)
        utc_frame = new_york_frame.copy()
        utc_frame.index = utc_frame.index.tz_convert("UTC")

        new_york_result = add_intraday_market_features(new_york_frame)
        utc_result = add_intraday_market_features(utc_frame)

        for column in (
            "session_volume_pace",
            "volume_time_ratio",
            "vwap_distance",
            "vwap_slope_2",
        ):
            np.testing.assert_allclose(
                new_york_result[column].to_numpy(),
                utc_result[column].to_numpy(),
                equal_nan=True,
            )

    def test_compatibility_clock_matches_for_new_york_and_utc_indexes(self):
        new_york_frame = self._frame(periods=4)
        utc_frame = new_york_frame.copy()
        utc_frame.index = utc_frame.index.tz_convert("UTC")

        from predictive_model.time_features import (
            add_intraday_compatibility_features,
        )

        new_york_result = add_intraday_compatibility_features(new_york_frame)
        utc_result = add_intraday_compatibility_features(utc_frame)

        for column in (
            "clock_minute",
            "is_regular_session",
            "is_first_15min",
            "is_last_15min",
        ):
            np.testing.assert_array_equal(
                new_york_result[column].to_numpy(),
                utc_result[column].to_numpy(),
            )

    def test_benchmark_features_keep_timezone_aware_asset_index(self):
        frame = self._frame(periods=20)
        benchmark = frame.copy()
        benchmark.index = benchmark.index.tz_convert("UTC")
        benchmark["Close"] = benchmark["Close"] * 0.99

        result = add_intraday_market_features(frame, benchmark_df=benchmark)

        self.assertTrue(result["relative_return_1"].iloc[1:].notna().all())
        self.assertTrue((result["relative_market_available"] == 1).all())

    def test_known_legacy_indicator_capitalization_is_aliased(self):
        current = pd.DataFrame(
            {
                "rsi": [55.0],
                "macd_signal": [0.25],
                "ema_14": [101.0],
            }
        )

        aligned = _validate_exact_feature_schema(
            current,
            ["Rsi", "Macd_signal", "Ema_14"],
        )

        self.assertEqual(list(aligned.columns), ["Rsi", "Macd_signal", "Ema_14"])
        self.assertEqual(aligned.iloc[0].tolist(), [55.0, 0.25, 101.0])

    def test_unrelated_missing_feature_remains_an_error(self):
        with self.assertRaisesRegex(ValueError, "Feature schema mismatch"):
            _validate_exact_feature_schema(
                pd.DataFrame({"signal": [1.0]}),
                ["Unknown_legacy_feature"],
            )

    def test_legacy_clock_features_are_derived_only_when_required(self):
        index = pd.DatetimeIndex(["2026-10-22 09:30:00"], tz="America/New_York")
        current = pd.DataFrame({"signal": [1.0]}, index=index)

        aligned = _validate_exact_feature_schema(
            current,
            [
                "signal",
                "hour_sin",
                "hour_cos",
                "minute_sin",
                "minute_cos",
            ],
        )

        self.assertAlmostEqual(aligned.iloc[0]["hour_sin"], np.sin(2 * np.pi * 9 / 24))
        self.assertAlmostEqual(aligned.iloc[0]["hour_cos"], np.cos(2 * np.pi * 9 / 24))
        self.assertAlmostEqual(aligned.iloc[0]["minute_sin"], 0.0)
        self.assertAlmostEqual(aligned.iloc[0]["minute_cos"], -1.0)


if __name__ == "__main__":
    unittest.main()
