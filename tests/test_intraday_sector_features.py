import unittest

import numpy as np
import pandas as pd

from config import SECTOR_BENCHMARKS
from predictive_model.features import (
    SECTOR_FEATURE_SETS,
    add_intraday_sector_features,
    add_intraday_vwap_volume_features,
)


class IntradaySectorFeatureTests(unittest.TestCase):
    @staticmethod
    def _frame(periods=28):
        index = pd.date_range(
            "2026-10-01 09:30",
            periods=periods,
            freq="15min",
            tz="America/New_York",
        )
        close = 100.0 + np.arange(periods, dtype=float)
        return pd.DataFrame(
            {
                "Open": close,
                "High": close + 1.0,
                "Low": close - 1.0,
                "Close": close,
                "Volume": 1000.0 + np.arange(periods) * 10.0,
            },
            index=index,
        )

    def test_sector_mapping_uses_liquid_industry_proxies(self):
        self.assertEqual(SECTOR_BENCHMARKS["NVDA"], "SMH")
        self.assertEqual(SECTOR_BENCHMARKS["AAPL"], "XLK")
        self.assertEqual(SECTOR_BENCHMARKS["ABBV"], "XLV")
        self.assertEqual(SECTOR_BENCHMARKS["PLTR"], "IGV")
        self.assertNotIn("SPY", SECTOR_BENCHMARKS)

    def test_returns_use_timestamp_matched_sector_bars(self):
        asset = add_intraday_vwap_volume_features(self._frame())
        sector = self._frame().copy()
        sector["Close"] = 200.0 + np.arange(len(sector), dtype=float) * 0.5
        sector["Open"] = sector["Close"]
        sector["High"] = sector["Close"] + 1.0
        sector["Low"] = sector["Close"] - 1.0

        result = add_intraday_sector_features(asset, sector, feature_set="all")
        row = asset.index[-1]
        expected = (
            asset["Close"].pct_change(4).loc[row]
            - sector["Close"].pct_change(4).loc[row]
        )
        self.assertAlmostEqual(result.loc[row, "sector_relative_return_4"], expected)
        self.assertEqual(result.loc[row, "sector_available"], 1.0)

    def test_alignment_never_uses_future_sector_bar(self):
        asset = add_intraday_vwap_volume_features(self._frame(periods=6))
        sector = self._frame(periods=6)
        sector.index = sector.index + pd.Timedelta(minutes=1)

        result = add_intraday_sector_features(asset, sector, feature_set="returns")

        self.assertEqual(result.iloc[0]["sector_available"], 0.0)
        self.assertEqual(result.iloc[1]["sector_available"], 1.0)
        self.assertAlmostEqual(result.iloc[1]["sector_return_1"], 0.0)
        self.assertAlmostEqual(
            result.iloc[2]["sector_return_1"],
            sector["Close"].iloc[1] / sector["Close"].iloc[0] - 1.0,
        )

    def test_missing_sector_is_explicit_and_neutral(self):
        asset = add_intraday_vwap_volume_features(self._frame())
        result = add_intraday_sector_features(asset, None, feature_set="all")
        self.assertTrue((result["sector_available"] == 0.0).all())
        self.assertTrue((result["sector_relative_return_4"] == 0.0).all())
        self.assertTrue((result["sector_beta_20"] == 0.0).all())

    def test_ablation_sets_control_feature_schema(self):
        asset = add_intraday_vwap_volume_features(self._frame())
        sector = self._frame()

        baseline = add_intraday_sector_features(asset, sector, feature_set="baseline")
        returns = add_intraday_sector_features(asset, sector, feature_set="returns")
        vwap = add_intraday_sector_features(asset, sector, feature_set="vwap")

        self.assertNotIn("sector_available", baseline.columns)
        self.assertIn("sector_relative_return_4", returns.columns)
        self.assertNotIn("sector_beta_20", returns.columns)
        self.assertIn("sector_relative_vwap_distance", vwap.columns)
        self.assertNotIn("sector_relative_return_4", vwap.columns)

    def test_sector_feature_column_order_is_deterministic(self):
        asset = add_intraday_vwap_volume_features(self._frame())
        original_columns = list(asset.columns)
        result = add_intraday_sector_features(
            asset,
            self._frame(),
            feature_set="all",
        )
        added_columns = list(result.columns[len(original_columns):])
        self.assertEqual(
            added_columns,
            [*sorted(SECTOR_FEATURE_SETS["all"]), "sector_available"],
        )


if __name__ == "__main__":
    unittest.main()
