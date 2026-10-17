import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from build_intraday_dataset import regular_session_only
from freeze_research_dataset import build_manifest
from lock_unified_design import select_designs
from validate_locked_design import align_timestamp_to_index
from unified_intraday_research import (
    add_causal_regime_feature,
    aggregate_non_overlapping_trades,
    model_factory,
    split_development_holdout,
    verify_dataset_manifest,
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

    def test_latest_sessions_are_reserved_from_development(self):
        sessions = []
        for day in pd.bdate_range("2026-01-02", periods=70):
            sessions.extend(
                pd.date_range(
                    f"{day.date()} 09:30", periods=3, freq="15min",
                    tz="America/New_York",
                )
            )
        frame = pd.DataFrame({"Close": np.arange(len(sessions))}, index=sessions)
        development, holdout, holdout_start = split_development_holdout(frame, 10)
        self.assertEqual(len(pd.Index(holdout.index.date).unique()), 10)
        self.assertLess(development.index.max(), holdout.index.min())
        self.assertEqual(holdout_start, str(holdout.index[0].date()))

    def test_frozen_manifest_detects_dataset_change(self):
        with TemporaryDirectory() as directory:
            data_dir = Path(directory)
            path = data_dir / "NVDA_15min.parquet"
            self._frame(30).to_parquet(path)
            manifest = build_manifest(data_dir)
            manifest_path = data_dir / "manifest.json"
            import json
            manifest_path.write_text(json.dumps(manifest))
            self.assertEqual(
                verify_dataset_manifest(data_dir, manifest_path), manifest["dataset_id"]
            )
            changed = self._frame(30)
            changed.iloc[-1, changed.columns.get_loc("Close")] += 1
            changed.to_parquet(path)
            with self.assertRaisesRegex(ValueError, "frozen dataset changed"):
                verify_dataset_manifest(data_dir, manifest_path)

    def test_lock_selects_only_stable_best_candidate(self):
        report = {
            "results": {
                "NVDA": {
                    "sector_benchmark": "SMH",
                    "sector_feature_set": "baseline",
                    "designs": {
                        "30m_050vol": {
                            "horizon_bars": 2,
                            "horizon_minutes": 30,
                            "volatility_multiplier": 0.5,
                            "models": {
                                "logistic": {
                                    "stable_candidate": True,
                                    "net_return": 0.04,
                                    "profit_factor": 1.3,
                                }
                            },
                        },
                        "60m_050vol": {
                            "horizon_bars": 4,
                            "horizon_minutes": 60,
                            "volatility_multiplier": 0.5,
                            "models": {
                                "xgboost": {
                                    "stable_candidate": False,
                                    "net_return": 0.20,
                                    "profit_factor": 2.0,
                                }
                            },
                        },
                    },
                }
            }
        }
        selected = select_designs(report)
        self.assertEqual(selected["NVDA"]["target_design"], "30m_050vol")
        self.assertEqual(selected["NVDA"]["model_kind"], "logistic")

    def test_validation_boundary_matches_naive_feature_index(self):
        index = pd.date_range("2026-07-21 13:30", periods=2, freq="15min")
        boundary = pd.Timestamp("2026-07-21 13:30", tz="UTC")
        aligned = align_timestamp_to_index(boundary, index)
        self.assertIsNone(aligned.tzinfo)
        self.assertEqual(aligned, index[0])

    def test_validation_boundary_matches_aware_feature_index(self):
        index = pd.date_range(
            "2026-07-21 09:30", periods=2, freq="15min", tz="America/New_York"
        )
        boundary = pd.Timestamp("2026-07-21 13:30", tz="UTC")
        aligned = align_timestamp_to_index(boundary, index)
        self.assertEqual(aligned, index[0])


if __name__ == "__main__":
    unittest.main()
