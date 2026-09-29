import unittest

import numpy as np
import pandas as pd

from predictive_model.model_xgb import (
    _build_thresholded_binary_target,
    predict_from_model,
)


class _ProbabilityModel:
    def __init__(self, positive_probability):
        self.positive_probability = positive_probability

    def predict_proba(self, frame):
        positive = np.full(len(frame), self.positive_probability, dtype=float)
        return np.column_stack([1.0 - positive, positive])


class TwoStageTargetTests(unittest.TestCase):
    def test_intraday_target_uses_four_fifteen_minute_bars(self):
        frame = pd.DataFrame({"Close": [100, 101, 102, 103, 104, 105]})
        labeled = _build_thresholded_binary_target(
            frame, mode="intraday_mom", use_two_stage=True
        )
        self.assertAlmostEqual(labeled.loc[0, "forward_return"], 0.04)
        self.assertTrue(np.isnan(labeled.loc[2, "forward_return"]))

    def test_sixty_minute_movement_band_is_two_tenths_percent(self):
        frame = pd.DataFrame(
            {"Close": [100.0, 100.0, 100.0, 100.0, 100.1, 100.3]}
        )
        labeled = _build_thresholded_binary_target(
            frame, mode="intraday_mom", use_two_stage=True
        )
        self.assertEqual(labeled.loc[0, "movement_target"], 0)
        self.assertEqual(labeled.loc[1, "movement_target"], 1)

    def test_low_movement_probability_returns_neutral_strategy_score(self):
        artifact = {
            "model": _ProbabilityModel(0.8),
            "movement_model": _ProbabilityModel(0.3),
            "movement_threshold": 0.6,
            "features": ["signal"],
            "feature_schema": {"features": ["signal"]},
            "num_classes": 2,
            "target_type": "two_stage",
            "decision_threshold": 0.6,
            "target_horizon": "60min",
        }
        prediction = predict_from_model(
            artifact, pd.DataFrame({"signal": [1.0]})
        )
        self.assertEqual(prediction["final_prob"], 0.5)
        self.assertFalse(prediction["movement_expected"])
        self.assertAlmostEqual(prediction["bullish_prob"], 0.24)

    def test_high_movement_probability_exposes_direction_probability(self):
        artifact = {
            "model": _ProbabilityModel(0.8),
            "movement_model": _ProbabilityModel(0.7),
            "movement_threshold": 0.6,
            "features": ["signal"],
            "feature_schema": {"features": ["signal"]},
            "num_classes": 2,
            "target_type": "two_stage",
            "decision_threshold": 0.6,
            "target_horizon": "60min",
        }
        prediction = predict_from_model(
            artifact, pd.DataFrame({"signal": [1.0]})
        )
        self.assertEqual(prediction["final_prob"], 0.8)
        self.assertTrue(prediction["movement_expected"])


if __name__ == "__main__":
    unittest.main()
