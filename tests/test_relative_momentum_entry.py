import unittest
from datetime import datetime

import pytz

from strategy import (
    _check_aapl_guarded_momentum_entry,
    check_relative_momentum_entry,
)


class RelativeMomentumEntryTests(unittest.TestCase):
    def _diagnostics(self, **overrides):
        values = {
            "relative_market_available": 1.0,
            "movement_expected": True,
            "session_return": 0.009,
            "relative_session_return": 0.012,
            "relative_return_4": 0.004,
            "relative_strength_accel": 0.001,
            "vwap_distance": 0.003,
            "vwap_slope_2": 0.001,
            "vwap_slope_4": 0.001,
            "volume_time_ratio": 1.15,
            "intraday_mom": 0.006,
        }
        values.update(overrides)
        return {"PLTR": values}

    @staticmethod
    def _market_time(hour=11, minute=0):
        return pytz.timezone("America/New_York").localize(
            datetime(2026, 10, 8, hour, minute)
        )

    def test_qualifies_before_move_is_extended(self):
        qualifies, score, reason = check_relative_momentum_entry(
            "PLTR",
            self._diagnostics(),
            {"PLTR": 0.54},
            now_ny=self._market_time(),
        )
        self.assertTrue(qualifies)
        self.assertGreater(score, 0.54)
        self.assertIn("RELATIVE MOMENTUM", reason)

    def test_rejects_move_above_chasing_limit(self):
        qualifies, _, _ = check_relative_momentum_entry(
            "PLTR",
            self._diagnostics(session_return=0.021),
            {"PLTR": 0.60},
            now_ny=self._market_time(),
        )
        self.assertFalse(qualifies)

    def test_rejects_weak_vwap_or_bearish_model(self):
        weak_vwap, _, _ = check_relative_momentum_entry(
            "PLTR",
            self._diagnostics(vwap_slope_2=-0.001),
            {"PLTR": 0.60},
            now_ny=self._market_time(),
        )
        bearish, _, _ = check_relative_momentum_entry(
            "PLTR",
            self._diagnostics(),
            {"PLTR": 0.49},
            now_ny=self._market_time(),
        )
        self.assertFalse(weak_vwap)
        self.assertFalse(bearish)

    def test_rejection_explains_failed_conditions(self):
        qualifies, _, reason = check_relative_momentum_entry(
            "PLTR",
            self._diagnostics(volume_time_ratio=0.50),
            {"PLTR": 0.49},
            now_ny=self._market_time(),
        )
        self.assertFalse(qualifies)
        self.assertIn("volume_time=0.50", reason)
        self.assertIn("model=0.490", reason)

    def test_rejects_late_entry(self):
        qualifies, _, _ = check_relative_momentum_entry(
            "PLTR",
            self._diagnostics(),
            {"PLTR": 0.60},
            now_ny=self._market_time(15, 5),
        )
        self.assertFalse(qualifies)

    def test_aapl_guarded_momentum_uses_intraday_probability(self):
        diagnostics = {
            "AAPL": {
                "relative_market_available": 1.0,
                "movement_expected": True,
                "intraday_model_used": "intraday_mom",
                "intraday_prob": 0.597,
                "session_return": 0.0142,
                "relative_session_return": 0.0056,
                "relative_return_4": 0.0147,
                "vwap_distance": 0.0054,
                "vwap_slope_2": 0.0004,
                "volume_time_ratio": 0.52,
                "intraday_mom": 0.0051,
            }
        }
        qualifies, score, reason = _check_aapl_guarded_momentum_entry(
            "AAPL", diagnostics, now_ny=self._market_time()
        )
        self.assertTrue(qualifies)
        self.assertGreater(score, 0.597)
        self.assertIn("size=50%", reason)

    def test_aapl_guarded_momentum_requires_confirmation(self):
        diagnostics = {
            "AAPL": {
                "relative_market_available": 1.0,
                "movement_expected": True,
                "intraday_model_used": "intraday_mom",
                "intraday_prob": 0.597,
                "session_return": 0.0142,
                "relative_session_return": 0.0056,
                "relative_return_4": 0.0147,
                "vwap_distance": 0.0054,
                "vwap_slope_2": -0.0001,
                "volume_time_ratio": 1.0,
                "intraday_mom": 0.0051,
            }
        }
        qualifies, _, reason = _check_aapl_guarded_momentum_entry(
            "AAPL", diagnostics, now_ny=self._market_time()
        )
        self.assertFalse(qualifies)
        self.assertIn("VWAP_slope_30m", reason)

    def test_aapl_guarded_momentum_allows_session_lag_with_recent_strength(self):
        diagnostics = {
            "AAPL": {
                "relative_market_available": 1.0,
                "movement_expected": True,
                "intraday_model_used": "intraday_mom",
                "intraday_prob": 0.61,
                "session_return": 0.006,
                "relative_session_return": -0.004,
                "relative_return_4": 0.001,
                "vwap_distance": 0.003,
                "vwap_slope_2": 0.0002,
                "volume_time_ratio": 0.9,
                "intraday_mom": 0.003,
            }
        }
        qualifies, _, reason = _check_aapl_guarded_momentum_entry(
            "AAPL", diagnostics, now_ny=self._market_time()
        )
        self.assertTrue(qualifies)
        self.assertIn("vs_SPY=-0.40%", reason)

    def test_guarded_momentum_applies_to_abbv(self):
        diagnostics = {
            "ABBV": {
                "relative_market_available": 1.0,
                "movement_expected": True,
                "intraday_model_used": "intraday_mom",
                "intraday_prob": 0.61,
                "session_return": 0.006,
                "relative_session_return": -0.002,
                "relative_return_4": 0.001,
                "vwap_distance": 0.003,
                "vwap_slope_2": 0.0002,
                "volume_time_ratio": 0.9,
                "intraday_mom": 0.003,
            }
        }
        qualifies, _, reason = check_relative_momentum_entry(
            "ABBV", diagnostics, {"ABBV": 0.54}, now_ny=self._market_time()
        )
        self.assertTrue(qualifies)
        self.assertIn("[GUARDED MOMENTUM] ABBV", reason)

    def test_aapl_guarded_momentum_requires_positive_recent_relative_strength(self):
        diagnostics = {
            "AAPL": {
                "relative_market_available": 1.0,
                "movement_expected": True,
                "intraday_model_used": "intraday_mom",
                "intraday_prob": 0.61,
                "session_return": 0.006,
                "relative_session_return": 0.004,
                "relative_return_4": 0.0,
                "vwap_distance": 0.003,
                "vwap_slope_2": 0.0002,
                "volume_time_ratio": 0.9,
                "intraday_mom": 0.003,
            }
        }
        qualifies, _, reason = _check_aapl_guarded_momentum_entry(
            "AAPL", diagnostics, now_ny=self._market_time()
        )
        self.assertFalse(qualifies)
        self.assertIn("rel_60m", reason)


if __name__ == "__main__":
    unittest.main()
