import unittest
from datetime import datetime

import pytz

from strategy import check_relative_momentum_entry


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
            self._diagnostics(vwap_slope_4=-0.001),
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

    def test_rejects_late_entry(self):
        qualifies, _, _ = check_relative_momentum_entry(
            "PLTR",
            self._diagnostics(),
            {"PLTR": 0.60},
            now_ny=self._market_time(15, 5),
        )
        self.assertFalse(qualifies)


if __name__ == "__main__":
    unittest.main()
