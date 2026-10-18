import tempfile
import unittest
from pathlib import Path

import joblib

from activate_production_binary_models import DEFAULT_ARTIFACTS, MODES, activate


def _artifact(symbol, mode, target_type="binary", marker="source"):
    return {
        "symbol": symbol,
        "mode": mode,
        "target_type": target_type,
        "model": marker,
        "features": ["feature"],
        "decision_threshold": 0.55,
    }


class ProductionBinaryActivationTests(unittest.TestCase):
    def _directories(self, root, target_type="binary"):
        source = root / "source"
        active = root / "models"
        source.mkdir()
        active.mkdir()
        for mode in MODES:
            name = f"NVDA_{mode}_xgb.pkl"
            joblib.dump(_artifact("NVDA", mode, target_type), source / name)
            joblib.dump(_artifact("NVDA", mode, marker="active"), active / name)
        return source, active

    def test_dry_run_changes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, active = self._directories(Path(tmp))
            result = activate(source, active, [("NVDA", "daily")], apply=False)
            self.assertFalse(result["applied"])
            self.assertEqual(joblib.load(active / "NVDA_daily_xgb.pkl")["model"], "active")

    def test_apply_backs_up_and_replaces_complete_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, active = self._directories(Path(tmp))
            artifacts = [("NVDA", mode) for mode in MODES]
            result = activate(source, active, artifacts, apply=True)
            self.assertTrue(result["applied"])
            self.assertTrue(Path(result["manifest"]).is_file())
            for mode in MODES:
                name = f"NVDA_{mode}_xgb.pkl"
                self.assertEqual(joblib.load(active / name)["model"], "source")
                self.assertEqual(
                    joblib.load(Path(result["backup_dir"]) / name)["model"],
                    "active",
                )

    def test_invalid_source_rejects_entire_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, active = self._directories(Path(tmp))
            joblib.dump(
                _artifact("NVDA", "intraday_mr", target_type="two_stage"),
                source / "NVDA_intraday_mr_xgb.pkl",
            )
            with self.assertRaisesRegex(ValueError, "target_type must be binary"):
                activate(
                    source,
                    active,
                    [("NVDA", "daily"), ("NVDA", "intraday_mr")],
                    apply=True,
                )
            self.assertEqual(joblib.load(active / "NVDA_daily_xgb.pkl")["model"], "active")

    def test_default_plan_only_targets_identified_regressions(self):
        self.assertEqual(
            DEFAULT_ARTIFACTS,
            (
                ("NVDA", "daily"),
                ("NVDA", "intraday_mr"),
                ("AAPL", "daily"),
            ),
        )


if __name__ == "__main__":
    unittest.main()
