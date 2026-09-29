import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib

import daily_model_retrain


class DailyAtomicPromotionTests(unittest.TestCase):
    def test_batch_failure_restores_every_champion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = root / "models"
            stage_dir = root / "stage"
            model_dir.mkdir()
            stage_dir.mkdir()

            for symbol in ("AAPL", "ABBV"):
                joblib.dump(
                    {"version": f"champion-{symbol}"},
                    model_dir / f"{symbol}_daily_xgb.pkl",
                )
                joblib.dump(
                    {"version": f"candidate-{symbol}"},
                    stage_dir / f"{symbol}_daily_xgb.pkl",
                )

            calls = []

            def validate(artifact, symbol, mode):
                calls.append(symbol)
                if symbol == "ABBV":
                    raise ValueError("simulated validation failure")

            candidates = {
                symbol: stage_dir / f"{symbol}_daily_xgb.pkl"
                for symbol in ("AAPL", "ABBV")
            }
            with patch.object(daily_model_retrain, "MODEL_DIR", model_dir), patch.object(
                daily_model_retrain, "validate_artifact", side_effect=validate
            ):
                with self.assertRaisesRegex(ValueError, "simulated"):
                    daily_model_retrain._promote_batch(candidates, "test-run")

            self.assertEqual(calls, ["AAPL", "ABBV"])
            for symbol in ("AAPL", "ABBV"):
                restored = joblib.load(model_dir / f"{symbol}_daily_xgb.pkl")
                self.assertEqual(restored["version"], f"champion-{symbol}")


if __name__ == "__main__":
    unittest.main()
