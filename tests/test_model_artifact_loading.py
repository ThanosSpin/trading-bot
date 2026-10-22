import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib

from predictive_model import model_xgb


class ModelArtifactLoadingTests(unittest.TestCase):
    def test_existing_artifact_is_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            artifact = {
                "symbol": "NVDA",
                "mode": "intraday_mr",
                "trained_at": "2026-10-23T12:00:00+00:00",
            }
            joblib.dump(artifact, model_dir / "NVDA_intraday_mr_xgb.pkl")

            with patch.object(model_xgb, "MODEL_DIR", model_dir):
                loaded = model_xgb.load_model("NVDA", "intraday_mr")

            self.assertEqual(loaded, artifact)

    def test_missing_artifact_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(model_xgb, "MODEL_DIR", Path(tmp)):
                loaded = model_xgb.load_model("NVDA", "intraday_mr")

            self.assertIsNone(loaded)

    def test_corrupt_artifact_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "NVDA_intraday_mr_xgb.pkl").write_text(
                "not a joblib artifact",
                encoding="utf-8",
            )

            with patch.object(model_xgb, "MODEL_DIR", model_dir):
                loaded = model_xgb.load_model("NVDA", "intraday_mr")

            self.assertIsNone(loaded)


if __name__ == "__main__":
    unittest.main()
