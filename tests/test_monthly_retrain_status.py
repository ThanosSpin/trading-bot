import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import monthly_model_retrain as monthly


class MonthlyRetrainStatusTests(unittest.TestCase):
    def test_accepted_artifacts_promote_independently(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            monthly, "MODEL_DIR", Path(tmp) / "models"
        ):
            model_dir = monthly.MODEL_DIR
            stage_dir = model_dir / ".monthly_retrain_staging" / "run"
            stage_dir.mkdir(parents=True)
            model_dir.mkdir(parents=True, exist_ok=True)

            joblib = monthly.joblib
            joblib.dump({"version": "champion"}, model_dir / "NVDA_daily_xgb.pkl")
            joblib.dump({"version": "champion"}, model_dir / "AAPL_daily_xgb.pkl")
            joblib.dump({"version": "challenger"}, stage_dir / "NVDA_daily_xgb.pkl")

            promoted, failures = monthly._promote_accepted(
                stage_dir,
                ["NVDA/daily"],
            )

            self.assertEqual(promoted, ["NVDA/daily"])
            self.assertEqual(failures, [])
            self.assertEqual(
                joblib.load(model_dir / "NVDA_daily_xgb.pkl")["version"],
                "challenger",
            )
            self.assertEqual(
                joblib.load(model_dir / "AAPL_daily_xgb.pkl")["version"],
                "champion",
            )

    def test_missing_accepted_artifact_does_not_block_other_promotion(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            monthly, "MODEL_DIR", Path(tmp) / "models"
        ):
            model_dir = monthly.MODEL_DIR
            stage_dir = model_dir / ".monthly_retrain_staging" / "run"
            stage_dir.mkdir(parents=True)
            monthly.joblib.dump(
                {"version": "challenger"},
                stage_dir / "AAPL_daily_xgb.pkl",
            )

            promoted, failures = monthly._promote_accepted(
                stage_dir,
                ["NVDA/daily", "AAPL/daily"],
            )

            self.assertEqual(promoted, ["AAPL/daily"])
            self.assertEqual(len(failures), 1)
            self.assertIn("NVDA/daily: FileNotFoundError", failures[0])
            self.assertTrue((model_dir / "AAPL_daily_xgb.pkl").exists())

    def test_gate_rejection_is_reported_without_failure_or_traceback(self):
        artifact = {
            "trained_at": "2026-10-01T00:00:00",
            "features": ["feature"],
            "split_metadata": {},
            "metrics": {},
        }
        decision = {
            "accepted": False,
            "kind": "absolute_gate",
            "overlap_samples": 0,
            "reasons": ["candidate has fewer than 20 evaluated trades"],
        }

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            monthly, "MODES", ("daily",)
        ), patch.object(
            monthly, "MODEL_DIR", Path(tmp) / "models"
        ), patch.object(
            monthly, "_fetch_training_data", return_value=pd.DataFrame({"x": [1]})
        ), patch.object(
            monthly, "train_model", return_value=artifact
        ), patch.object(
            monthly, "validate_artifact"
        ), patch.object(
            monthly, "promotion_decision", return_value=decision
        ):
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                summaries, rejections, failures = monthly._train_to_stage(
                    ["NVDA"], Path(tmp) / "stage", enforce_promotion_gate=True
                )

        self.assertEqual(failures, [])
        self.assertEqual(
            rejections,
            ["NVDA/daily: candidate has fewer than 20 evaluated trades"],
        )
        self.assertEqual(summaries["NVDA/daily"]["status"], "rejected")
        self.assertEqual(stderr.getvalue(), "")

    def test_main_returns_success_when_only_gate_rejections_occur(self):
        summary = {
            "promotion": {"accepted": False},
            "status": "rejected",
        }
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            monthly, "MODES", ("daily",)
        ), patch.object(
            monthly, "MODEL_DIR", Path(tmp) / "models"
        ), patch.object(
            monthly, "_symbols", return_value=["NVDA"]
        ), patch.object(
            monthly, "_train_to_stage",
            return_value=(
                {"NVDA/daily": summary},
                ["NVDA/daily: insufficient evaluated trades"],
                [],
            ),
        ), patch.object(
            monthly, "_write_report", return_value=Path(tmp) / "report.json"
        ), patch(
            "sys.argv", ["monthly_model_retrain.py", "--no-email"]
        ):
            result = monthly.main()

        self.assertEqual(result, 0)

    def test_unexpected_training_exception_remains_failure(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            monthly, "MODES", ("daily",)
        ), patch.object(
            monthly, "MODEL_DIR", Path(tmp) / "models"
        ), patch.object(
            monthly, "_fetch_training_data", return_value=pd.DataFrame({"x": [1]})
        ), patch.object(
            monthly, "train_model", side_effect=RuntimeError("training crashed")
        ):
            summaries, rejections, failures = monthly._train_to_stage(
                ["NVDA"], Path(tmp) / "stage", enforce_promotion_gate=True
            )

        self.assertEqual(summaries, {})
        self.assertEqual(rejections, [])
        self.assertEqual(
            failures, ["NVDA/daily: RuntimeError: training crashed"]
        )


if __name__ == "__main__":
    unittest.main()
