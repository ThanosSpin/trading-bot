import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from predictive_model.model_monitor import evaluate_predictions


class ModelMonitorTests(unittest.TestCase):
    def test_evaluate_predictions_accepts_fresh_log_without_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            pd.DataFrame(
                [
                    {
                        "timestamp": "2026-09-24T14:00:00+00:00",
                        "symbol": "AAPL",
                        "mode": "daily",
                        "predicted_prob": 0.62,
                        "price": 100.0,
                    }
                ]
            ).to_csv(Path(directory) / "predictions_AAPL.csv", index=False)

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = evaluate_predictions(
                    symbol="AAPL",
                    mode="daily",
                    lookback_days=7,
                    logs_dir=directory,
                )

        self.assertEqual(result["sample_size"], 0)
        self.assertNotIn("[ERROR]", output.getvalue())


if __name__ == "__main__":
    unittest.main()
