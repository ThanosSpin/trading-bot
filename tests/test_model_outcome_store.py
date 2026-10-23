import tempfile
import unittest
from pathlib import Path

import pandas as pd

from model_outcome_store import load_outcome_history, upsert_resolved_outcomes


class ModelOutcomeStoreTests(unittest.TestCase):
    def _rows(self):
        return pd.DataFrame(
            [
                {
                    "timestamp": "2026-10-22T14:00:00Z",
                    "symbol": "NVDA",
                    "mode": "intraday_mr",
                    "predicted_prob": 0.6,
                    "price": 100.0,
                    "actual_outcome": 1,
                    "actual_price": 101.0,
                    "return_pct": 1.0,
                    "outcome_horizon": "15min",
                },
                {
                    "timestamp": "2026-10-22T14:06:00Z",
                    "symbol": "NVDA",
                    "mode": "intraday_mr",
                    "predicted_prob": 0.7,
                    "price": 100.0,
                    "actual_outcome": pd.NA,
                },
            ]
        )

    def test_upsert_is_durable_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model_outcomes.csv"
            self.assertEqual(upsert_resolved_outcomes(self._rows(), path), 1)
            self.assertEqual(upsert_resolved_outcomes(self._rows(), path), 0)

            saved = load_outcome_history(path=path)
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved.iloc[0]["symbol"], "NVDA")
            self.assertEqual(saved.iloc[0]["mode"], "intraday_mr")


if __name__ == "__main__":
    unittest.main()
