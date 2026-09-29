import unittest

import numpy as np
import pandas as pd

from predictive_model.model_evaluation import (
    EVALUATION_VERSION,
    cost_aware_metrics,
    evaluate_walk_forward,
    promotion_decision,
    walk_forward_splits,
)


class ModelEvaluationTests(unittest.TestCase):
    def test_walk_forward_splits_have_embargo_and_expand(self):
        folds = walk_forward_splits(240, n_splits=4, gap_bars=2)
        self.assertEqual(len(folds), 4)
        previous_train_end = 0
        for fold in folds:
            self.assertGreater(fold["train_end"], previous_train_end)
            self.assertEqual(fold["test_start"] - fold["train_end"], 2)
            self.assertGreater(fold["test_end"], fold["test_start"])
            previous_train_end = fold["train_end"]

    def test_transaction_costs_reduce_net_return(self):
        actual = np.array([1, 1, 0, 1])
        probability = np.array([0.8, 0.7, 0.2, 0.9])
        returns = np.array([0.01, 0.005, -0.004, 0.002])
        free = cost_aware_metrics(actual, probability, returns, 0.5, cost_bps=0)
        costly = cost_aware_metrics(actual, probability, returns, 0.5, cost_bps=10)
        self.assertLess(costly["net_return"], free["net_return"])
        self.assertEqual(costly["trade_count"], 2)
        self.assertEqual(costly["bars_in_market"], 3)

    def test_fold_boundary_resets_a_held_position(self):
        metrics = cost_aware_metrics(
            actual=np.ones(4),
            probability=np.full(4, 0.9),
            forward_returns=np.full(4, 0.002),
            threshold=0.5,
            cost_bps=10,
            group_ids=np.array([1, 1, 2, 2]),
        )
        self.assertEqual(metrics["trade_count"], 2)
        self.assertEqual(metrics["bars_in_market"], 4)

    def test_walk_forward_evaluation_records_out_of_fold_predictions(self):
        rows = 180
        index = pd.date_range("2025-01-01", periods=rows, freq="D")
        signal = np.tile([0.0, 1.0], rows // 2)
        X = pd.DataFrame(
            {
                "signal": signal,
                "noise": np.sin(np.arange(rows) / 5.0),
            },
            index=index,
        )
        y = pd.Series(signal.astype(int), index=index)
        training_target = y.astype(float).copy()
        training_target.iloc[::3] = np.nan
        movement_target = pd.Series(
            np.tile([0, 1, 1], rows // 3), index=index, dtype=int
        )
        returns = pd.Series(np.where(y == 1, 0.01, -0.01), index=index)
        result = evaluate_walk_forward(
            X,
            y,
            returns,
            model_params={
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": "hist",
                "n_estimators": 8,
                "max_depth": 2,
                "learning_rate": 0.2,
                "random_state": 42,
                "n_jobs": 1,
            },
            mode="daily",
            n_splits=3,
            gap_bars=1,
            cost_bps=10,
            training_target=training_target,
            movement_target=movement_target,
            holding_period_bars=4,
        )
        self.assertEqual(result["n_splits"], 3)
        self.assertGreater(len(result["predictions"]), 0)
        record_index = pd.to_datetime(
            [record["timestamp"] for record in result["predictions"]]
        )
        self.assertGreater(
            len(record_index),
            int(training_target.reindex(record_index).notna().sum()),
        )
        self.assertGreater(result["aggregate"]["trade_count"], 0)
        self.assertIn("movement_probability", result["predictions"][0])
        self.assertEqual(result["holding_period_bars"], 4)
        self.assertTrue(
            all(fold["test_rows_non_overlapping"] < fold["test_rows_all_market_bars"]
                for fold in result["folds"])
        )
        for fold in result["folds"]:
            self.assertEqual(fold["test_start"] - fold["train_end"], 1)

    def test_legacy_champion_allows_first_candidate_after_absolute_gates(self):
        records = []
        for i in range(40):
            actual = i % 2
            records.append(
                {
                    "timestamp": f"2025-01-{i + 1:02d}",
                    "actual": actual,
                    "probability": 0.8 if actual else 0.2,
                    "forward_return": 0.01 if actual else -0.01,
                    "threshold": 0.5,
                    "fold": 1,
                }
            )
        candidate = {
            "mode": "daily",
            "walk_forward_evaluation": {
                "version": EVALUATION_VERSION,
                "n_splits": 4,
                "gap_bars": 1,
                "cost_bps": 10.0,
                "aggregate": {
                    "trade_count": 20,
                    "net_return": 0.10,
                    "profit_factor": 2.0,
                    "brier_score": 0.04,
                    "max_drawdown": -0.03,
                },
                "predictions": records,
            },
        }
        decision = promotion_decision(candidate, {"mode": "daily"})
        self.assertTrue(decision["accepted"])
        self.assertEqual(decision["kind"], "absolute_gate")

    def test_challenger_must_improve_on_overlapping_observations(self):
        champion_records = []
        challenger_records = []
        for i in range(40):
            actual = i % 2
            realized = 0.01 if actual else -0.01
            common = {
                "timestamp": f"row-{i:03d}",
                "actual": actual,
                "forward_return": realized,
                "threshold": 0.5,
                "fold": 1,
            }
            champion_records.append(
                {**common, "probability": 0.55 if actual else 0.45}
            )
            challenger_records.append(
                {**common, "probability": 0.85 if actual else 0.15}
            )

        aggregate = {
            "trade_count": 20,
            "net_return": 0.10,
            "profit_factor": 2.0,
            "brier_score": 0.04,
            "max_drawdown": -0.03,
        }
        champion = {
            "mode": "daily",
            "walk_forward_evaluation": {
                "version": EVALUATION_VERSION,
                "n_splits": 4,
                "gap_bars": 1,
                "cost_bps": 10.0,
                "aggregate": aggregate,
                "predictions": champion_records,
            },
        }
        challenger = {
            "mode": "daily",
            "walk_forward_evaluation": {
                "version": EVALUATION_VERSION,
                "n_splits": 4,
                "gap_bars": 1,
                "cost_bps": 10.0,
                "aggregate": aggregate,
                "predictions": challenger_records,
            },
        }
        decision = promotion_decision(challenger, champion)
        self.assertTrue(decision["accepted"])
        self.assertEqual(decision["kind"], "champion_challenger")
        self.assertEqual(decision["overlap_samples"], 40)


if __name__ == "__main__":
    unittest.main()
