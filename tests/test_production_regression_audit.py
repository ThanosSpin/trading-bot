import unittest

import pandas as pd

from production_regression_audit import (
    assign_periods,
    find_near_duplicate_orders,
)


class ProductionRegressionAuditTests(unittest.TestCase):
    def test_period_assignment_uses_fixed_cutoff(self):
        cycles = pd.DataFrame(
            {
                "entry_time": pd.to_datetime(
                    ["2026-09-20T14:00:00Z", "2026-09-25T14:00:00Z"], utc=True
                )
            }
        )
        result = assign_periods(cycles, pd.Timestamp("2026-09-24", tz="UTC"))
        post = result[result["period"].eq("post_cutoff")]
        pre = result[result["period"].eq("pre_cutoff_90d")]
        self.assertEqual(len(post), 1)
        self.assertEqual(len(pre), 1)

    def test_near_duplicate_orders_require_same_source_symbol_action_and_qty(self):
        executions = pd.DataFrame(
            {
                "audit_source": ["paper", "paper", "paper"],
                "symbol": ["NVDA", "NVDA", "AAPL"],
                "action": ["buy", "buy", "buy"],
                "exec_qty": [114.0, 114.0, 114.0],
                "timestamp": pd.to_datetime(
                    [
                        "2026-10-16T14:24:15.4Z",
                        "2026-10-16T14:24:15.7Z",
                        "2026-10-16T14:24:15.8Z",
                    ],
                    utc=True,
                ),
                "broker_order_id": ["first", "second", "third"],
                "source_commit": ["abc", "abc", "abc"],
            }
        )
        result = find_near_duplicate_orders(executions)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["symbol"], "NVDA")
        self.assertEqual(result.iloc[0]["first_broker_order_id"], "first")
        self.assertEqual(result.iloc[0]["second_broker_order_id"], "second")


if __name__ == "__main__":
    unittest.main()
