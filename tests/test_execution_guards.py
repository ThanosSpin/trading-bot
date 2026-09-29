import unittest
from types import SimpleNamespace
from unittest.mock import patch

import strategy
import trader


class _OrderClient:
    def __init__(self, statuses):
        self.statuses = iter(statuses)
        self.last = None

    def get_order(self, _order_id):
        try:
            self.last = next(self.statuses)
        except StopIteration:
            pass
        return self.last


class ExecutionGuardTests(unittest.TestCase):
    def setUp(self):
        strategy.reset_session_state()

    def test_pyramid_cooldown_and_cap_reset_when_position_closes(self):
        allowed, _ = strategy._pyramid_allowed("ABBV")
        self.assertTrue(allowed)

        strategy.mark_session_buy("ABBV", is_pyramid=True)
        allowed, reason = strategy._pyramid_allowed("ABBV")
        self.assertFalse(allowed)
        self.assertIn("cooldown", reason)

        strategy._session_state["last_pyramid_times"].pop("ABBV")
        strategy.mark_session_buy("ABBV", is_pyramid=True)
        allowed, reason = strategy._pyramid_allowed("ABBV")
        self.assertFalse(allowed)
        self.assertIn("maximum", reason)

        strategy.mark_session_position_closed("ABBV")
        allowed, _ = strategy._pyramid_allowed("ABBV")
        self.assertTrue(allowed)

    def test_stop_exit_blocks_immediate_reentry(self):
        strategy.mark_session_stop("ABBV", "dollar_stop")
        allowed, reason = strategy._rebuy_allowed("ABBV", 0.99)
        self.assertFalse(allowed)
        self.assertIn("post-stop rebuy blocked", reason)

    def test_rotation_uses_guarded_buy_and_requires_flat_target(self):
        decisions = {
            "AAPL": {"action": "hold", "explain": "RSI blocked"},
            "ABBV": {"action": "buy"},
            "NVDA": {"action": "buy"},
        }
        eligible = strategy._eligible_rotation_targets(
            ["AAPL", "ABBV", "NVDA"],
            decisions,
            {"AAPL": 0, "ABBV": 277, "NVDA": 0},
        )
        self.assertEqual(eligible, ["NVDA"])

    def test_position_stop_scales_with_average_cost_value(self):
        pm = SimpleNamespace(
            data={"shares": 20, "avg_price": 267.378, "max_price": 267.378},
            save=lambda: None,
        )
        with patch.object(strategy, "MAX_LOSS_PER_POSITION_PCT", 0.01), patch.object(
            strategy, "MAX_LOSS_PER_TRADE", None
        ):
            self.assertIsNone(strategy.check_stop_tp("ABBV", 266.76, pm))
            decision = strategy.check_stop_tp("ABBV", 264.70, pm)

        self.assertEqual(decision["action"], "sell")
        self.assertEqual(decision["risk_exit"], "position_stop")
        self.assertIn("1.00% of cost", decision["explain"])

    def test_order_polling_waits_through_partial_fill_to_terminal_fill(self):
        client = _OrderClient(
            [
                SimpleNamespace(status="new", filled_qty="0"),
                SimpleNamespace(status="partially_filled", filled_qty="3"),
                SimpleNamespace(status="filled", filled_qty="5"),
            ]
        )
        with patch.object(trader.time, "sleep", return_value=None):
            result = trader._wait_for_terminal_order(
                client, "order-1", timeout_seconds=5
            )
        self.assertEqual(result.status, "filled")
        self.assertEqual(float(result.filled_qty), 5.0)


if __name__ == "__main__":
    unittest.main()
