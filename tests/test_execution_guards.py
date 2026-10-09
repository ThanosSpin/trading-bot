import unittest
import json
import tempfile
from pathlib import Path
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

    def test_momentum_breakout_can_fetch_daily_history(self):
        import pandas as pd

        history = pd.DataFrame({"Close": [100.0] * 60})
        diagnostics = {
            "NVDA": {
                "intraday_mom": 0.01,
                "intraday_vol": 0.013,
                "price": 103.0,
            }
        }
        with patch.object(
            strategy, "fetch_historical_data", return_value=history
        ) as fetch:
            force_buy, reason = strategy.check_momentum_breakout(
                "NVDA", diagnostics, {"NVDA": 0.60}
            )

        self.assertTrue(force_buy)
        self.assertIn("MOMENTUM BREAKOUT", reason)
        fetch.assert_called_once_with("NVDA", period="3mo", interval="1d")

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

    def test_loading_legacy_state_resets_at_new_ny_trading_date(self):
        legacy_state = {
            "buys": ["NVDA"],
            "sells": ["NVDA"],
            "buy_times": {},
            "sell_times": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "session_state_live.json"
            state_path.write_text(json.dumps(legacy_state))
            with patch.object(strategy, "SESSION_STATE_PATH", str(state_path)):
                strategy.load_session_state()
                saved = json.loads(state_path.read_text())

        self.assertNotIn("NVDA", strategy._session_state["buys"])
        self.assertNotIn("NVDA", strategy._session_state["sells"])
        self.assertEqual(
            saved["session_date"],
            strategy._dt.now(strategy.NY_TZ).date().isoformat(),
        )

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

    def test_secondary_buy_is_suppressed_when_all_candidates_are_blocked(self):
        self.assertTrue(
            strategy._suppress_unselected_secondary_buy("ABBV", "buy", None)
        )
        self.assertFalse(
            strategy._suppress_unselected_secondary_buy("ABBV", "buy", "ABBV")
        )

    def test_artifact_thresholds_blend_with_probability_weight(self):
        info = strategy.combine_artifact_decision_thresholds(
            "NVDA", 0.60, 0.64, 0.50
        )
        self.assertAlmostEqual(info["raw_decision_threshold"], 0.62)
        self.assertAlmostEqual(info["decision_threshold"], 0.62)
        self.assertEqual(info["threshold_source"], "blended")

    def test_artifact_boundary_uses_buffers_and_model_safety_floor(self):
        nvda = strategy.combine_artifact_decision_thresholds(
            "NVDA", 0.50, 0.50, 0.50
        )
        aapl = strategy.combine_artifact_decision_thresholds(
            "AAPL", 0.50, 0.50, 0.50
        )
        self.assertAlmostEqual(nvda["decision_threshold"], 0.50)
        self.assertAlmostEqual(aapl["decision_threshold"], 0.50)
        diagnostics = {
            "NVDA": {**nvda, "cost_aware_threshold": True},
            "AAPL": {**aapl, "cost_aware_threshold": True},
        }
        self.assertAlmostEqual(
            strategy._effective_buy_threshold("NVDA", diagnostics), 0.56
        )
        self.assertAlmostEqual(
            strategy._effective_sell_threshold("NVDA", diagnostics), 0.52
        )
        self.assertGreaterEqual(
            strategy._effective_buy_threshold("NVDA", diagnostics)
            - strategy._effective_sell_threshold("NVDA", diagnostics),
            0.04,
        )
        self.assertAlmostEqual(
            strategy._effective_buy_threshold("AAPL", diagnostics), 0.62
        )
        self.assertAlmostEqual(
            strategy._effective_sell_threshold("AAPL", diagnostics), 0.58
        )

    def test_legacy_artifact_keeps_configured_entry_floor(self):
        info = strategy.combine_artifact_decision_thresholds(
            "NVDA", 0.40, 0.45, 0.50
        )
        diagnostics = {"NVDA": {**info, "cost_aware_threshold": False}}
        self.assertAlmostEqual(
            strategy._effective_buy_threshold("NVDA", diagnostics), 0.57
        )

    def test_invalid_artifact_threshold_uses_valid_daily_value(self):
        info = strategy.combine_artifact_decision_thresholds(
            "NVDA", 0.61, float("nan"), float("nan")
        )
        self.assertAlmostEqual(info["decision_threshold"], 0.61)
        self.assertEqual(info["threshold_source"], "daily")

    def test_two_stage_no_movement_suppresses_model_trade(self):
        pm = SimpleNamespace(
            data={"shares": 0, "cash": 10000.0},
            refresh_live=lambda: None,
        )
        diagnostics = {
            "NVDA": {
                "decision_threshold": 0.50,
                "movement_expected": False,
            }
        }
        with patch.object(strategy, "PortfolioManager", return_value=pm), patch.object(
            strategy, "fetch_latest_price", return_value=100.0
        ):
            decision = strategy.should_trade(
                "NVDA", 0.90, diagnostics=diagnostics
            )

        self.assertEqual(decision["action"], "hold")
        self.assertIn("movement gate expects no meaningful move", decision["explain"])

    def test_declining_session_blocks_buy_even_on_high_probability(self):
        pm = SimpleNamespace(
            data={"shares": 0, "cash": 10000.0},
            refresh_live=lambda: None,
        )
        diagnostics = {
            "ABBV": {
                "decision_threshold": 0.45,
                "movement_expected": True,
                "session_return": -0.006,
            }
        }
        with patch.object(strategy, "PortfolioManager", return_value=pm), patch.object(
            strategy, "fetch_latest_price", return_value=260.0
        ):
            decision = strategy.should_trade(
                "ABBV", 0.90, diagnostics=diagnostics
            )

        self.assertEqual(decision["action"], "hold")
        self.assertIn("entry blocked while session return", decision["explain"])

    def test_dynamic_stop_uses_smaller_account_equity_cap(self):
        pm = SimpleNamespace(
            data={"shares": 20, "avg_price": 267.378, "max_price": 267.378},
            save=lambda: None,
        )
        with patch.object(
            strategy, "MAX_LOSS_ACCOUNT_EQUITY_PCT", 0.005
        ), patch.object(strategy, "MAX_LOSS_PER_POSITION_PCT", 0.015), patch.object(
            strategy, "MAX_LOSS_PER_TRADE", None
        ), patch.object(
            strategy.account_cache,
            "get_account",
            return_value={"equity": 10800.0},
        ):
            self.assertIsNone(strategy.check_stop_tp("ABBV", 266.76, pm))
            decision = strategy.check_stop_tp("ABBV", 264.65, pm)

        self.assertEqual(decision["action"], "sell")
        self.assertEqual(decision["risk_exit"], "dynamic_risk_stop")
        self.assertIn("effective cap $54.00", decision["explain"])
        self.assertIn("position=1.50% ($80.21)", decision["explain"])

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
