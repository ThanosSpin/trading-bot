import unittest
import json
import tempfile
from datetime import datetime, timedelta, timezone
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

    def test_nvda_guarded_momentum_entry_requires_full_confirmation(self):
        diagnostics = {
            "NVDA": {
                "relative_market_available": 1.0,
                "movement_expected": True,
                "intraday_model_used": "intraday_mom",
                "session_return": 0.01,
                "relative_session_return": 0.003,
                "relative_return_4": 0.002,
                "vwap_distance": 0.004,
                "vwap_slope_2": 0.001,
                "volume_time_ratio": 1.1,
                "intraday_mom": 0.008,
            }
        }
        now_ny = datetime(2026, 10, 26, 11, 0, tzinfo=strategy.NY_TZ)

        qualifies, _score, reason = strategy._check_nvda_guarded_momentum_entry(
            "NVDA", diagnostics, {"NVDA": 0.55}, now_ny=now_ny
        )

        self.assertTrue(qualifies)
        self.assertIn("NVDA GUARDED MOMENTUM", reason)
        self.assertIn("initial_exposure=25%", reason)

    def test_nvda_guarded_momentum_entry_rejects_weak_probability(self):
        diagnostics = {
            "NVDA": {
                "relative_market_available": 1.0,
                "movement_expected": True,
                "intraday_model_used": "intraday_mom",
                "session_return": 0.01,
                "relative_session_return": 0.003,
                "relative_return_4": 0.002,
                "vwap_distance": 0.004,
                "vwap_slope_2": 0.001,
                "volume_time_ratio": 1.1,
                "intraday_mom": 0.008,
            }
        }
        now_ny = datetime(2026, 10, 26, 11, 0, tzinfo=strategy.NY_TZ)

        qualifies, _score, reason = strategy._check_nvda_guarded_momentum_entry(
            "NVDA", diagnostics, {"NVDA": 0.549}, now_ny=now_ny
        )

        self.assertFalse(qualifies)
        self.assertIn("final_prob=0.549", reason)

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

    def test_three_of_four_weak_probabilities_block_core_entries(self):
        symbols = ["NVDA", "AAPL", "ABBV", "PLTR"]
        predictions = {
            "NVDA": 0.648,
            "AAPL": 0.49,
            "ABBV": 0.48,
            "PLTR": 0.47,
        }
        with patch.object(strategy, "WEAK_RATIO_THRESHOLD", 0.75), patch.object(
            strategy, "fetch_previous_close", return_value=100.0
        ), patch.object(strategy, "fetch_latest_price", return_value=100.0):
            self.assertTrue(strategy._weak_market(symbols, predictions))

    def test_four_of_four_weak_probabilities_block_core_entries(self):
        symbols = ["NVDA", "AAPL", "ABBV", "PLTR"]
        predictions = {symbol: 0.49 for symbol in symbols}
        with patch.object(strategy, "WEAK_RATIO_THRESHOLD", 0.75), patch.object(
            strategy, "fetch_previous_close", return_value=100.0
        ), patch.object(strategy, "fetch_latest_price", return_value=100.0):
            self.assertTrue(strategy._weak_market(symbols, predictions))

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

    def test_prior_ny_day_sell_does_not_mutate_session_state(self):
        now_ny = datetime.now(timezone.utc).astimezone(strategy.NY_TZ)
        prior_day = now_ny - timedelta(days=1)

        changed = strategy.record_broker_sell_fills(
            [
                {
                    "id": "old-order",
                    "symbol": "NVDA",
                    "filled_qty": 10,
                    "filled_at": prior_day.isoformat(),
                }
            ]
        )

        self.assertFalse(changed)
        self.assertNotIn("NVDA", strategy._session_state["sells"])
        self.assertNotIn("NVDA", strategy._session_state["sell_times"])
        self.assertNotIn(
            "old-order", strategy._session_state["processed_sell_order_ids"]
        )

    def test_current_ny_day_sell_is_recorded_only_once(self):
        fill = {
            "id": "current-order",
            "symbol": "NVDA",
            "filled_qty": 10,
            "filled_at": datetime.now(timezone.utc).isoformat(),
        }

        self.assertTrue(strategy.record_broker_sell_fills([fill]))
        first_time = strategy._session_state["sell_times"]["NVDA"]
        self.assertFalse(strategy.record_broker_sell_fills([fill]))

        self.assertIn("NVDA", strategy._session_state["sells"])
        self.assertEqual(strategy._session_state["sell_times"]["NVDA"], first_time)
        self.assertEqual(
            strategy._session_state["processed_sell_order_ids"], {"current-order"}
        )

    def test_direct_sell_without_timestamp_uses_current_session(self):
        self.assertTrue(strategy.mark_session_sell("AAPL"))
        self.assertIn("AAPL", strategy._session_state["sells"])

    def test_recent_sell_query_excludes_prior_ny_date(self):
        now_ny = datetime.now(timezone.utc).astimezone(trader.NY_TZ)
        current_fill = SimpleNamespace(
            id="current",
            symbol="NVDA",
            side="sell",
            filled_qty="2",
            filled_at=now_ny,
        )
        prior_fill = SimpleNamespace(
            id="prior",
            symbol="NVDA",
            side="sell",
            filled_qty="3",
            filled_at=now_ny - timedelta(days=1),
        )
        client = SimpleNamespace(
            list_orders=lambda **_kwargs: [current_fill, prior_fill]
        )

        with patch.object(trader, "_api", return_value=client):
            fills = trader.get_recent_filled_sells(["NVDA"])

        self.assertEqual([fill["id"] for fill in fills], ["current"])

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
        with patch.object(strategy, "LAST_HOUR_BUY_BUFFER", 0.0), patch.object(
            strategy, "FINAL_15_MIN_BUY_BUFFER", 0.0
        ):
            nvda_buy = strategy._effective_buy_threshold("NVDA", diagnostics)
            aapl_buy = strategy._effective_buy_threshold("AAPL", diagnostics)
        self.assertAlmostEqual(nvda_buy, 0.56)
        self.assertAlmostEqual(
            strategy._effective_sell_threshold("NVDA", diagnostics), 0.52
        )
        self.assertGreaterEqual(
            nvda_buy
            - strategy._effective_sell_threshold("NVDA", diagnostics),
            0.04,
        )
        self.assertAlmostEqual(aapl_buy, 0.62)
        self.assertAlmostEqual(
            strategy._effective_sell_threshold("AAPL", diagnostics), 0.58
        )

    def test_legacy_artifact_keeps_configured_entry_floor(self):
        info = strategy.combine_artifact_decision_thresholds(
            "NVDA", 0.40, 0.45, 0.50
        )
        diagnostics = {"NVDA": {**info, "cost_aware_threshold": False}}
        with patch.object(strategy, "LAST_HOUR_BUY_BUFFER", 0.0), patch.object(
            strategy, "FINAL_15_MIN_BUY_BUFFER", 0.0
        ):
            threshold = strategy._effective_buy_threshold("NVDA", diagnostics)
        self.assertAlmostEqual(threshold, 0.57)

    def test_reduced_pyramid_buffer_keeps_absolute_confidence_floor(self):
        diagnostics = {
            "NVDA": {
                "decision_threshold": 0.45,
                "cost_aware_threshold": False,
            },
            "AAPL": {
                "decision_threshold": 0.52,
                "cost_aware_threshold": False,
            },
        }
        with patch.object(strategy, "LAST_HOUR_BUY_BUFFER", 0.0), patch.object(
            strategy, "FINAL_15_MIN_BUY_BUFFER", 0.0
        ):
            self.assertAlmostEqual(
                strategy._effective_pyramid_threshold("NVDA", diagnostics), 0.65
            )
            self.assertAlmostEqual(
                strategy._effective_pyramid_threshold("AAPL", diagnostics), 0.67
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

    def _pyramid_decision(self, *, price=101.0, avg_price=100.0,
                          session_return=0.01, intraday_mom=0.005):
        pm = SimpleNamespace(
            data={
                "shares": 10,
                "cash": 100000.0,
                "avg_price": avg_price,
                "max_price": max(price, avg_price),
            },
            refresh_live=lambda: None,
            save=lambda: None,
        )
        diagnostics = {
            "NVDA": {
                "decision_threshold": 0.50,
                "session_return": session_return,
                "intraday_mom": intraday_mom,
                "vwap_distance": 0.003,
                "vwap_slope_2": 0.001,
                "relative_session_return": 0.004,
                "relative_return_4": 0.002,
                "volume_time_ratio": 1.1,
            }
        }
        with patch.object(strategy, "PortfolioManager", return_value=pm), patch.object(
            strategy, "fetch_latest_price", return_value=price
        ), patch.object(
            strategy, "fetch_historical_data", return_value=None
        ), patch.object(
            strategy.account_cache,
            "get_account",
            return_value={"equity": 100000.0},
        ):
            return strategy.should_trade("NVDA", 0.90, diagnostics=diagnostics)

    def test_pyramid_requires_profitable_existing_position(self):
        decision = self._pyramid_decision(price=99.8, avg_price=100.0)
        self.assertEqual(decision["action"], "hold")
        self.assertIn("requires position return at least", decision["explain"])

    def test_pyramid_requires_positive_session_return(self):
        decision = self._pyramid_decision(session_return=0.0)
        self.assertEqual(decision["action"], "hold")
        self.assertIn("requires positive session return", decision["explain"])

    def test_pyramid_requires_positive_intraday_momentum(self):
        decision = self._pyramid_decision(intraday_mom=-0.001)
        self.assertEqual(decision["action"], "hold")
        self.assertIn("requires positive intraday momentum", decision["explain"])

    def test_confirmed_nvda_pyramid_can_scale_toward_eighty_percent(self):
        decision = self._pyramid_decision()
        self.assertEqual(decision["action"], "buy")
        self.assertTrue(decision["pyramid"])
        self.assertEqual(decision["qty"], 782)

    def test_pyramid_requires_vwap_relative_strength_and_normalized_volume(self):
        pm = SimpleNamespace(
            data={
                "shares": 10,
                "cash": 100000.0,
                "avg_price": 100.0,
                "max_price": 101.0,
            },
            refresh_live=lambda: None,
            save=lambda: None,
        )
        diagnostics = {
            "NVDA": {
                "decision_threshold": 0.50,
                "session_return": 0.01,
                "intraday_mom": 0.005,
                "vwap_distance": -0.001,
                "vwap_slope_2": 0.001,
                "relative_session_return": 0.004,
                "relative_return_4": 0.002,
                "volume_time_ratio": 1.1,
            }
        }
        with patch.object(strategy, "PortfolioManager", return_value=pm), patch.object(
            strategy, "fetch_latest_price", return_value=101.0
        ), patch.object(strategy, "fetch_historical_data", return_value=None):
            decision = strategy.should_trade("NVDA", 0.90, diagnostics=diagnostics)

        self.assertEqual(decision["action"], "hold")
        self.assertIn("pyramid confirmation incomplete", decision["explain"])

    def test_initial_position_is_capped_at_twenty_five_percent_equity(self):
        with patch.object(
            strategy.account_cache,
            "get_account",
            return_value={"equity": 100000.0},
        ):
            qty = strategy.apply_position_limits(1000, 100.0, 100000.0, "NVDA")
        self.assertEqual(qty, 250)

    def test_confirmed_mr_reversal_enters_small_before_momentum_crosses_zero(self):
        pm = SimpleNamespace(
            data={"shares": 0, "cash": 100000.0},
            refresh_live=lambda: None,
        )
        diagnostics = {
            "ABBV": {
                "decision_threshold": 0.55,
                "movement_expected": True,
                "session_return": -0.001,
                "intraday_regime": "mr",
                "intraday_mom": -0.002,
                "relative_strength_accel": 0.0015,
                "relative_return_4": 0.001,
                "vwap_distance": 0.002,
                "vwap_slope_2": 0.001,
                "volume_time_ratio": 0.9,
            }
        }
        with patch.object(strategy, "PortfolioManager", return_value=pm), patch.object(
            strategy, "fetch_latest_price", return_value=100.0
        ), patch.object(
            strategy.account_cache,
            "get_account",
            return_value={"equity": 100000.0},
        ):
            first = strategy.should_trade("ABBV", 0.70, diagnostics=diagnostics)
            second = strategy.should_trade("ABBV", 0.70, diagnostics=diagnostics)

        self.assertEqual(first["action"], "hold")
        self.assertIn("reversal confirmation 1/2", first["explain"])
        self.assertEqual(second["action"], "buy")
        self.assertEqual(second["qty"], 125)
        self.assertTrue(second["mr_reversal"])
        self.assertIn("MR REVERSAL BUY", second["explain"])

    def test_momentum_regime_cannot_use_early_reversal_entry(self):
        pm = SimpleNamespace(
            data={"shares": 0, "cash": 100000.0},
            refresh_live=lambda: None,
        )
        diagnostics = {
            "NVDA": {
                "decision_threshold": 0.50,
                "movement_expected": True,
                "session_return": 0.001,
                "intraday_regime": "mom",
                "intraday_mom": -0.001,
                "relative_strength_accel": 0.002,
                "relative_return_4": 0.002,
                "vwap_distance": 0.002,
                "vwap_slope_2": 0.001,
                "volume_time_ratio": 1.0,
            }
        }
        with patch.object(strategy, "PortfolioManager", return_value=pm), patch.object(
            strategy, "fetch_latest_price", return_value=100.0
        ):
            decision = strategy.should_trade("NVDA", 0.70, diagnostics=diagnostics)

        self.assertEqual(decision["action"], "hold")
        self.assertNotIn("reversal confirmation", decision["explain"])

    def test_explicit_small_initial_exposure_can_be_below_default(self):
        with patch.object(
            strategy.account_cache,
            "get_account",
            return_value={"equity": 100000.0},
        ):
            qty = strategy.apply_position_limits(
                1000,
                100.0,
                100000.0,
                "ABBV",
                initial_exposure_pct=0.125,
            )
        self.assertEqual(qty, 125)

    def test_confirmed_initial_entry_remains_at_twenty_five_percent(self):
        diagnostics = {
            "NVDA": {
                "decision_threshold": 0.50,
                "session_return": 0.01,
                "intraday_mom": 0.005,
                "vwap_distance": 0.003,
                "vwap_slope_2": 0.001,
                "relative_session_return": 0.004,
                "volume_time_ratio": 1.1,
            }
        }
        with patch.object(strategy, "LAST_HOUR_BUY_BUFFER", 0.0), patch.object(
            strategy, "FINAL_15_MIN_BUY_BUFFER", 0.0
        ):
            exposure_pct = strategy.exceptional_initial_exposure_pct(
                "NVDA", 0.65, diagnostics
            )
        with patch.object(
            strategy.account_cache,
            "get_account",
            return_value={"equity": 100000.0},
        ):
            qty = strategy.apply_position_limits(
                1000,
                100.0,
                100000.0,
                "NVDA",
                initial_exposure_pct=exposure_pct,
            )
        self.assertEqual(exposure_pct, 0.25)
        self.assertEqual(qty, 250)

    def test_confirmed_half_percent_loss_bypasses_minimum_hold(self):
        pm = SimpleNamespace(
            data={
                "shares": 10,
                "cash": 100000.0,
                "avg_price": 100.0,
                "max_price": 100.0,
                "entry_time": datetime.utcnow().isoformat(),
            },
            refresh_live=lambda: None,
            save=lambda: None,
        )
        diagnostics = {
            "AAPL": {
                "decision_threshold": 0.60,
                "intraday_mom": -0.001,
                "vwap_distance": -0.002,
            }
        }
        with patch.object(strategy, "PortfolioManager", return_value=pm), patch.object(
            strategy, "fetch_latest_price", return_value=99.5
        ):
            decision = strategy.should_trade("AAPL", 0.50, diagnostics=diagnostics)

        self.assertEqual(decision["action"], "sell")
        self.assertEqual(decision["risk_exit"], "confirmed_loss_exit")
        self.assertIn("CONFIRMED LOSS EXIT", decision["explain"])

    def test_half_percent_loss_without_bearish_confirmation_does_not_force_exit(self):
        decision = strategy._confirmed_loss_exit(
            "AAPL",
            10,
            99.5,
            100.0,
            0.70,
            0.58,
            {
                "AAPL": {
                    "intraday_mom": 0.002,
                    "vwap_distance": 0.001,
                }
            },
        )
        self.assertIsNone(decision)

    def test_sell_probability_with_rising_price_defers_but_never_adds(self):
        pm = SimpleNamespace(
            data={
                "shares": 10,
                "cash": 100000.0,
                "avg_price": 100.0,
                "max_price": 101.0,
                "entry_time": datetime.utcnow().isoformat(),
            },
            refresh_live=lambda: None,
            save=lambda: None,
        )
        diagnostics = {
            "AAPL": {
                "decision_threshold": 0.60,
                "session_return": 0.01,
                "intraday_mom": 0.003,
                "vwap_distance": 0.002,
                "vwap_slope_2": 0.001,
                "relative_session_return": 0.002,
                "relative_return_4": 0.001,
                "volume_time_ratio": 1.0,
            }
        }
        with patch.object(strategy, "PortfolioManager", return_value=pm), patch.object(
            strategy, "fetch_latest_price", return_value=101.0
        ):
            decision = strategy.should_trade("AAPL", 0.50, diagnostics=diagnostics)

        self.assertEqual(decision["action"], "hold")
        self.assertNotIn("pyramid", decision)
        self.assertIn("SELL deferred", decision["explain"])

    def test_profitable_rising_position_defers_sell_after_minimum_hold(self):
        pm = SimpleNamespace(
            data={
                "shares": 10,
                "cash": 100000.0,
                "avg_price": 100.0,
                "max_price": 101.0,
                "entry_time": (datetime.utcnow() - timedelta(hours=1)).isoformat(),
            },
            refresh_live=lambda: None,
            save=lambda: None,
        )
        diagnostics = {
            "AAPL": {
                "decision_threshold": 0.60,
                "session_return": 0.01,
                "intraday_mom": 0.0105,
                "vwap_distance": 0.003,
                "vwap_slope_2": 0.001,
            }
        }
        with patch.object(strategy, "PortfolioManager", return_value=pm), patch.object(
            strategy, "fetch_latest_price", return_value=101.0
        ):
            decision = strategy.should_trade("AAPL", 0.513, diagnostics=diagnostics)

        self.assertEqual(decision["action"], "hold")
        self.assertTrue(decision["trend_protected_sell"])
        self.assertIn("SELL trend-protected", decision["explain"])

    def test_sell_executes_after_minimum_hold_when_momentum_turns_negative(self):
        pm = SimpleNamespace(
            data={
                "shares": 10,
                "cash": 100000.0,
                "avg_price": 100.0,
                "max_price": 101.0,
                "entry_time": (datetime.utcnow() - timedelta(hours=1)).isoformat(),
            },
            refresh_live=lambda: None,
            save=lambda: None,
        )
        diagnostics = {
            "AAPL": {
                "decision_threshold": 0.60,
                "session_return": 0.01,
                "intraday_mom": -0.001,
                "vwap_distance": 0.003,
                "vwap_slope_2": 0.001,
            }
        }
        with patch.object(strategy, "PortfolioManager", return_value=pm), patch.object(
            strategy, "fetch_latest_price", return_value=101.0
        ):
            decision = strategy.should_trade("AAPL", 0.513, diagnostics=diagnostics)

        self.assertEqual(decision["action"], "sell")
        self.assertNotIn("trend_protected_sell", decision)

    def test_exceptional_entry_falls_back_when_confirmation_is_incomplete(self):
        diagnostics = {
            "NVDA": {
                "decision_threshold": 0.50,
                "session_return": 0.01,
                "intraday_mom": 0.005,
                "vwap_distance": 0.003,
                "vwap_slope_2": 0.001,
                "relative_session_return": -0.001,
                "volume_time_ratio": 1.1,
            }
        }
        self.assertEqual(
            strategy.exceptional_initial_exposure_pct("NVDA", 0.75, diagnostics),
            0.25,
        )

    def test_secondary_pyramid_total_exposure_is_capped_at_fifty_percent(self):
        with patch.object(
            strategy.account_cache,
            "get_account",
            return_value={"equity": 100000.0},
        ):
            qty = strategy.apply_position_limits(
                1000,
                100.0,
                100000.0,
                "ABBV",
                current_position_value=25000.0,
                is_pyramid=True,
            )
        self.assertEqual(qty, 250)

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
