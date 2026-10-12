import unittest

import pandas as pd

from trade_attribution import attribution_summary, build_trade_cycles


class TradeAttributionTests(unittest.TestCase):
    def test_builds_cycle_with_partial_entries_and_exits(self):
        trades = pd.DataFrame(
            [
                {
                    "timestamp": "2026-01-02T14:30:00Z",
                    "symbol": "PLTR",
                    "action": "buy",
                    "qty": 10,
                    "price": 100,
                    "shares_before": 0,
                    "shares_after": 10,
                    "model_used": "intraday_mom",
                    "regime": "mom",
                    "entry_reason": "ordinary",
                    "decision_probability": 0.62,
                    "buy_threshold": 0.56,
                    "sell_threshold": 0.52,
                },
                {
                    "timestamp": "2026-01-02T15:00:00Z",
                    "symbol": "PLTR",
                    "action": "buy",
                    "qty": 5,
                    "price": 102,
                    "shares_before": 10,
                    "shares_after": 15,
                    "entry_reason": "pyramid",
                },
                {
                    "timestamp": "2026-01-02T16:00:00Z",
                    "symbol": "PLTR",
                    "action": "sell",
                    "qty": 8,
                    "price": 104,
                    "shares_before": 15,
                    "shares_after": 7,
                    "exit_reason": "profit_trim",
                },
                {
                    "timestamp": "2026-01-02T17:00:00Z",
                    "symbol": "PLTR",
                    "action": "sell",
                    "qty": 7,
                    "price": 105,
                    "shares_before": 7,
                    "shares_after": 0,
                    "exit_reason": "model_exit",
                },
            ]
        )

        cycles = build_trade_cycles(trades)

        self.assertEqual(len(cycles), 1)
        cycle = cycles.iloc[0]
        self.assertEqual(cycle["model_used"], "intraday_mom")
        self.assertEqual(cycle["regime"], "mom")
        self.assertEqual(cycle["entry_reason"], "ordinary")
        self.assertEqual(cycle["entry_reasons"], "ordinary, pyramid")
        self.assertEqual(cycle["exit_reason"], "model_exit")
        self.assertAlmostEqual(cycle["invested_capital"], 1510.0)
        self.assertAlmostEqual(cycle["sale_proceeds"], 1567.0)
        self.assertAlmostEqual(cycle["pnl_dollars"], 57.0)
        self.assertAlmostEqual(cycle["return_pct"], 57.0 / 1510.0)

    def test_legacy_rows_remain_usable(self):
        legacy = pd.DataFrame(
            [
                {
                    "timestamp": "2026-01-02T14:30:00Z",
                    "symbol": "ABBV",
                    "action": "buy",
                    "qty": 2,
                    "price": 100,
                    "shares": 2,
                },
                {
                    "timestamp": "2026-01-03T14:30:00Z",
                    "symbol": "ABBV",
                    "action": "sell",
                    "qty": 2,
                    "price": 110,
                    "shares": 0,
                },
            ]
        )

        cycles = build_trade_cycles(legacy)

        self.assertEqual(len(cycles), 1)
        self.assertEqual(cycles.iloc[0]["model_used"], "unknown")
        self.assertAlmostEqual(cycles.iloc[0]["return_pct"], 0.10)

    def test_summary_calculates_dollar_and_return_profit_factor(self):
        cycles = pd.DataFrame(
            [
                {
                    "symbol": "ABBV",
                    "model_used": "intraday_mr",
                    "regime": "mr",
                    "entry_reason": "ordinary",
                    "exit_reason": "model_exit",
                    "pnl_dollars": 20.0,
                    "return_pct": 0.02,
                },
                {
                    "symbol": "ABBV",
                    "model_used": "intraday_mr",
                    "regime": "mr",
                    "entry_reason": "ordinary",
                    "exit_reason": "model_exit",
                    "pnl_dollars": -10.0,
                    "return_pct": -0.005,
                },
            ]
        )

        summary = attribution_summary(cycles)

        self.assertEqual(int(summary.iloc[0]["cycles"]), 2)
        self.assertAlmostEqual(summary.iloc[0]["dollar_profit_factor"], 2.0)
        self.assertAlmostEqual(summary.iloc[0]["return_profit_factor"], 4.0)


if __name__ == "__main__":
    unittest.main()
