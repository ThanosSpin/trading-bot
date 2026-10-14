import unittest

from entry_policy import apply_entry_symbol_guard, is_new_entry_allowed


class EntryPolicyTests(unittest.TestCase):
    def test_paper_allows_research_symbols(self):
        for symbol in ("NVDA", "AAPL", "ABBV", "PLTR", "SPY"):
            self.assertTrue(is_new_entry_allowed(symbol, "paper"))

    def test_live_allows_abbv_and_blocks_pltr_entries(self):
        self.assertTrue(is_new_entry_allowed("NVDA", "live"))
        self.assertTrue(is_new_entry_allowed("AAPL", "live2"))
        self.assertTrue(is_new_entry_allowed("SPY", "live"))
        self.assertTrue(is_new_entry_allowed("ABBV", "live"))
        self.assertFalse(is_new_entry_allowed("PLTR", "live2"))

    def test_guard_preserves_risk_sell_and_blocks_buy(self):
        decisions = {
            "PLTR": {"action": "buy", "qty": 10, "explain": "signal"},
            "ABBV": {
                "action": "sell",
                "qty": 5,
                "explain": "Daily loss guard SELL",
                "risk_exit": "daily_loss_guard",
            },
        }

        guarded = apply_entry_symbol_guard(decisions, "live")

        self.assertEqual(guarded["PLTR"]["action"], "hold")
        self.assertEqual(guarded["PLTR"]["qty"], 0)
        self.assertEqual(guarded["ABBV"]["action"], "sell")
        self.assertEqual(guarded["ABBV"]["qty"], 5)

    def test_funding_sell_is_canceled_when_destination_is_blocked(self):
        decisions = {
            "AAPL": {
                "action": "sell",
                "qty": 4,
                "explain": "AAPL sold to fund PLTR rotation",
            },
            "PLTR": {
                "action": "buy",
                "qty": 4,
                "explain": "PLTR rotation BUY",
            },
        }

        guarded = apply_entry_symbol_guard(decisions, "live")

        self.assertEqual(guarded["PLTR"]["action"], "hold")
        self.assertEqual(guarded["AAPL"]["action"], "hold")
        self.assertTrue(guarded["AAPL"]["funding_sell_blocked"])

    def test_funding_sell_remains_when_allowed_buy_still_exists(self):
        decisions = {
            "SPY": {
                "action": "sell",
                "qty": 2,
                "explain": "SPY SELL (rotate into core)",
            },
            "NVDA": {"action": "buy", "qty": 2, "explain": "priority"},
            "PLTR": {"action": "buy", "qty": 2, "explain": "secondary"},
        }

        guarded = apply_entry_symbol_guard(decisions, "live")

        self.assertEqual(guarded["NVDA"]["action"], "buy")
        self.assertEqual(guarded["PLTR"]["action"], "hold")
        self.assertEqual(guarded["SPY"]["action"], "sell")

    def test_targeted_funding_sell_is_canceled_even_with_other_allowed_buy(self):
        decisions = {
            "AAPL": {
                "action": "sell",
                "qty": 4,
                "explain": "AAPL sold to fund PLTR rotation",
            },
            "NVDA": {"action": "buy", "qty": 2, "explain": "priority"},
            "PLTR": {"action": "buy", "qty": 4, "explain": "rotation BUY"},
        }

        guarded = apply_entry_symbol_guard(decisions, "live")

        self.assertEqual(guarded["NVDA"]["action"], "buy")
        self.assertEqual(guarded["PLTR"]["action"], "hold")
        self.assertEqual(guarded["AAPL"]["action"], "hold")


if __name__ == "__main__":
    unittest.main()
