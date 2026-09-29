import unittest

import strategy


class EntryGuardSelectionTests(unittest.TestCase):
    def test_buy_is_suppressed_when_no_secondary_passes_guards(self):
        self.assertTrue(
            strategy._suppress_unselected_secondary_buy("ABBV", "buy", None)
        )

    def test_selected_secondary_buy_is_preserved(self):
        self.assertFalse(
            strategy._suppress_unselected_secondary_buy("ABBV", "buy", "ABBV")
        )


if __name__ == "__main__":
    unittest.main()
