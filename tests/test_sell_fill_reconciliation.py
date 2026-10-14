import unittest
from unittest.mock import patch

import main


class SellFillReconciliationTests(unittest.TestCase):
    def test_changed_reconciliation_is_persisted_immediately(self):
        fills = [{"id": "sell-1"}]
        with patch.object(main, "get_recent_filled_sells", return_value=fills), patch.object(
            main, "record_broker_sell_fills", return_value=True
        ) as record, patch.object(main, "save_session_state") as save:
            changed = main.reconcile_recent_sell_fills(["NVDA"])

        self.assertTrue(changed)
        record.assert_called_once_with(fills)
        save.assert_called_once_with()

    def test_unchanged_reconciliation_is_not_rewritten(self):
        with patch.object(main, "get_recent_filled_sells", return_value=[]), patch.object(
            main, "record_broker_sell_fills", return_value=False
        ), patch.object(main, "save_session_state") as save:
            changed = main.reconcile_recent_sell_fills(["NVDA"])

        self.assertFalse(changed)
        save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
