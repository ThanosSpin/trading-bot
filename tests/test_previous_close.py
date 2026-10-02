import unittest
from datetime import date, datetime, timezone

import pandas as pd

from predictive_model.data_loader import previous_completed_close


class PreviousCloseTests(unittest.TestCase):
    def test_excludes_current_incomplete_daily_row(self):
        frame = pd.DataFrame(
            {"Close": [261.59, 259.93, 262.21]},
            index=pd.to_datetime(["2026-09-30", "2026-10-01", "2026-10-02"]),
        )
        self.assertAlmostEqual(
            previous_completed_close(frame, as_of=date(2026, 10, 2)),
            259.93,
        )

    def test_uses_latest_row_when_current_session_is_absent(self):
        frame = pd.DataFrame(
            {"Close": [261.59, 259.93]},
            index=pd.to_datetime(["2026-09-30", "2026-10-01"]),
        )
        self.assertAlmostEqual(
            previous_completed_close(
                frame,
                as_of=datetime(2026, 10, 2, 14, tzinfo=timezone.utc),
            ),
            259.93,
        )


if __name__ == "__main__":
    unittest.main()
