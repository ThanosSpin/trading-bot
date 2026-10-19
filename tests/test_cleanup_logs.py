import os
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from cleanup_logs import cleanup_logs_directory


class CleanupLogsDirectoryTests(unittest.TestCase):
    def test_removes_old_files_recursively_and_keeps_recent_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logs_dir = Path(temp_dir) / "logs"
            live2_dir = logs_dir / "live2"
            live2_dir.mkdir(parents=True)

            old_top_level = logs_dir / "trading.log"
            old_nested = live2_dir / "trading.log"
            recent_nested = live2_dir / "update_portfolio.log"

            for path in (old_top_level, old_nested, recent_nested):
                path.write_text("test\n", encoding="utf-8")

            old_timestamp = (datetime.now() - timedelta(days=31)).timestamp()
            recent_timestamp = (datetime.now() - timedelta(days=1)).timestamp()
            os.utime(old_top_level, (old_timestamp, old_timestamp))
            os.utime(old_nested, (old_timestamp, old_timestamp))
            os.utime(recent_nested, (recent_timestamp, recent_timestamp))

            cleanup_logs_directory(str(logs_dir), days_to_keep=30)

            self.assertFalse(old_top_level.exists())
            self.assertFalse(old_nested.exists())
            self.assertTrue(recent_nested.exists())
            self.assertTrue(live2_dir.exists())

    def test_skips_symbolic_links(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            logs_dir = root / "logs"
            logs_dir.mkdir()
            external_file = root / "outside.log"
            external_file.write_text("keep\n", encoding="utf-8")
            link = logs_dir / "linked.log"

            try:
                link.symlink_to(external_file)
            except OSError:
                self.skipTest("Symbolic links are unavailable on this platform")

            old_timestamp = (datetime.now() - timedelta(days=31)).timestamp()
            os.utime(external_file, (old_timestamp, old_timestamp))

            cleanup_logs_directory(str(logs_dir), days_to_keep=30)

            self.assertTrue(link.is_symlink())
            self.assertTrue(external_file.exists())


if __name__ == "__main__":
    unittest.main()
