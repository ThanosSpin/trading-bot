#!/usr/bin/env python3
"""Launch one bot cycle with environment-specific durable logging."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, choices=("paper", "live", "live2"))
    args = parser.parse_args()

    bot_dir = Path(__file__).resolve().parent
    os.chdir(bot_dir)
    os.environ["BOT_ENV"] = args.env
    os.environ["PYTHONUNBUFFERED"] = "1"

    # Import only after BOT_ENV is fixed; config selects credentials and paths
    # at import time.
    from config import LOGS_DIR

    log_dir = Path(LOGS_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "trading.log"
    stream = log_path.open("a", buffering=1)
    os.dup2(stream.fileno(), sys.stdout.fileno())
    os.dup2(stream.fileno(), sys.stderr.fileno())

    print(f"[LAUNCHER] env={args.env} checkout={bot_dir} log={log_path}")
    os.execv(sys.executable, [sys.executable, str(bot_dir / "main.py")])


if __name__ == "__main__":
    main()
