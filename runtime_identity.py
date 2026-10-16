"""Per-environment process locking and order execution identity."""

from __future__ import annotations

import atexit
import fcntl
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from config import BOT_ENV, LOGS_DIR


ENVIRONMENT = str(BOT_ENV).strip().lower()
EXECUTION_ID = (
    f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    f"-{os.getpid()}-{uuid.uuid4().hex[:6]}"
)
# Keep this distinct from any outer cron/flock filename. An outer wrapper may
# hold its own lock while this in-process lock protects every direct entrypoint.
LOCK_PATH = Path(f"/tmp/trading-bot-runtime-{ENVIRONMENT}.lock")
EXPECTED_LOG_PATH = Path(LOGS_DIR) / "trading.log"
_lock_handle = None


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


GIT_COMMIT = _git_commit()
SOURCE_CHECKOUT = str(Path(__file__).resolve().parent)


def acquire_runtime_lock() -> None:
    """Ensure only one bot process controls an account environment."""
    global _lock_handle
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    handle = LOCK_PATH.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.seek(0)
        owner = handle.read().strip() or "unknown owner"
        handle.close()
        raise RuntimeError(
            f"Another {ENVIRONMENT} trading process holds {LOCK_PATH}: {owner}"
        ) from exc

    handle.seek(0)
    handle.truncate()
    handle.write(
        f"pid={os.getpid()} execution_id={EXECUTION_ID} "
        f"cwd={Path.cwd()} commit={GIT_COMMIT}\n"
    )
    handle.flush()
    _lock_handle = handle
    atexit.register(release_runtime_lock)

    print(
        f"[RUNTIME] env={ENVIRONMENT} pid={os.getpid()} "
        f"execution_id={EXECUTION_ID} commit={GIT_COMMIT} cwd={Path.cwd()}"
    )
    print(f"[RUNTIME] lock={LOCK_PATH} expected_log={EXPECTED_LOG_PATH}")


def release_runtime_lock() -> None:
    global _lock_handle
    if _lock_handle is None:
        return
    try:
        fcntl.flock(_lock_handle.fileno(), fcntl.LOCK_UN)
        _lock_handle.close()
    finally:
        _lock_handle = None


def client_order_id(action: str, symbol: str) -> str:
    """Return a short Alpaca-compatible ID that identifies the bot run."""
    return (
        f"tb-{ENVIRONMENT}-{EXECUTION_ID[-13:]}-"
        f"{str(action).lower()[:1]}-{str(symbol).upper()[:5]}"
    )[:48]
