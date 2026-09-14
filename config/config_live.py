import os
from pathlib import Path

from .config_base import (
    LIVE_API_KEY,
    LIVE_API_SECRET,
    LIVE_BASE_URL,
    LIVE2_API_KEY,
    LIVE2_API_SECRET,
    LIVE2_BASE_URL,
)


BASE_DIR = Path(__file__).resolve().parent.parent

BOT_ENV = os.getenv("BOT_ENV", "live").strip().lower()

if BOT_ENV not in {"live", "live2"}:
    raise ValueError(
        f"config_live.py cannot be used with "
        f"BOT_ENV={BOT_ENV!r}. "
        "Expected live or live2."
    )


# ============================================================
# Active Alpaca account
# ============================================================

if BOT_ENV == "live":
    API_KEY = LIVE_API_KEY
    API_SECRET = LIVE_API_SECRET
    BASE_URL = LIVE_BASE_URL

    DATA_DIR = BASE_DIR / "data"

elif BOT_ENV == "live2":
    API_KEY = LIVE2_API_KEY
    API_SECRET = LIVE2_API_SECRET
    BASE_URL = LIVE2_BASE_URL

    DATA_DIR = BASE_DIR / "data" / "live2"


# ============================================================
# Validation
# ============================================================

if not API_KEY:
    raise RuntimeError(
        f"Missing Alpaca API key for BOT_ENV={BOT_ENV!r}"
    )

if not API_SECRET:
    raise RuntimeError(
        f"Missing Alpaca API secret for BOT_ENV={BOT_ENV!r}"
    )

if not BASE_URL:
    raise RuntimeError(
        f"Missing Alpaca BASE_URL for BOT_ENV={BOT_ENV!r}"
    )


# ============================================================
# Environment-specific logs
# ============================================================

LOGS_DIR = BASE_DIR / "logs" / BOT_ENV


# ============================================================
# Create directories
# ============================================================

DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Environment-specific files
# ============================================================

PORTFOLIO_PATH = str(DATA_DIR / "portfolio.json")
LOG_FILE = str(LOGS_DIR / "trading_bot.log")


# ============================================================
# Environment identity
# ============================================================

ENV_NAME = BOT_ENV
USE_LIVE_TRADING = True