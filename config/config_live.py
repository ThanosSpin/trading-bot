import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

BOT_ENV = os.getenv("BOT_ENV", "live").strip().lower()

if BOT_ENV == "live":
    DATA_DIR = BASE_DIR / "data"

elif BOT_ENV == "live2":
    DATA_DIR = BASE_DIR / "data" / "live2"

else:
    raise ValueError(
        f"config_live.py cannot be used with "
        f"BOT_ENV={BOT_ENV!r}. "
        "Expected live or live2."
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
