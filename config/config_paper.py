from .config_base import *
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

BOT_ENV = "paper"
ENV_NAME = "paper"


# ============================================================
# Active Alpaca account
# ============================================================

API_KEY = PAPER_API_KEY
API_SECRET = PAPER_API_SECRET
BASE_URL = PAPER_BASE_URL


if not API_KEY:
    raise RuntimeError("Missing paper Alpaca API key.")

if not API_SECRET:
    raise RuntimeError("Missing paper Alpaca API secret.")


# ============================================================
# Paper directories
# ============================================================

DATA_DIR = BASE_DIR / "data_paper"
LOGS_DIR = BASE_DIR / "logs_paper"

DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

PORTFOLIO_PATH = str(DATA_DIR / "portfolio.json")
LOG_FILE = str(LOGS_DIR / "trading_bot.log")


INITIAL_CAPITAL = 100000

USE_LIVE_TRADING = True
ENV_NAME = "paper"

MAX_POSITION_SIZE_PCT = 0.25
MAX_POSITION_SIZE_DOLLARS = 25000
RISK_FRACTION = 0.25
MAX_LOSS_PER_TRADE = 250.00

PDT_TIERING_ENABLED = False