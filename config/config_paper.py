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

# Shared equity-based sizing now caps initial entries at 25% and confirmed
# pyramids at their total-position ceiling. Keep these legacy per-order limits
# out of the way so paper exercises the same scaling policy as live.
MAX_POSITION_SIZE_PCT = 0.90
MAX_POSITION_SIZE_DOLLARS = None
RISK_FRACTION = 0.25
# Match the live risk policy during paper validation.
MAX_LOSS_PER_TRADE = None
MAX_LOSS_ACCOUNT_EQUITY_PCT = 0.005
MAX_LOSS_PER_POSITION_PCT = 0.015

PDT_TIERING_ENABLED = False

# Validate earnings opening continuation separately in the paper account.
EARNINGS_OPENING_CONTINUATION_ENABLED = True
