from .config_base import *
import os

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
API_MARKET_KEY = os.getenv("ALPACA_MARKET_API_KEY")
API_MARKET_SECRET = os.getenv("ALPACA_MARKET_SECRET_KEY")

BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
MARKET_BASE_URL = os.getenv("ALPACA_MARKET_BASE_URL")

INITIAL_CAPITAL = 100000
PORTFOLIO_PATH = "data_paper/portfolio.json"
LOG_FILE = "logs_paper/trading_bot.log"

DATA_DIR = "data_paper"
LOGS_DIR = "logs_paper"

USE_LIVE_TRADING = False
ENV_NAME = "paper"

MAX_POSITION_SIZE_PCT = 0.25
MAX_POSITION_SIZE_DOLLARS = 25000
RISK_FRACTION = 0.25
MAX_LOSS_PER_TRADE = 250.00

PDT_TIERING_ENABLED = False