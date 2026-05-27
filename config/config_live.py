from .config_base import *
import os

API_KEY = os.getenv("ALPACA_MARKET_API_KEY")
API_SECRET = os.getenv("ALPACA_MARKET_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_MARKET_BASE_URL", "https://api.alpaca.markets")

INITIAL_CAPITAL = 0
PORTFOLIO_PATH = "data/portfolio.json"
LOG_FILE = "logs/trading_bot.log"

DATA_DIR = "data"
LOGS_DIR = "logs"

USE_LIVE_TRADING = True
ENV_NAME = "live"