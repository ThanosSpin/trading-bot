from .config_base import *
import os

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
API_MARKET_KEY = os.getenv("ALPACA_MARKET_API_KEY")
API_MARKET_SECRET = os.getenv("ALPACA_MARKET_SECRET_KEY")

BASE_URL = os.getenv("ALPACA_BASE_URL")
MARKET_BASE_URL = os.getenv("ALPACA_MARKET_BASE_URL")

INITIAL_CAPITAL = 0
PORTFOLIO_PATH = "data/portfolio.json"
LOG_FILE = "logs/trading_bot.log"

USE_LIVE_TRADING = True
ENV_NAME = "live"