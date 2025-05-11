import os
from dotenv import load_dotenv

load_dotenv()  # Load from .env file

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL")

SYMBOL = 'NVDA'
INITIAL_CAPITAL = 0
THRESHOLD = 0.05

# Paths
MODEL_PATH = "models/model.pkl"
PORTFOLIO_PATH = "data/portfolio.json"
LOG_FILE = "logs/trading_bot.log"

# Other configs
USE_LIVE_TRADING = False  # Switch to True to go live
TIMEZONE = "US/Eastern"