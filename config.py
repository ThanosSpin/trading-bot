import os
from dotenv import load_dotenv

load_dotenv()  # Load from .env file

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
API_MARKET_KEY = os.getenv("ALPACA_MARKET_API_KEY")
API_MARKET_SECRET = os.getenv("ALPACA_MARKET_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL")
MARKET_BASE_URL = os.getenv("ALPACA_MARKET_BASE_URL") # For live trading
EMAIL_SENDER = os.getenv("EMAIL")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")  # Use App Password
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")

SYMBOL = 'NVDA'
INITIAL_CAPITAL = 0
THRESHOLD = 0.05

# Paths
MODEL_PATH = "models/model.pkl"
PORTFOLIO_PATH = "data/portfolio.json"
LOG_FILE = "logs/trading_bot.log"

# Other configs
USE_LIVE_TRADING = True  # Switch to True to go live
TIMEZONE = "US/Eastern"