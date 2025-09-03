import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API keys
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
API_MARKET_KEY = os.getenv("ALPACA_MARKET_API_KEY")
API_MARKET_SECRET = os.getenv("ALPACA_MARKET_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL")
MARKET_BASE_URL = os.getenv("ALPACA_MARKET_BASE_URL")  # For live trading

# Email notifications
EMAIL_SENDER = os.getenv("EMAIL")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")  # Use App Password
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")

# Trading config
SYMBOL = "NVDA"  # symbol
INITIAL_CAPITAL = 0
THRESHOLD = 0.05

# Model path management
MODEL_DIR = "models"
def get_model_path(symbol):
    """Return the model path for a given symbol."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    return os.path.join(MODEL_DIR, f"model_{symbol}.pkl")

# Other paths
PORTFOLIO_PATH = "data/portfolio.json"
LOG_FILE = "logs/trading_bot.log"

# Other configs
USE_LIVE_TRADING = False  # Switch to True to go live
TIMEZONE = "US/Eastern"