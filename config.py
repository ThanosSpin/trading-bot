import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base data folder
DATA_DIR = "data"

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
SYMBOL = ["NVDA", "AAPL"]  # symbol
INITIAL_CAPITAL = 0

# --------------------
# Model blending weights
# --------------------
INTRADAY_WEIGHT = 0.65   # default intraday dominance

# --- Trading thresholds ---
BUY_THRESHOLD = 0.65    # require strong confidence to buy
SELL_THRESHOLD = 0.35   # require strong confidence to sell

# --- Risk management ---
STOP_LOSS = 0.95        # sell if price falls 5% below last buy price
TAKE_PROFIT = 1.1      # sell if price rises 5% above last buy price
RISK_FRACTION = 0.5     # default: invest or sell 50%

# Model path management
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
def get_model_path(symbol):
    """Return the model path for a given symbol."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    return os.path.join(MODEL_DIR, f"model_{symbol}.pkl")

# Other paths
PORTFOLIO_PATH = "data/portfolio.json"
LOG_FILE = "logs/trading_bot.log"

# Other configs
USE_LIVE_TRADING = True  # Switch to True to go live
TIMEZONE = "US/Eastern"