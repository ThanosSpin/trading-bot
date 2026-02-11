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
SYMBOL = ["NVDA", "AAPL", "ABBV", "PLTR"]  # symbol
INITIAL_CAPITAL = 0

# --- Trading thresholds ---
BUY_THRESHOLD = 0.55    # require strong confidence to buy
SELL_THRESHOLD = 0.47   # require strong confidence to sell


# --------------------
# Model blending weights
# --------------------
INTRADAY_WEIGHT = 0.65   # default intraday dominance
MIN_INTRADAY_BARS_FOR_FEATURES = 25
RS_MARGIN = 0.05

# Model training configuration
USE_MULTICLASS_MODELS = False  # Train 5-class models instead of binary

# =========================
# SPY fallback configuration
# =========================

SPY_SYMBOL = "SPY"

# Weakness definition for individual stocks:
WEAK_PROB_THRESHOLD = 0.45

# Market is weak if this fraction of your stock symbols are weak:
WEAK_RATIO_THRESHOLD = 0.60

# Only trade SPY if SPY confirms strength:
SPY_ENTRY_THRESHOLD = 0.70
SPY_EXIT_THRESHOLD = 0.50
SPY_RISK_FRACTION = 1.0   # 100% of available cash for SPY trades

# If True: only trade SPY when you took NO stock trades this cycle.
SPY_MUTUAL_EXCLUSIVE = True

# =========================
# Intraday regime detection (15m bars)
# =========================

# Default thresholds (most symbols)
INTRADAY_MOM_TRIG = 0.0030    # +0.30% over ~1h (4 x 15m bars)
INTRADAY_VOL_TRIG = 0.0030    # 0.30% per-bar volatility (15m)

# Symbol-specific overrides (optional)
INTRADAY_REGIME_OVERRIDES = {
    "NVDA": {
        "mom_trig": 0.0035,   # NVDA needs stronger push to be "momentum"
        "vol_trig": 0.0032,
    },
    "PLTR": {
        "mom_trig": 999.0,      # ✅ Effectively disable momentum regime
        "vol_trig": 999.0,      # Force PLTR to always use mean-reversion or legacy model
        "disable_adaptive": True,  # Skip adaptive calculation
    },
    "SPY": {
        "mom_trig": 0.0030,  # Higher threshold
        "vol_trig": 0.0020,
    },
    # You can add more later:
    # "AAPL": {"mom_trig": 0.0022, "vol_trig": 0.0028},
}

# Hysteresis: once in MOM regime, keep it until it cools down
MOM_HOLD = 0.003   # 0.30%
VOL_HOLD = 0.0035  # 0.35%

TRAIN_SYMBOLS = ["NVDA", "AAPL", "SPY", "ABBV", "PLTR"]  # used by retrain_model.py

# Feature selection
SHAP_TOP_N = 40  # Number of features to keep after SHAP selection

# --- Risk management ---
STOP_LOSS = 0.95        # sell if price falls 5% below last buy price
TAKE_PROFIT = None      # sell if price rises 5% above last buy price
TRAIL_STOP = 0.96        # trailing stop vs max_price since entry (e.g., 0.97 = 3% trail)
TRAIL_ACTIVATE = 1.05   # ✅ activate trailing only after +5% profit
RISK_FRACTION = 0.5     # default: invest or sell 50%

# NEW: Hard limits (always enforced)
MAX_POSITION_SIZE_PCT = 0.90  # Never invest >90% in single symbol
MAX_POSITION_SIZE_DOLLARS = None  # Optional: Set to dollar amount like 50000

MIN_RETURN_THRESHOLD = 0.002  # 0.2% - must beat transaction costs

# DIP-BUY OVERRIDE: 100% capital if high conviction + dip
DIP_BUY_ENABLED = True
DIP_BUY_MIN_PROB = 0.75
DIP_BUY_THRESHOLD = 0.015

# PDT-aware stop tiers (only relevant if equity < 25k and dt_api >= 3)
PDT_TIERING_ENABLED = True

# If position opened today, do NOT stop out unless loss exceeds this
PDT_SAMEDAY_STOP_BLOCK = 0.020   # 2.0% loss blocks normal stop sell

# Emergency override: if loss exceeds this, allow selling opened-today shares
# (consumes a day trade; set None to disable emergency exits entirely)
PDT_EMERGENCY_STOP = 0.060       # 6.0% loss triggers emergency exit

# Only allow at most N emergency day-trade exits per day
PDT_EMERGENCY_MAX_PER_DAY = 1

# ============================================================
# PRE-MARKET SCANNER SETTINGS
# ============================================================
PRE_MARKET_ENABLED = True
PRE_MARKET_MIN_PROB = 0.55          # Only queue orders with 55%+ conviction
PRE_MARKET_MAX_ALLOCATION = 1.0     # Use 100% of available cash
PRE_MARKET_LIMIT_BUFFER = 0.002     # Place limit 0.2% above last price for quick fill
PRE_MARKET_SYMBOLS = SYMBOL         # Use same symbols as main bot
PRE_MARKET_ALLOW_SHORT_SELLING = True  # Enable short selling
PRE_MARKET_MIN_SELL_PROB = 0.47        # Sell if prob < 45%
PRE_MARKET_SHORT_ALLOCATION = 0.10     # Max 10% per short position


# Time settings
PRE_MARKET_SCAN_HOUR = 9            # Run at 9:00 AM (1.5h before market open)

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