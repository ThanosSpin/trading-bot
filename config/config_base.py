import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# Raw Alpaca credentials
# ============================================================

# Paper account
PAPER_API_KEY = os.getenv("ALPACA_API_KEY")
PAPER_API_SECRET = os.getenv("ALPACA_SECRET_KEY")
PAPER_BASE_URL = os.getenv(
    "ALPACA_BASE_URL",
    "https://paper-api.alpaca.markets",
)

# Primary live account
LIVE_API_KEY = os.getenv("ALPACA_MARKET_API_KEY")
LIVE_API_SECRET = os.getenv("ALPACA_MARKET_SECRET_KEY")
LIVE_BASE_URL = os.getenv(
    "ALPACA_MARKET_BASE_URL",
    "https://api.alpaca.markets",
)

# Secondary live account
LIVE2_API_KEY = os.getenv("ALPACA_LIVE2_MARKET_API_KEY")
LIVE2_API_SECRET = os.getenv("ALPACA_LIVE2_MARKET_SECRET_KEY")
LIVE2_BASE_URL = os.getenv(
    "ALPACA_LIVE2_MARKET_BASE_URL",
    "https://api.alpaca.markets",
)

# ============================================================
# Email
# ============================================================

EMAIL_SENDER = os.getenv("EMAIL")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")


# ============================================================
# Trading Strategy
# ============================================================

SYMBOL = ["NVDA", "AAPL", "ABBV", "PLTR"]
PAPER_TRADE_SYMBOLS = []
PAPER_TRADE_NOTES = {}

BUY_THRESHOLD = 0.55
SELL_THRESHOLD = 0.45
PYRAMID_THRESHOLD = 0.65
AAPL_BUY_THRESHOLD = 0.60

USE_ARTIFACT_THRESHOLDS = True
ARTIFACT_THRESHOLD_FALLBACK = 0.55

MODEL_ENTRY_BUFFER = 0.02
MODEL_EXIT_BUFFER = 0.02
MODEL_REBUY_BUFFER = 0.04
MODEL_PYRAMID_BUFFER = 0.08

SPY_USE_ARTIFACT_THRESHOLDS = True
SPY_MODEL_ENTRY_BUFFER = 0.03
SPY_MODEL_EXIT_BUFFER = 0.02

REBUY_THRESHOLD = BUY_THRESHOLD + 0.06
REBUY_COOLDOWN_MINUTES = 30
ALLOW_SAME_DAY_REBUY = True

INTRADAY_WEIGHT = 0.65
MIN_INTRADAY_BARS_FOR_FEATURES = 25
RS_MARGIN = 0.05
USE_MULTICLASS_MODELS = False
# Train a meaningful-move gate followed by conditional direction. Existing
# binary and multiclass artifacts remain loadable for champion comparison.
USE_TWO_STAGE_TARGETS = True

SPY_SYMBOL = "SPY"
PRICE_WEAK_THRESHOLD = -0.01 
WEAK_PROB_THRESHOLD = 0.50
WEAK_RATIO_THRESHOLD = 0.50
SPY_ENTRY_THRESHOLD = 0.70
SPY_EXIT_THRESHOLD = 0.50
SPY_RISK_FRACTION = 1.0
SPY_MUTUAL_EXCLUSIVE = True

INTRADAY_MOM_TRIG = 0.0030
INTRADAY_VOL_TRIG = 0.0030

INTRADAY_REGIME_OVERRIDES = {
    "NVDA": {"mom_trig": 0.0035, "vol_trig": 0.0032},
    "PLTR": {},
    "SPY": {"mom_trig": 0.0030, "vol_trig": 0.0020},
}

MOM_HOLD = 0.003
VOL_HOLD = 0.0035

TRAIN_SYMBOLS = ["NVDA", "AAPL", "SPY", "ABBV", "PLTR"]
SHAP_TOP_N = 40

# Leakage-aware evaluation and promotion assumptions.
MODEL_EVAL_WALK_FORWARD_FOLDS = 4
MODEL_EVAL_GAP_BARS = 1
MODEL_EVAL_TRANSACTION_COST_BPS = 10.0
# Keep training responsive on the small VM. Models are trained sequentially,
# so one worker avoids CPU saturation and calibration process duplication.
MODEL_TRAIN_N_JOBS = 1

STOP_LOSS = 0.97
TAKE_PROFIT = None
TRAIL_STOP = 0.985
TRAIL_ACTIVATE = 1.02
RISK_FRACTION = 0.5
MAX_LOSS_PER_TRADE = 10.00
PROFIT_TRIGGER_PCT = 0.02  # +2% intraday profit trigger

MAX_POSITION_SIZE_PCT = 0.90
MAX_POSITION_SIZE_DOLLARS = None
MIN_RETURN_THRESHOLD = 0.002

DIP_BUY_ENABLED = True
DIP_BUY_MIN_PROB = 0.75
DIP_BUY_THRESHOLD = 0.015

MARGIN_TIERING_ENABLED = True
MARGIN_SAMEDAY_STOP_BLOCK = 0.020
MARGIN_EMERGENCY_STOP = None
MARGIN_EMERGENCY_MAX_PER_DAY = 1
MARGIN_EMERGENCY_PROB_THRESH = 0.40

ROTATION_MIN_EDGE = 0.03

PRE_MARKET_ENABLED = True
PRE_MARKET_MIN_PROB = 0.55
PRE_MARKET_MAX_ALLOCATION = 1.0
PRE_MARKET_LIMIT_BUFFER = 0.002
PRE_MARKET_SYMBOLS = SYMBOL
PRE_MARKET_ALLOW_SHORT_SELLING = True
PRE_MARKET_MIN_SELL_PROB = 0.30
PRE_MARKET_SHORT_ALLOCATION = 0.10

PRE_MARKET_SCAN_HOUR = 9
LIMIT_BUFFER_PCT = 0.01

# ============================================================
# Performance Fee
# ============================================================

# Performance fee applied to profit remaining after the resource fee
# and the 5% hurdle. The hurdle does not reduce the fee rate.
RESOURCE_FEE_PCT = 2.0
HURDLE_PCT = 5.0
PERFORMANCE_FEE_PCT = 20.0

TIMEZONE = "US/Eastern"
USE_LIVE_TRADING = True

# ============================================================
# Data Directory
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
MODEL_DIR = BASE_DIR / "models"

def get_model_path(symbol):
    os.makedirs(MODEL_DIR, exist_ok=True)
    return os.path.join(MODEL_DIR, f"model_{symbol}.pkl")
