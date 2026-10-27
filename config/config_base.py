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

# Keep the complete universe monitored while limiting which symbols may open
# new positions in each environment. SELL and risk-exit eligibility is not
# controlled by these lists.
LIVE_ENTRY_SYMBOLS = ["NVDA", "AAPL", "ABBV", "PLTR"]
PAPER_ENTRY_SYMBOLS = list(SYMBOL) + ["SPY"]

BUY_THRESHOLD = 0.55
SELL_THRESHOLD = 0.45
PYRAMID_THRESHOLD = 0.65
PYRAMID_COOLDOWN_MINUTES = 60
MAX_PYRAMID_ADDITIONS = 2
POST_STOP_REBUY_COOLDOWN_MINUTES = 120
POST_STOP_REBUY_BUFFER = 0.10
ORDER_FILL_TIMEOUT_SECONDS = 30
AAPL_BUY_THRESHOLD = 0.60
AAPL_MOMENTUM_INTRADAY_MIN_PROB = 0.58
AAPL_MOMENTUM_MIN_RELATIVE_SESSION = 0.003
AAPL_MOMENTUM_MIN_RELATIVE_60M = 0.005
AAPL_MOMENTUM_MIN_VOLUME_TIME_RATIO = 0.80
AAPL_MOMENTUM_LOW_VOLUME_SIZE_MULTIPLIER = 0.50

# The legacy NVDA momentum artifact classifies above roughly 0.53, while the
# ordinary strategy floor remains deliberately stricter at 0.57. Permit a
# separately attributed initial entry only when live evidence confirms it.
NVDA_GUARDED_MOMENTUM_ENTRY_ENABLED = True
NVDA_MOMENTUM_FINAL_PROB_MIN = 0.55
NVDA_MOMENTUM_MIN_RELATIVE_SESSION = 0.0
NVDA_MOMENTUM_MIN_RELATIVE_60M = 0.0
NVDA_MOMENTUM_MIN_VOLUME_TIME_RATIO = 0.80
NVDA_MOMENTUM_INITIAL_POSITION_EQUITY_PCT = 0.25

USE_ARTIFACT_THRESHOLDS = True
ARTIFACT_THRESHOLD_FALLBACK = 0.55

MODEL_ENTRY_BUFFER = 0.02
MODEL_EXIT_BUFFER = 0.02
# Cost-aware thresholds may relax the legacy entry floor, but keep a wider
# neutral band to limit churn between six-minute decision cycles.
COST_AWARE_BUY_FLOOR = 0.56
COST_AWARE_SELL_CEILING = 0.52
COST_AWARE_MIN_HYSTERESIS = 0.04
MODEL_REBUY_BUFFER = 0.04
MODEL_PYRAMID_BUFFER = 0.05
# A trained, cost-aware artifact may lower the legacy entry floor, but never
# below a neutral class boundary. AAPL retains a slightly higher safety floor.
MODEL_MIN_CLASS_BOUNDARY = 0.50
AAPL_MODEL_MIN_CLASS_BOUNDARY = 0.55
# Do not enter, re-enter, or pyramid a symbol that is down at least 0.5%
# from the previous close. This prevents buying a brief bounce in a weak day.
BUY_SESSION_RETURN_FLOOR = -0.005

# A mean-reversion entry may begin before the one-hour momentum measure crosses
# above zero, but only after shorter-horizon market evidence confirms a turn.
MR_REVERSAL_ENTRY_ENABLED = True
MR_REVERSAL_PROB_BUFFER = 0.05
MR_REVERSAL_MIN_MOMENTUM = -0.0035
MR_REVERSAL_MIN_RELATIVE_ACCEL = 0.0010
MR_REVERSAL_MIN_RELATIVE_60M = 0.0
MR_REVERSAL_MIN_VOLUME_TIME_RATIO = 0.80
MR_REVERSAL_CONFIRM_CYCLES = 2
MR_REVERSAL_INITIAL_POSITION_EQUITY_PCT = 0.125

SPY_USE_ARTIFACT_THRESHOLDS = True
SPY_MODEL_ENTRY_BUFFER = 0.03
SPY_MODEL_EXIT_BUFFER = 0.02

REBUY_THRESHOLD = BUY_THRESHOLD + 0.06
REBUY_COOLDOWN_MINUTES = 30
ALLOW_SAME_DAY_REBUY = True

INTRADAY_WEIGHT = 0.65
MIN_INTRADAY_BARS_FOR_FEATURES = 40
RS_MARGIN = 0.05
USE_MULTICLASS_MODELS = False
# Train a meaningful-move gate followed by conditional direction. Existing
# binary and multiclass artifacts remain loadable for champion comparison.
USE_TWO_STAGE_TARGETS = True

SPY_SYMBOL = "SPY"
PRICE_WEAK_THRESHOLD = -0.01 
WEAK_PROB_THRESHOLD = 0.50
# With the four-symbol core universe, three weak model probabilities meet this
# inclusive threshold and block new core entries.
WEAK_RATIO_THRESHOLD = 0.75
PRICE_WEAK_RATIO_THRESHOLD = 0.75
LAST_HOUR_BUY_BUFFER = 0.04
FINAL_15_MIN_BUY_BUFFER = 0.06
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

# Experimental guarded entry path. Keep disabled unless explicitly enabled in
# the isolated paper worktree with RELATIVE_MOMENTUM_ENTRY_ENABLED=true.
RELATIVE_MOMENTUM_ENTRY_ENABLED = os.getenv(
    "RELATIVE_MOMENTUM_ENTRY_ENABLED", "false"
).strip().lower() in {"1", "true", "yes", "on"}

STOP_LOSS = 0.97
TAKE_PROFIT = None
TRAIL_STOP = 0.985
TRAIL_ACTIVATE = 1.02
RISK_FRACTION = 0.5
# Dynamic per-position risk cap: use the smaller of 0.5% of current account
# equity and 1.5% of the position's average-cost value.
MAX_LOSS_PER_TRADE = None
MAX_LOSS_ACCOUNT_EQUITY_PCT = 0.005
MAX_LOSS_PER_POSITION_PCT = 0.015
PROFIT_TRIGGER_PCT = 0.02  # +2% intraday profit trigger

MAX_POSITION_SIZE_PCT = 0.90
MAX_POSITION_SIZE_DOLLARS = None
# New positions start small enough to leave room for confirmation-based adds.
INITIAL_POSITION_EQUITY_PCT = 0.25
EXCEPTIONAL_INITIAL_POSITION_EQUITY_PCT = 0.25
EXCEPTIONAL_ENTRY_MIN_VOLUME_TIME_RATIO = 0.80
# Every new position starts at no more than 25%. NVDA and AAPL may scale to
# 80% only after the prediction and live confirmation features agree.
CORE_CONFIRMED_POSITION_EQUITY_PCT = 0.80
OTHER_CONFIRMED_POSITION_EQUITY_PCT = 0.50
CORE_CONFIRMED_POSITION_SYMBOLS = ["NVDA", "AAPL"]
CONFIRMED_PYRAMID_MIN_POSITION_RETURN = 0.002
CONFIRMED_PYRAMID_MIN_VOLUME_TIME_RATIO = 0.80

# A model-driven minimum holding period must never delay a confirmed loss exit.
# Measure the loss from the broker average entry and require deteriorating
# model or price evidence before exiting at this tighter threshold.
CONFIRMED_LOSS_EXIT_PCT = -0.005

# Defer an ordinary model SELL while an already-profitable position continues
# to trend upward. Risk exits are evaluated first and always take precedence.
TREND_PROTECTED_SELL_ENABLED = True
TREND_PROTECTED_SELL_MIN_PROBABILITY = 0.50
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
