# broker.py
import os

from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

load_dotenv()

# ============================================================
# PAPER ACCOUNT
# ============================================================

PAPER_API_KEY = os.getenv("ALPACA_API_KEY")
PAPER_API_SECRET = os.getenv("ALPACA_SECRET_KEY")
PAPER_BASE_URL = os.getenv("ALPACA_BASE_URL")


# ============================================================
# LIVE ACCOUNT #1
# ============================================================

LIVE_API_KEY = os.getenv("ALPACA_MARKET_API_KEY")
LIVE_API_SECRET = os.getenv("ALPACA_MARKET_SECRET_KEY")
LIVE_BASE_URL = os.getenv("ALPACA_MARKET_BASE_URL")


# ============================================================
# LIVE ACCOUNT #2
# ============================================================

LIVE2_API_KEY = os.getenv("ALPACA_LIVE2_MARKET_API_KEY")
LIVE2_API_SECRET = os.getenv("ALPACA_LIVE2_MARKET_SECRET_KEY")
LIVE2_BASE_URL = os.getenv("ALPACA_LIVE2_MARKET_BASE_URL")

# ============================================================
# Expected Alpaca account IDs
# ============================================================

PAPER_ACCOUNT_ID = os.getenv("ALPACA_PAPER_ACCOUNT_ID")
LIVE_ACCOUNT_ID = os.getenv("ALPACA_LIVE_ACCOUNT_ID")
LIVE2_ACCOUNT_ID = os.getenv("ALPACA_LIVE2_ACCOUNT_ID")

# ============================================================
# Validate credentials
# ============================================================

if not (PAPER_API_KEY and PAPER_API_SECRET and PAPER_BASE_URL):
    raise ValueError(
        "Paper Alpaca API key/secret/base_url not set in environment"
    )

if not (LIVE_API_KEY and LIVE_API_SECRET and LIVE_BASE_URL):
    raise ValueError(
        "Live Alpaca API key/secret/base_url not set in environment"
    )

if not (LIVE2_API_KEY and LIVE2_API_SECRET and LIVE2_BASE_URL):
    raise ValueError(
        "Live2 Alpaca API key/secret/base_url not set in environment"
    )


# ============================================================
# Create API clients
# ============================================================

api_paper = tradeapi.REST(
    PAPER_API_KEY,
    PAPER_API_SECRET,
    PAPER_BASE_URL,
    api_version="v2",
)

api_live = tradeapi.REST(
    LIVE_API_KEY,
    LIVE_API_SECRET,
    LIVE_BASE_URL,
    api_version="v2",
)

api_live2 = tradeapi.REST(
    LIVE2_API_KEY,
    LIVE2_API_SECRET,
    LIVE2_BASE_URL,
    api_version="v2",
)


# ============================================================
# Active environment
# ============================================================

def get_active_env():
    """
    Returns:
        paper
        live
        live2
    """
    env = os.getenv("BOT_ENV", "live").strip().lower()

    if env not in ("paper", "live", "live2"):
        raise ValueError(
            f"Invalid BOT_ENV='{env}'. "
            "Expected: paper, live, or live2."
        )

    return env


# ============================================================
# Active Alpaca API
# ============================================================

def get_trading_api():
    env = get_active_env()

    if env == "paper":
        return api_paper

    if env == "live":
        return api_live

    if env == "live2":
        return api_live2

    raise RuntimeError(f"Unsupported trading environment: {env}")


# ============================================================
# Active account information
# ============================================================

def get_active_account():
    api = get_trading_api()
    return api.get_account()

def verify_active_account():
    """
    Verify that BOT_ENV is connected to the expected Alpaca account.

    Fails closed if the expected account ID is missing or mismatched.
    """
    env = get_active_env()
    api = get_trading_api()
    account = api.get_account()

    actual_id = str(account.id)

    expected_ids = {
        "paper": PAPER_ACCOUNT_ID,
        "live": LIVE_ACCOUNT_ID,
        "live2": LIVE2_ACCOUNT_ID,
    }

    expected_id = expected_ids[env]

    if not expected_id:
        raise RuntimeError(
            f"Missing expected Alpaca account ID for BOT_ENV={env}. "
            f"Set the appropriate account ID in .env."
        )

    if actual_id != str(expected_id):
        raise RuntimeError(
            "ACCOUNT SAFETY CHECK FAILED: "
            f"BOT_ENV={env} is connected to account {actual_id}, "
            f"but expected account {expected_id}."
        )

    return account

# ============================================================
# Connection test
# ============================================================

def test_connection():
    env = get_active_env()
    api = get_trading_api()

    account = api.get_account()

    print("=" * 70)
    print(f"TRADING ENVIRONMENT : {env.upper()}")
    print(f"ALPACA ACCOUNT      : {account.id}")
    print(f"STATUS              : {account.status}")
    print(f"EQUITY              : ${float(account.equity):,.2f}")
    print(f"CASH                : ${float(account.cash):,.2f}")
    print("=" * 70)

    return account
