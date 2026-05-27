# broker.py
import os
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

# Load .env variables
load_dotenv()

# --------------------------
# API Keys
# --------------------------
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL")

API_MARKET_KEY = os.getenv("ALPACA_MARKET_API_KEY")
API_MARKET_SECRET = os.getenv("ALPACA_MARKET_SECRET_KEY")
MARKET_BASE_URL = os.getenv("ALPACA_MARKET_BASE_URL")

# --------------------------
# Validate Keys
# --------------------------
if not (API_KEY and API_SECRET and BASE_URL):
    raise ValueError("Alpaca general API key/secret/base_url not set in environment")

if not (API_MARKET_KEY and API_MARKET_SECRET and MARKET_BASE_URL):
    raise ValueError("Alpaca market API key/secret/base_url not set in environment")

# --------------------------
# REST Clients
# --------------------------
api_general = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")
api_market = tradeapi.REST(API_MARKET_KEY, API_MARKET_SECRET, MARKET_BASE_URL, api_version="v2")

def get_trading_api():
    bot_env = os.getenv("BOT_ENV", "live").lower()
    return api_general if bot_env == "paper" else api_market

def get_active_env():
    return os.getenv("BOT_ENV", "live").lower()

# --------------------------
# Optional Test When Run Directly
# --------------------------
def test_connection():
    """Manual test: checks both clients and the active selected client."""
    ok = True

    try:
        general_account = api_general.get_account()
        print(f"✅ GENERAL connection OK | equity=${general_account.equity} | url={BASE_URL}")
    except Exception as e:
        ok = False
        print(f"❌ GENERAL connection failed: {e}")

    try:
        market_account = api_market.get_account()
        print(f"✅ MARKET connection OK | equity=${market_account.equity} | url={MARKET_BASE_URL}")
    except Exception as e:
        ok = False
        print(f"❌ MARKET connection failed: {e}")

    try:
        active_env = get_active_env()
        active_api = get_trading_api()
        active_account = active_api.get_account()
        print(f"🎯 ACTIVE client = {active_env.upper()} | equity=${active_account.equity}")
    except Exception as e:
        ok = False
        print(f"❌ ACTIVE client check failed: {e}")

    return ok

if __name__ == "__main__":
    raise SystemExit(0 if test_connection() else 1)
