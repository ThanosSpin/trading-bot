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
api_general = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
api_market = tradeapi.REST(API_MARKET_KEY, API_MARKET_SECRET, MARKET_BASE_URL, api_version='v2')

# --------------------------
# Optional Test When Run Directly
# --------------------------
def test_connection():
    """Manual test: returns True if Market API reachable."""
    try:
        account = api_market.get_account()
        print(f"✅ Alpaca connection OK! Equity: ${account.equity}")
        return True
    except Exception as e:
        print(f"❌ Alpaca connection failed: {e}")
        return False


if __name__ == "__main__":
    test_connection()