# broker.py
import os
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

# Load .env variables
load_dotenv()

# --------------------------
# API Keys
# --------------------------
API_KEY = os.getenv("ALPACA_API_KEY")                # General account API key
API_SECRET = os.getenv("ALPACA_SECRET_KEY")          # General account secret
BASE_URL = os.getenv("ALPACA_BASE_URL")             # e.g., https://paper-api.alpaca.markets

API_MARKET_KEY = os.getenv("ALPACA_MARKET_API_KEY")       # Market/trading key
API_MARKET_SECRET = os.getenv("ALPACA_MARKET_SECRET_KEY")
MARKET_BASE_URL = os.getenv("ALPACA_MARKET_BASE_URL")     # Live trading base URL

# --------------------------
# Validate keys
# --------------------------
if not (API_KEY and API_SECRET and BASE_URL):
    raise ValueError("Alpaca general API key/secret/base_url not set in environment")

if not (API_MARKET_KEY and API_MARKET_SECRET and MARKET_BASE_URL):
    raise ValueError("Alpaca market/trading API key/secret/base_url not set in environment")

# --------------------------
# Create REST instances
# --------------------------
# General account API (can be used for documents, etc.)
api_general = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# Market/trading API (for orders, portfolio, etc.)
api_market = tradeapi.REST(API_MARKET_KEY, API_MARKET_SECRET, MARKET_BASE_URL, api_version='v2')

# --------------------------
# Test connections
# --------------------------
try:
    account = api_general.get_account()
    print(f"✅ Connected to Alpaca simulating account! Equity: ${account.equity}")
except Exception as e:
    print(f"❌ Failed to connect to Alpaca general API: {e}")
    raise

try:
    market_account = api_market.get_account()
    print(f"✅ Connected to Alpaca trading account! Equity: ${market_account.equity}")
except Exception as e:
    print(f"❌ Failed to connect to Alpaca market API: {e}")
    raise