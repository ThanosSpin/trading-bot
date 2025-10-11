# broker.py
import os
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

# Load .env variables
load_dotenv()

API_KEY = os.getenv("ALPACA_MARKET_API_KEY")
API_SECRET = os.getenv("ALPACA_MARKET_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_MARKET_BASE_URL")  # e.g., paper trading: https://paper-api.alpaca.markets

if not API_KEY or not API_SECRET or not BASE_URL:
    raise ValueError("Alpaca API key/secret/base_url not set in environment")

# Create single shared API instance
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# Test connection
try:
    account = api.get_account()
    print(f"✅ Connected to Alpaca! Account equity: ${account.equity}")
except Exception as e:
    print(f"❌ Failed to connect to Alpaca API: {e}")
    raise