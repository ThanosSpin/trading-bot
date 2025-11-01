# test_alpaca_connection.py
import os
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

# Load environment variables from .env
load_dotenv()

API_KEY = os.getenv("ALPACA_MARKET_API_KEY")
API_SECRET = os.getenv("ALPACA_MARKET_SECRET_KEY")
API_BASE_URL = os.getenv("ALPACA_MARKET_BASE_URL")

# Quick validation
if not API_KEY or not API_SECRET or not API_BASE_URL:
    raise ValueError("Alpaca API key/secret/base_url not set in environment")

print("✅ Alpaca environment variables loaded successfully!")

# Initialize API
api = tradeapi.REST(API_KEY, API_SECRET, API_BASE_URL, api_version='v2')

# Fetch account info
try:
    account = api.get_account()
    print("✅ Connected to Alpaca API successfully!")
    print(f"Account status: {account.status}, Equity: ${account.equity}")
except Exception as e:
    print(f"❌ Failed to connect to Alpaca API: {e}")