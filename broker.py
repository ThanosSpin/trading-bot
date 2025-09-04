# broker.py
import alpaca_trade_api as tradeapi
from config import API_MARKET_KEY, API_MARKET_SECRET, MARKET_BASE_URL

# Single Alpaca API instance shared everywhere
api = tradeapi.REST(API_MARKET_KEY, API_MARKET_SECRET, MARKET_BASE_URL, api_version='v2')