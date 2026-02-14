from datetime import datetime, timedelta
from config import API_MARKET_KEY, API_MARKET_SECRET, MARKET_BASE_URL
import alpaca_trade_api as tradeapi

api = tradeapi.REST(API_MARKET_KEY, API_MARKET_SECRET, MARKET_BASE_URL)

try:
    account = api.get_account()
    print(f"[OK] Connected to Alpaca. Account status: {account.status}")
except Exception as e:
    print(f"[ERROR] Could not connect to Alpaca API: {e}")

# Calculate date range
end = datetime.now()
start = end - timedelta(days=5)

# Format dates to 'YYYY-MM-DD'
start_str = start.strftime('%Y-%m-%d')
end_str = end.strftime('%Y-%m-%d')

try:
    bars = api.get_bars("NVDA", "1Day", start=start_str, end=end_str).df
    print(bars.tail())
except Exception as e:
    print(f"[ERROR] Could not fetch NVDA data: {e}")