from datetime import datetime
import alpaca_trade_api as tradeapi
from config import API_MARKET_KEY, API_MARKET_SECRET, MARKET_BASE_URL
import pytz

api = tradeapi.REST(API_MARKET_KEY, API_MARKET_SECRET, MARKET_BASE_URL)

def is_market_open():
    try:
        clock = api.get_clock()
        return clock.is_open
    except Exception as e:
        print(f"[ERROR] Could not retrieve market clock: {e}")
        return False

def is_trading_day():
    try:
        eastern = pytz.timezone("US/Eastern")
        today = datetime.now(tz=eastern).date()

        calendar = api.get_calendar(start=str(today), end=str(today))
        return len(calendar) > 0
    except Exception as e:
        print(f"[ERROR] Could not retrieve market calendar: {e}")
        return False