import datetime, time
from zoneinfo import ZoneInfo

def is_market_open():
    # Get current time in US Eastern Time
    eastern = ZoneInfo("America/New_York")
    now = datetime.now(tz=eastern)

    # Check if it's a weekday (Monday=0 to Friday=4)
    if now.weekday() >= 5:  # Saturday or Sunday
        return False

    # Define market open and close times
    market_open = time(hour=9, minute=30)
    market_close = time(hour=16, minute=0)

    # Check if current time is within market hours
    return market_open <= now.time() <= market_close