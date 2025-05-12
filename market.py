from datetime import datetime
import pytz

def is_market_open():
    # Define the timezone for US Eastern Time
    eastern = pytz.timezone("US/Eastern")
    now = datetime.now(tz=eastern)

    # Check if today is a weekday (0=Monday, 4=Friday)
    if now.weekday() >= 5:  # 5=Saturday, 6=Sunday
        return False

    # Market hours in ET: 9:30 AM to 4:00 PM
    market_open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close_time = now.replace(hour=16, minute=0, second=0, microsecond=0)

    return market_open_time <= now <= market_close_time