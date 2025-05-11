import datetime

def is_market_open():
    # Get the current date and time in UTC
    now = datetime.datetime.utcnow()

    # Check if today is a weekday (0: Monday, 4: Friday)
    if now.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return False
    
    # Market hours: 9:30 AM to 4:00 PM (UTC)
    market_open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close_time = now.replace(hour=16, minute=0, second=0, microsecond=0)

    # Check if the current time is between market open and close
    return market_open_time <= now <= market_close_time