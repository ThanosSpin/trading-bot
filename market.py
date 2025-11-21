# market.py
import pytz
from datetime import datetime, time
import pandas_market_calendars as mcal
import alpaca_trade_api as tradeapi
from alpaca_trade_api import REST
    
from config import (
    TIMEZONE,
    API_MARKET_KEY,
    API_MARKET_SECRET,
    MARKET_BASE_URL,
)

# Optional Alpaca clock (used only when valid)
alpaca = tradeapi.REST(API_MARKET_KEY, API_MARKET_SECRET, MARKET_BASE_URL, api_version="v2")

# Timezones
LOCAL_TZ = pytz.timezone(TIMEZONE)
NY_TZ = pytz.timezone("America/New_York")

# NYSE calendar from pandas_market_calendars
nyse = mcal.get_calendar("NYSE")


def is_trading_day(date=None):
    """Use NYSE calendar to determine if market is open TODAY."""
    if date is None:
        date = datetime.now(NY_TZ).date()

    schedule = nyse.schedule(start_date=date, end_date=date)

    return not schedule.empty


def is_market_open():
    """
    Accurate, Alpaca-independent market open detection:
      1. Uses NYSE schedule (holidays, half-days, early close)
      2. Uses local wall-clock time
      3. Only uses Alpaca clock when valid & consistent
    """

    now_local = datetime.now(LOCAL_TZ)
    now_ny = now_local.astimezone(NY_TZ)

    # Step 1 — Check NYSE trading day
    if not is_trading_day(now_ny.date()):
        print("[MARKET] Today is NOT a NYSE trading day.")
        return False

    # Step 2 — Get today's schedule (open/close)
    schedule = nyse.schedule(start_date=now_ny.date(), end_date=now_ny.date())

    if schedule.empty:
        print("[MARKET] NYSE schedule empty — not a trading day.")
        return False

    market_open = schedule.iloc[0]["market_open"].astimezone(NY_TZ)
    market_close = schedule.iloc[0]["market_close"].astimezone(NY_TZ)

    # Step 3 — Check if within scheduled hours
    if market_open <= now_ny <= market_close:
        within_hours = True
    else:
        within_hours = False

    # Step 4 — (Optional) Cross-check Alpaca clock (only if valid)
    try:
        clock = alpaca.get_clock()

        if isinstance(clock.is_open, bool):
            alpaca_open = bool(clock.is_open)

            # If Alpaca disagrees BUT their timestamps are broken → ignore them
            if alpaca_open != within_hours:
                print(
                    f"[WARN] Alpaca clock mismatch: alpaca={alpaca_open} schedule={within_hours} — using schedule."
                )

            return within_hours

    except Exception:
        pass  # ignore Alpaca failures entirely

    # Final fallback: schedule only
    return within_hours

def debug_market(return_dict=False):
    """
    Returns either:
      - printed diagnostic output (default)
      - OR a dictionary for dashboards (return_dict=True)
    """
    ny_tz = pytz.timezone("America/New_York")

    # Time objects for the market
    market_open_t = time(9, 30)
    market_close_t = time(16, 0)

    # Local → NY time
    local_now = datetime.now()
    ny_now = local_now.astimezone(ny_tz)

    # Trading day check
    is_trading_day_flag = ny_now.weekday() < 5

    # Within NYSE hours
    within_hours = (market_open_t <= ny_now.time() <= market_close_t)

    # Alpaca clock check
    alpaca_is_open = None
    try:
        api = REST(API_MARKET_KEY, API_MARKET_SECRET, MARKET_BASE_URL, api_version="v2")
        clock = api.get_clock()
        alpaca_is_open = clock.is_open
    except Exception:
        alpaca_is_open = None

    # Final decision logic
    if is_trading_day_flag and within_hours:
        decision = True
    else:
        decision = False

    # Package as dict for dashboard use
    info = {
        "local_time": local_now.strftime("%Y-%m-%d %H:%M:%S"),
        "ny_time": ny_now.strftime("%Y-%m-%d %H:%M:%S"),
        "market_open": market_open_t.strftime("%H:%M"),
        "market_close": market_close_t.strftime("%H:%M"),
        "is_trading_day": is_trading_day_flag,
        "within_hours": within_hours,
        "alpaca_is_open": alpaca_is_open,
        "decision": decision,
    }

    if return_dict:
        return info

    # Fallback text output
    print("\n========== MARKET DEBUG ==========")
    for k, v in info.items():
        print(f"{k}: {v}")
    print("=================================\n")