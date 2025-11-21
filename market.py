# market.py
import pytz
from datetime import datetime
import alpaca_trade_api as tradeapi

from config import API_MARKET_KEY, API_MARKET_SECRET, MARKET_BASE_URL

api = tradeapi.REST(API_MARKET_KEY, API_MARKET_SECRET, MARKET_BASE_URL, api_version='v2')

EASTERN = pytz.timezone("US/Eastern")


def _safe_parse_time(ts):
    """
    Alpaca sometimes returns timestamps as:
       - Python datetime
       - ISO string without timezone
       - ISO string with timezone
    This normalizes ANY of these into a timezone-aware UTC datetime.
    """
    if ts is None:
        return None

    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=pytz.UTC)

    try:
        # Generic ISO parser
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=pytz.UTC)
    except Exception:
        return None


def is_market_open():
    """
    Returns True if market is currently open.
    """
    try:
        clock = api.get_clock()

        # Some Alpaca versions return strings, some return datetime objects
        ts_now = _safe_parse_time(getattr(clock, "timestamp", None))
        ts_open = _safe_parse_time(getattr(clock, "next_open", None))
        ts_close = _safe_parse_time(getattr(clock, "next_close", None))

        # If parsing failed → fall back to simple flag
        if ts_now is None or ts_open is None or ts_close is None:
            return bool(clock.is_open)

        now_est = ts_now.astimezone(EASTERN)
        open_est = ts_open.astimezone(EASTERN)
        close_est = ts_close.astimezone(EASTERN)

        return open_est <= now_est <= close_est

    except Exception as e:
        print(f"[WARN] Could not parse Alpaca clock timestamps: {e}")
        return False


def is_trading_day():
    """
    Returns True if today is an Alpaca trading day.
    """
    try:
        today = datetime.now(EASTERN).date()
        cal = api.get_calendar(start=str(today), end=str(today))
        return len(cal) > 0
    except Exception as e:
        print(f"[WARN] Could not check trading day: {e}")
        return False