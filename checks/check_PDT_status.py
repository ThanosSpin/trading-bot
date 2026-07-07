# check_pdt_status.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
import pytz
from collections import defaultdict
from config.config import API_MARKET_KEY, API_MARKET_SECRET, MARKET_BASE_URL
from trader import estimate_daytrade_count


def check_pdt_status():
    api = tradeapi.REST(API_MARKET_KEY, API_MARKET_SECRET, MARKET_BASE_URL, api_version='v2')
    acc = api.get_account()

    print("\n🔍 Alpaca PDT Status Check")
    print("-" * 40)
    equity = float(acc.equity or 0.0)
    bp = float(acc.buying_power or 0.0)
    print(f"Equity: ${equity:.2f}")
    print(f"Buying Power: ${bp:.2f}")

    # PDT-related fields may be deprecated or missing
    pattern_dt = getattr(acc, "pattern_day_trader", None)
    daytrade_count = getattr(acc, "daytrade_count", None)

    print(f"Pattern Day Trader: {pattern_dt if pattern_dt is not None else 'N/A'}")
    print(f"API Reported Day Trades (5d): {daytrade_count if daytrade_count is not None else 'N/A'}")

    # Your heuristic estimator can still run
    est = estimate_daytrade_count(api)
    print(f"Estimated Day Trades (5d): {est}")

    if equity < 25000:
        dt_api = daytrade_count if isinstance(daytrade_count, (int, float)) else None

        if (dt_api is not None and dt_api >= 4) or est >= 4:
            print("\n⚠️ PDT risk: One more round-trip trade could trigger restriction!")
        else:
            print("\n✅ Safe: You still have at least one day-trade slot available.")
    else:
        print("\n💰 Equity > $25k — PDT rules do not apply.")
    
    print(f"API Reported Day Trades (5d): {daytrade_count if daytrade_count is not None else 'N/A'}  ✅ (authoritative if available)")
    print(f"Estimated Day Trades (5d): {est}  ⚠️ (heuristic, may differ)")

if __name__ == "__main__":
    check_pdt_status()