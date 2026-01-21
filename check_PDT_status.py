# check_pdt_status.py
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
import pytz
from collections import defaultdict
from config import API_MARKET_KEY, API_MARKET_SECRET, MARKET_BASE_URL
from trader import estimate_daytrade_count


def check_pdt_status():
    api = tradeapi.REST(API_MARKET_KEY, API_MARKET_SECRET, MARKET_BASE_URL, api_version='v2')
    acc = api.get_account()

    print("\n🔍 Alpaca PDT Status Check")
    print("-" * 40)
    print(f"Equity: ${float(acc.equity):.2f}")
    print(f"Buying Power: ${float(acc.buying_power):.2f}")
    print(f"Pattern Day Trader: {acc.pattern_day_trader}")
    print(f"API Reported Day Trades (5d): {acc.daytrade_count}")
    est = estimate_daytrade_count(api)
    print(f"Estimated Day Trades (5d): {est}")

    if float(acc.equity) < 25000:
        if acc.daytrade_count >= 4 or est >= 4:
            print("\n⚠️ PDT risk: One more round-trip trade could trigger restriction!")
        else:
            print("\n✅ Safe: You still have at least one day-trade slot available.")
    else:
        print("\n💰 Equity > $25k — PDT rules do not apply.")
    
    print(f"API Reported Day Trades (5d): {acc.daytrade_count}  ✅ (authoritative)")
    print(f"Estimated Day Trades (5d): {est}  ⚠️ (heuristic, may differ)")

if __name__ == "__main__":
    check_pdt_status()