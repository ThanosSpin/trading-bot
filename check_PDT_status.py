# check_pdt_status.py
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
import pytz
from collections import defaultdict
from config import API_MARKET_KEY, API_MARKET_SECRET, MARKET_BASE_URL


def estimate_daytrade_count(api_client, days=5):
    cutoff = datetime.utcnow() - timedelta(days=days)
    cutoff = cutoff.replace(tzinfo=pytz.UTC)
    try:
        orders = api_client.list_orders(status="filled", limit=1000, nested=True)
    except Exception as e:
        print(f"[WARN] Unable to fetch filled orders for PDT estimate: {e}")
        return 0

    buckets = defaultdict(lambda: {"buy": 0, "sell": 0})
    for o in orders:
        filled_at = getattr(o, "filled_at", None)
        if filled_at is None:
            continue
        if filled_at.tzinfo is None:
            filled_at = filled_at.replace(tzinfo=pytz.UTC)
        if filled_at < cutoff:
            continue
        date = filled_at.date()
        key = (getattr(o, "symbol", None), date)
        side = getattr(o, "side", "").lower()
        if side in ("buy", "sell"):
            buckets[key][side] += 1

    matched_pairs = sum(min(v["buy"], v["sell"]) for v in buckets.values())
    return matched_pairs


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


if __name__ == "__main__":
    check_pdt_status()