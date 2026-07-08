# check_margin_status.py (refactored from check_pdt_status.py)
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alpaca_trade_api as tradeapi
from config.config import API_MARKET_KEY, API_MARKET_SECRET, MARKET_BASE_URL

def check_margin_status():
    api = tradeapi.REST(API_MARKET_KEY, API_MARKET_SECRET, MARKET_BASE_URL, api_version='v2')
    acc = api.get_account()

    print("\n🔍 Alpaca Margin / Buying Power Check")
    print("-" * 40)

    equity = float(getattr(acc, "equity", 0.0) or 0.0)
    bp = float(getattr(acc, "buying_power", 0.0) or 0.0)
    regt_bp = getattr(acc, "regt_buying_power", None)
    day_bp = getattr(acc, "daytrading_buying_power", None)  # may now be removed
    nonmargin_bp = getattr(acc, "non_marginable_buying_power", None)
    multiplier = getattr(acc, "multiplier", None)
    trading_blocked = bool(getattr(acc, "trading_blocked", False))

    print(f"Equity:        ${equity:.2f}")
    print(f"Buying Power:  ${bp:.2f}")
    if regt_bp is not None:
        print(f"Reg T BP:      {regt_bp}")
    if day_bp is not None:
        print(f"Legacy Day BP: {day_bp}")
    if nonmargin_bp is not None:
        print(f"Non-margin BP: {nonmargin_bp}")
    print(f"Multiplier:    {multiplier if multiplier is not None else 'N/A'}")
    print(f"Trading Blocked: {trading_blocked}")
    print("-" * 40)

    # Simple “sanity” checks under intraday margin
    if trading_blocked:
        print("⚠️ Account is currently trading_blocked — broker will reject new orders.")
    else:
        if equity < 2000:
            print("⚠️ Equity < $2,000 — margin availability may be limited.")
        else:
            print("✅ Equity above $2,000 — margin/trading likely available.")

        if bp < 0.5 * equity:
            print("⚠️ Buying power is less than 50% of equity — margin fairly tight.")
        else:
            print("✅ Buying power is healthy relative to equity.")

if __name__ == "__main__":
    check_margin_status()