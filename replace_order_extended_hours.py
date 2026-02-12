#!/usr/bin/env python3
"""Replace regular order with extended hours order"""

from dotenv import load_dotenv
import os
from alpaca_trade_api.rest import REST

load_dotenv()

api = REST(
    os.getenv('ALPACA_MARKET_API_KEY'),
    os.getenv('ALPACA_MARKET_SECRET_KEY'),
    os.getenv('ALPACA_MARKET_BASE_URL'),
    api_version='v2'
)

print("="*60)
print("REPLACING WITH EXTENDED HOURS ORDER")
print("="*60)

# Step 1: Cancel existing order
print("\n🗑️ Step 1: Canceling existing order...")
try:
    order_id = "44fafad5-c442-4de8-970a-093a21b1d9c1"
    api.cancel_order(order_id)
    print(f"✅ Canceled order: {order_id}")
except Exception as e:
    print(f"⚠️ Cancel warning: {e}")

# Step 2: Get current price
print("\n📊 Step 2: Checking current after-hours price...")
try:
    quote = api.get_latest_quote("AAPL")
    bid = float(quote.bid_price)
    ask = float(quote.ask_price)
    
    print(f"   Bid: ${bid:.2f}")
    print(f"   Ask: ${ask:.2f}")
    print(f"   Spread: ${ask-bid:.2f}")
    
    # Set limit price slightly below bid for quick fill
    limit_price = round(bid - 0.25, 2)  # $0.25 below bid
    
except Exception as e:
    print(f"⚠️ Quote error: {e}")
    limit_price = 274  # Conservative fallback

# Step 3: Place extended hours limit order
print(f"\n📝 Step 3: Placing extended hours LIMIT order @ ${limit_price}...")

try:
    order = api.submit_order(
        symbol='AAPL',
        qty=2,
        side='sell',
        type='limit',
        limit_price=limit_price,
        time_in_force='gtc',
        extended_hours=True  # ✅ KEY
    )
    
    print(f"\n✅ SUCCESS! Extended hours order placed!")
    print(f"   Order ID: {order.id}")
    print(f"   Status: {order.status}")
    print(f"   Limit: ${limit_price}")
    print(f"   Extended hours: YES")
    print(f"\n⏰ Should fill within 1-5 minutes if buyers available")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\nTrying market order as backup...")
    
    # Backup: Try market order
    try:
        order = api.submit_order(
            symbol='AAPL',
            qty=2,
            side='sell',
            type='market',
            time_in_force='day',
            extended_hours=True
        )
        print(f"✅ Market order placed: {order.id}")
    except Exception as e2:
        print(f"❌ Market order also failed: {e2}")

print("="*60)