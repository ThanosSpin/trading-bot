#!/usr/bin/env python3
"""Check and cancel open orders"""

import os
from dotenv import load_dotenv
from alpaca_trade_api.rest import REST

# ✅ Load environment variables from .env file
load_dotenv()

# Initialize Alpaca client
API_KEY = os.getenv('ALPACA_MARKET_API_KEY')
API_SECRET = os.getenv('ALPACA_MARKET_SECRET_KEY')
BASE_URL = os.getenv('ALPACA_MARKET_BASE_URL')

if not API_KEY or not API_SECRET:
    print("❌ Error: API credentials not found!")
    print("Make sure your .env file contains:")
    print("  APCA_API_KEY_ID=your_key")
    print("  APCA_API_SECRET_KEY=your_secret")
    exit(1)

api = REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

print("="*60)
print("CHECKING OPEN ORDERS")
print("="*60)

try:
    orders = api.list_orders(status='open')
    
    if not orders:
        print("\n✅ No open orders - you're clear!\n")
    else:
        print(f"\n⚠️ Found {len(orders)} open order(s):\n")
        
        for order in orders:
            print(f"📋 {order.symbol}: {order.side.upper()} {order.qty} shares")
            print(f"   Status: {order.status}")
            print(f"   ID: {order.id}")
            print()
        
        # Check for AAPL SELL orders
        aapl_sells = [o for o in orders if o.symbol == 'AAPL' and o.side == 'sell']
        
        if aapl_sells:
            print("🚨 FOUND AAPL SELL ORDER(S)!")
            response = input("Cancel AAPL SELL orders? (y/n): ")
            
            if response.lower() == 'y':
                for order in aapl_sells:
                    try:
                        api.cancel_order(order.id)
                        print(f"✅ Canceled AAPL SELL {order.id}")
                    except Exception as e:
                        print(f"❌ Failed: {e}")
    
    print("="*60)

except Exception as e:
    print(f"❌ Error: {e}")
