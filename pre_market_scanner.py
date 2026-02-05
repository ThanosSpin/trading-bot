#!/usr/bin/env python
"""
Pre-Market Scanner
Runs before market open to identify high-probability opportunities
and queue limit orders for market open execution.

Usage: python pre_market_scanner.py
Schedule: cron at 8:00 AM weekdays
"""

import os
import sys
from datetime import datetime, timezone

from model_xgb import compute_signals
from data_loader import fetch_latest_price
from account_cache import account_cache
from market import is_trading_day
from trader import api  # ✅ FIXED: Import 'api' instead of 'trading_client'

from config import (
    PRE_MARKET_ENABLED,
    PRE_MARKET_MIN_PROB,
    PRE_MARKET_MAX_ALLOCATION,
    PRE_MARKET_LIMIT_BUFFER,
    PRE_MARKET_SYMBOLS,
    INTRADAY_WEIGHT
)

# Set timezone
os.environ["TZ"] = "America/New_York"

def send_notification(title: str, message: str):
    """Send email/log notification"""
    print(f"\n{'='*60}")
    print(f"📧 {title}")
    print(f"{'='*60}")
    print(message)
    print(f"{'='*60}\n")
    
    # TODO: Add email integration if you have it configured
    # send_email(title, message)


def get_open_orders(symbol: str = None):
    """Get list of open orders, optionally filtered by symbol"""
    try:
        # ✅ FIXED: Use legacy alpaca_trade_api syntax
        orders = api.list_orders(
            status='open',
            symbols=symbol if symbol else None
        )
        return orders
        
    except Exception as e:
        print(f"[WARN] Could not fetch open orders: {e}")
        return []


def cancel_existing_premarket_orders():
    """Cancel any existing pre-market orders from previous scans"""
    try:
        open_orders = get_open_orders()
        
        for order in open_orders:
            # Cancel orders placed before market open (before 9:30 AM)
            order_time = getattr(order, 'created_at', None)
            if order_time and (order_time.hour < 9 or (order_time.hour == 9 and order_time.minute < 30)):
                api.cancel_order(order.id)
                limit_price = getattr(order, 'limit_price', 'N/A')
                print(f"  🗑️ Cancelled old pre-market order: {order.symbol} {order.qty} @ ${limit_price}")
                
    except Exception as e:
        print(f"[WARN] Could not cancel old orders: {e}")


def scan_and_queue_orders():
    """
    Main scanner function:
    1. Compute signals for all symbols
    2. Identify high-conviction opportunities (prob >= 75%)
    3. Queue limit orders for market open
    """
    
    if not PRE_MARKET_ENABLED:
        print("[INFO] Pre-market scanner is disabled in config")
        return []
    
    # Check if today is a trading day
    if not is_trading_day():
        print(f"[INFO] {datetime.now().strftime('%Y-%m-%d')} is not a trading day. Exiting.")
        return []
    
    print(f"\n{'='*80}")
    print(f"🌅 PRE-MARKET SCANNER - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"{'='*80}\n")
    
    # Cancel old orders first
    print("🧹 Cleaning up old pre-market orders...")
    cancel_existing_premarket_orders()
    
    # Get symbols
    symbols = PRE_MARKET_SYMBOLS if isinstance(PRE_MARKET_SYMBOLS, list) else [PRE_MARKET_SYMBOLS]
    
    # Get account info
    account_cache.invalidate()
    account = account_cache.get_account()
    available_cash = float(account.get("cash", 0))
    
    print(f"\n💰 Available Cash: ${available_cash:,.2f}")
    print(f"📊 Scanning {len(symbols)} symbols for high-conviction signals...")
    print(f"🎯 Minimum Probability: {PRE_MARKET_MIN_PROB:.0%}\n")
    
    queued_orders = []
    
    for sym in symbols:
        print(f"\n{'─'*60}")
        print(f"🔍 Analyzing {sym}...")
        
        try:
            # Compute signals
            sig = compute_signals(
                sym,
                lookback_minutes=2400,
                intraday_weight=INTRADAY_WEIGHT,
                resample_to="15min"
            )
            
            prob = sig.get("final_prob")
            daily_prob = sig.get("daily_prob")
            intraday_prob = sig.get("intraday_prob")
            price = sig.get("price")
            
            if prob is None or price is None or price <= 0:
                print(f"  ⚠️ Invalid signal data for {sym}, skipping")
                continue
            
            print(f"  📈 Daily Prob: {daily_prob:.1%}" if daily_prob else "  📈 Daily Prob: N/A")
            print(f"  ⚡ Intraday Prob: {intraday_prob:.1%}" if intraday_prob else "  ⚡ Intraday Prob: N/A")
            print(f"  🎲 Final Prob: {prob:.1%}")
            print(f"  💵 Last Price: ${price:.2f}")
            
            # Check if high conviction
            if prob >= PRE_MARKET_MIN_PROB:
                print(f"\n  🎯 HIGH CONVICTION SIGNAL DETECTED!")
                
                # Calculate position size
                allocation = available_cash * PRE_MARKET_MAX_ALLOCATION
                max_qty = int(allocation // price)
                
                if max_qty <= 0:
                    print(f"  ⚠️ Insufficient cash for even 1 share")
                    continue
                
                # Set limit price slightly above to ensure fill at open
                limit_price = round(price * (1 + PRE_MARKET_LIMIT_BUFFER), 2)
                
                print(f"  📋 Preparing limit order:")
                print(f"     Quantity: {max_qty} shares")
                print(f"     Limit Price: ${limit_price:.2f}")
                print(f"     Total Value: ${max_qty * limit_price:,.2f}")
                
                # Submit limit order using legacy API
                try:
                    # ✅ FIXED: Use legacy alpaca_trade_api syntax
                    order = api.submit_order(
                        symbol=sym,
                        qty=max_qty,
                        side='buy',
                        type='limit',
                        time_in_force='day',
                        limit_price=limit_price
                    )
                    
                    queued_orders.append({
                        'symbol': sym,
                        'qty': max_qty,
                        'limit_price': limit_price,
                        'probability': prob,
                        'daily_prob': daily_prob,
                        'intraday_prob': intraday_prob,
                        'order_id': order.id,
                        'estimated_value': max_qty * limit_price
                    })
                    
                    print(f"  ✅ Order queued successfully! Order ID: {order.id}")
                    
                    # Update available cash (for next symbol)
                    available_cash -= (max_qty * limit_price)
                    
                except Exception as e:
                    print(f"  ❌ Failed to submit order: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"  ℹ️ Probability {prob:.1%} below threshold {PRE_MARKET_MIN_PROB:.0%} - skipping")
                
        except Exception as e:
            print(f"  ❌ Error analyzing {sym}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*80}")
    print(f"📊 SCAN COMPLETE")
    print(f"{'='*80}")
    
    if queued_orders:
        print(f"\n✅ Queued {len(queued_orders)} pre-market order(s):\n")
        
        total_value = 0
        for order in queued_orders:
            daily = order['daily_prob']
            intraday = order['intraday_prob']
            prob_str = f"D:{daily:.1%} I:{intraday:.1%}" if daily and intraday else f"{order['probability']:.1%}"
            
            print(f"  🎯 {order['symbol']}")
            print(f"     Probability: {order['probability']:.1%} ({prob_str})")
            print(f"     Order: BUY {order['qty']} @ ${order['limit_price']:.2f}")
            print(f"     Value: ${order['estimated_value']:,.2f}")
            print(f"     Order ID: {order['order_id']}\n")
            total_value += order['estimated_value']
        
        print(f"  💰 Total Queued Value: ${total_value:,.2f}")
        print(f"\n📅 Orders will execute at market open (9:30 AM ET)")
        
        # Send notification
        notification_msg = f"Pre-Market Scanner queued {len(queued_orders)} order(s):\n\n"
        for order in queued_orders:
            notification_msg += f"{order['symbol']}: BUY {order['qty']} @ ${order['limit_price']:.2f} ({order['probability']:.1%})\n"
        notification_msg += f"\nTotal Value: ${total_value:,.2f}"
        
        send_notification(
            title=f"🌅 Pre-Market: {len(queued_orders)} Order(s) Queued",
            message=notification_msg
        )
    else:
        print("\nℹ️ No high-conviction opportunities found")
        print("💤 No orders queued for today")
    
    print(f"\n{'='*80}\n")
    
    return queued_orders


def main():
    """Entry point"""
    try:
        queued_orders = scan_and_queue_orders()
        
        if queued_orders:
            print("✅ Pre-market scan completed successfully")
            sys.exit(0)
        else:
            print("ℹ️ Pre-market scan completed - no orders queued")
            sys.exit(0)
            
    except Exception as e:
        print(f"\n❌ Pre-market scanner failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
