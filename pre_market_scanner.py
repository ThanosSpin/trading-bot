#!/usr/bin/env python
"""
Pre-Market Scanner
Runs before market open to identify high-probability opportunities
and queue limit orders for market open execution.

✨ ENHANCED: Now supports both BUY and SELL (short) signals

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
from trader import api

from config import (
    PRE_MARKET_ENABLED,
    PRE_MARKET_MIN_PROB,
    PRE_MARKET_MAX_ALLOCATION,
    PRE_MARKET_LIMIT_BUFFER,
    PRE_MARKET_SYMBOLS,
    INTRADAY_WEIGHT,
    PRE_MARKET_ALLOW_SHORT_SELLING,
    PRE_MARKET_MIN_SELL_PROB,
    PRE_MARKET_SHORT_ALLOCATION,
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
        cancelled_count = 0
        
        for order in open_orders:
            # Cancel orders placed before market open (before 9:30 AM)
            order_time = getattr(order, 'created_at', None)
            if order_time and (order_time.hour < 9 or (order_time.hour == 9 and order_time.minute < 30)):
                api.cancel_order(order.id)
                limit_price = getattr(order, 'limit_price', 'N/A')
                side = getattr(order, 'side', 'N/A')
                print(f"  🗑️ Cancelled old pre-market order: {side.upper()} {order.symbol} {order.qty} @ ${limit_price}")
                cancelled_count += 1
        
        if cancelled_count > 0:
            print(f"  ✅ Cancelled {cancelled_count} old order(s)")
        else:
            print(f"  ℹ️ No old orders to cancel")
                
    except Exception as e:
        print(f"[WARN] Could not cancel old orders: {e}")


def check_margin_account():
    """
    Check if account has margin enabled (required for short selling).
    
    Returns:
        (is_margin_enabled, account_equity)
    """
    try:
        account = account_cache.get_account()
        
        # Check account type
        account_type = account.get('account_blocked', False)
        if account_type:
            return False, 0.0
        
        # Check equity (need $2,000+ for margin)[web:68]
        equity = float(account.get('equity', 0))
        
        # Alpaca requires $2,000 minimum for margin/short selling
        if equity < 2000:
            return False, equity
        
        return True, equity
        
    except Exception as e:
        print(f"[WARN] Could not check margin status: {e}")
        return False, 0.0


def is_shortable(symbol: str) -> bool:
    """
    Check if a symbol is available for short selling (Easy-To-Borrow).
    
    Args:
        symbol: Stock symbol
    
    Returns:
        True if shortable, False otherwise
    """
    try:
        # Get asset info
        asset = api.get_asset(symbol)
        
        # Check if shortable (Alpaca only supports ETB stocks)[web:68]
        shortable = getattr(asset, 'easy_to_borrow', False)
        
        if not shortable:
            print(f"  ⚠️ {symbol} is not easy-to-borrow (cannot short)")
        
        return shortable
        
    except Exception as e:
        print(f"  ⚠️ Could not check shortability for {symbol}: {e}")
        return False


def scan_and_queue_orders():
    """
    Main scanner function:
    1. Compute signals for all symbols
    2. Identify high-conviction opportunities (prob >= 75% for BUY, prob <= 25% for SELL)
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
    
    # Check margin account (for short selling)
    allow_short = getattr(PRE_MARKET_ALLOW_SHORT_SELLING, '__bool__', lambda: False)()
    is_margin, account_equity = check_margin_account()
    
    if allow_short and not is_margin:
        print(f"\n⚠️ Short selling enabled but margin not available")
        if account_equity > 0:
            print(f"   Account equity: ${account_equity:,.2f} (need $2,000+ for margin)")
        print(f"   Only BUY signals will be processed\n")
        allow_short = False
    elif allow_short:
        print(f"\n✅ Margin account enabled - both BUY and SELL signals active")
        print(f"   Account equity: ${account_equity:,.2f}\n")
    
    # Get symbols
    symbols = PRE_MARKET_SYMBOLS if isinstance(PRE_MARKET_SYMBOLS, list) else [PRE_MARKET_SYMBOLS]
    
    # Get account info
    account_cache.invalidate()
    account = account_cache.get_account()
    available_cash = float(account.get("cash", 0))
    buying_power = float(account.get("buying_power", available_cash))
    
    print(f"\n💰 Available Cash: ${available_cash:,.2f}")
    print(f"💪 Buying Power: ${buying_power:,.2f}")
    print(f"📊 Scanning {len(symbols)} symbols for high-conviction signals...")
    
    # ✅ ENHANCED: Show thresholds for both BUY and SELL
    min_sell_prob = getattr(PRE_MARKET_MIN_SELL_PROB, 0.35)
    print(f"🎯 BUY Threshold:  Probability >= {PRE_MARKET_MIN_PROB:.0%}")
    if allow_short:
        print(f"🎯 SELL Threshold: Probability <= {min_sell_prob:.0%}\n")
    else:
        print()
    
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
            
            # ============================================================
            # BUY SIGNAL (High probability)
            # ============================================================
            if prob >= PRE_MARKET_MIN_PROB:
                print(f"\n  🎯 HIGH CONVICTION BUY SIGNAL!")
                
                # Calculate position size
                allocation = available_cash * PRE_MARKET_MAX_ALLOCATION
                max_qty = int(allocation // price)
                
                if max_qty <= 0:
                    print(f"  ⚠️ Insufficient cash for even 1 share")
                    continue
                
                # Set limit price slightly above to ensure fill at open
                limit_price = round(price * (1 + PRE_MARKET_LIMIT_BUFFER), 2)
                
                print(f"  📋 Preparing BUY limit order:")
                print(f"     Quantity: {max_qty} shares")
                print(f"     Limit Price: ${limit_price:.2f}")
                print(f"     Total Value: ${max_qty * limit_price:,.2f}")
                
                # Submit BUY order
                try:

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
                        'side': 'BUY',
                        'qty': max_qty,
                        'limit_price': limit_price,
                        'probability': prob,
                        'daily_prob': daily_prob,
                        'intraday_prob': intraday_prob,
                        'order_id': order.id,
                        'estimated_value': max_qty * limit_price
                    })
                    
                    print(f"  ✅ BUY order queued! Order ID: {order.id}")
                    
                    # Update available cash
                    available_cash -= (max_qty * limit_price)
                    
                except Exception as e:
                    print(f"  ❌ Failed to submit BUY order: {e}")
                    import traceback
                    traceback.print_exc()
            
            # ============================================================
            # SELL SIGNAL (Low probability - short selling)
            # ============================================================
            elif allow_short and prob <= min_sell_prob:
                print(f"\n  🎯 HIGH CONVICTION SELL SIGNAL!")
                
                # Check if symbol is shortable
                if not is_shortable(sym):
                    print(f"  ⚠️ {sym} not available for short selling - skipping")
                    continue
                
                # Calculate position size for short (use buying power, not cash)
                short_allocation = getattr(PRE_MARKET_SHORT_ALLOCATION, 0.10)
                allocation = buying_power * short_allocation
                max_qty = int(allocation // price)
                
                if max_qty <= 0:
                    print(f"  ⚠️ Insufficient buying power for even 1 share")
                    continue
                
                # Set limit price slightly below to ensure fill at open (for short)
                # Note: For shorting, we want to sell at current or lower price
                limit_price = round(price * (1 - PRE_MARKET_LIMIT_BUFFER), 2)
                
                print(f"  📋 Preparing SHORT limit order:")
                print(f"     Quantity: {max_qty} shares")
                print(f"     Limit Price: ${limit_price:.2f} (short sell)")
                print(f"     Total Value: ${max_qty * limit_price:,.2f}")
                
                # Submit SHORT order
                try:
                    order = api.submit_order(
                        symbol=sym,
                        qty=max_qty,
                        side='sell',  # 'sell' = short sell for Alpaca[web:65]
                        type='limit',
                        time_in_force='day',
                        limit_price=limit_price
                    )
                    
                    queued_orders.append({
                        'symbol': sym,
                        'side': 'SELL',
                        'qty': max_qty,
                        'limit_price': limit_price,
                        'probability': prob,
                        'daily_prob': daily_prob,
                        'intraday_prob': intraday_prob,
                        'order_id': order.id,
                        'estimated_value': max_qty * limit_price
                    })
                    
                    print(f"  ✅ SHORT order queued! Order ID: {order.id}")
                    
                    # Update buying power (shorting uses margin)
                    buying_power -= (max_qty * limit_price)
                    
                except Exception as e:
                    print(f"  ❌ Failed to submit SHORT order: {e}")
                    import traceback
                    traceback.print_exc()
            
            else:
                # Neutral signal
                print(f"  ℹ️ Probability {prob:.1%} in neutral zone ({min_sell_prob:.0%} - {PRE_MARKET_MIN_PROB:.0%}) - skipping")
                
        except Exception as e:
            print(f"  ❌ Error analyzing {sym}: {e}")
            import traceback
            traceback.print_exc()
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'='*80}")
    print(f"📊 SCAN COMPLETE")
    print(f"{'='*80}")
    
    if queued_orders:
        # Count by side
        buy_orders = [o for o in queued_orders if o['side'] == 'BUY']
        sell_orders = [o for o in queued_orders if o['side'] == 'SELL']
        
        print(f"\n✅ Queued {len(queued_orders)} pre-market order(s):")
        print(f"   🟢 {len(buy_orders)} BUY order(s)")
        print(f"   🔴 {len(sell_orders)} SELL (short) order(s)\n")
        
        total_buy_value = 0
        total_sell_value = 0
        for order in queued_orders:
            daily = order['daily_prob']
            intraday = order['intraday_prob']
            prob_str = f"D:{daily:.1%} I:{intraday:.1%}" if daily and intraday else f"{order['probability']:.1%}"
            
            side_emoji = "🟢" if order['side'] == 'BUY' else "🔴"
            
            print(f"  {side_emoji} {order['symbol']}")
            print(f"     Signal: {order['side']} (Probability: {order['probability']:.1%})")
            print(f"     Breakdown: {prob_str}")
            print(f"     Order: {order['side']} {order['qty']} @ ${order['limit_price']:.2f}")
            print(f"     Value: ${order['estimated_value']:,.2f}")
            print(f"     Order ID: {order['order_id']}\n")
            
            if order['side'] == 'BUY':
                total_buy_value += order['estimated_value']
            else:
                total_sell_value += order['estimated_value']
        
        print(f"  💰 Total BUY Value:  ${total_buy_value:,.2f}")
        if total_sell_value > 0:
            print(f"  💰 Total SELL Value: ${total_sell_value:,.2f}")
        print(f"\n📅 Orders will execute at market open (9:30 AM ET)")
        
        # Send notification
        notification_msg = f"Pre-Market Scanner queued {len(queued_orders)} order(s):\n\n"
        if buy_orders:
            notification_msg += f"🟢 BUY Orders ({len(buy_orders)}):\n"
            for order in buy_orders:
                notification_msg += f"  {order['symbol']}: {order['qty']} @ ${order['limit_price']:.2f} ({order['probability']:.1%})\n"
        
        if sell_orders:
            notification_msg += f"\n🔴 SELL (Short) Orders ({len(sell_orders)}):\n"
            for order in sell_orders:
                notification_msg += f"  {order['symbol']}: {order['qty']} @ ${order['limit_price']:.2f} ({order['probability']:.1%})\n"
        
        notification_msg += f"\nTotal Value: BUY ${total_buy_value:,.2f}"
        if total_sell_value > 0:
            notification_msg += f" | SELL ${total_sell_value:,.2f}"
        
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
