#!/usr/bin/env python
"""
Pre-Market Scanner
Runs before market open to identify high-probability opportunities
and queue limit orders for market open execution.

✨ ENHANCED: Now supports both BUY and SELL signals
- BUY: Opens new long positions on high conviction signals
- SELL: Closes existing long positions OR shorts (if margin enabled)

Usage: python pre_market_scanner.py
Schedule: cron at 8:00 AM weekdays
"""

import os
import sys
from datetime import datetime, timezone

from predictive_model.model_xgb import compute_signals
from predictive_model.data_loader import fetch_latest_price
from account_cache import account_cache
from market import is_trading_day
from trader import api
from pdt.pdt_tracker import get_opened_today_qty
from order_utils import get_order_params, print_market_status, get_market_session

from config.config import (
    PRE_MARKET_ENABLED,
    PRE_MARKET_MIN_PROB,
    PRE_MARKET_MAX_ALLOCATION,
    PRE_MARKET_LIMIT_BUFFER,
    PRE_MARKET_SYMBOLS,
    INTRADAY_WEIGHT,
    PRE_MARKET_ALLOW_SHORT_SELLING,
    PRE_MARKET_MIN_SELL_PROB,
    PRE_MARKET_SHORT_ALLOCATION,
    LIMIT_BUFFER_PCT,
)

# Set timezone
os.environ["TZ"] = "America/New_York"


# ============================================================
# ✨ NEW: Session-aware order submission function
# ============================================================
def submit_order_smart(symbol, side, qty, current_price, limit_buffer_pct=LIMIT_BUFFER_PCT, 
                       available_cash=None, buying_power=None, refresh_price=True):
    """
    Submit order with automatic type selection based on market session.
    Includes pre-validation to prevent insufficient buying power errors.
    
    Args:
        symbol: Stock symbol
        side: 'buy' or 'sell'
        qty: Number of shares
        current_price: Current stock price (from signal)
        limit_buffer_pct: Buffer percentage for limit orders (default 2%)
        available_cash: Available cash (for buy orders)
        buying_power: Available buying power (for short sells)
        refresh_price: If True, fetch latest real-time price (default True)
    
    Returns:
        (order, order_type, limit_price or None)
    
    Raises:
        ValueError: If insufficient funds
    """
    
    # ═══════════════════════════════════════════════════════════
    # ✅ STEP 1: REFRESH PRICE (Get real-time market price)
    # ═══════════════════════════════════════════════════════════
    if refresh_price:
        print(f"  🔍 Refreshing {symbol} price...")
        print(f"     Signal price: ${current_price:.2f}")
        
        try:
            # Import utilities
            from order_utils import get_market_session
            from predictive_model.data_loader import fetch_latest_price
            
            # Check if we're in extended hours (pre-market, after-hours, or closed)
            session = get_market_session()
            in_extended_hours = (session in ['pre_market', 'after_hours', 'closed'])
            
            print(f"     Market session: {session}")
            
            # Force yfinance during extended hours (Alpaca returns stale data)
            if in_extended_hours:
                print(f"     ⏰ Extended hours - using yfinance for real-time price")
            
            # Fetch fresh price
            fresh_price = fetch_latest_price(symbol, prefer_yfinance=in_extended_hours)
            
            if fresh_price and fresh_price > 0:
                price_diff = fresh_price - current_price
                price_diff_pct = (price_diff / current_price) * 100
                
                # Always update if different (even by 1 cent)
                if abs(price_diff) > 0.01:
                    print(f"  🔄 Price update: ${current_price:.2f} → ${fresh_price:.2f} ({price_diff_pct:+.2f}%)")
                    current_price = fresh_price
                else:
                    print(f"  ✓ Price confirmed: ${current_price:.2f}")
                
                # Warn if major price movement
                if abs(price_diff_pct) > 2.0:
                    print(f"  ⚠️ WARNING: Price moved {abs(price_diff_pct):.2f}% since signal!")
            else:
                print(f"  ⚠️ Could not fetch fresh price")
                print(f"  📌 Using signal price: ${current_price:.2f}")
        
        except Exception as e:
            print(f"  ❌ Price refresh failed: {e}")
            print(f"  📌 Using signal price: ${current_price:.2f}")
            import traceback
            traceback.print_exc()
    else:
        print(f"  ℹ️ Price refresh disabled - using signal price: ${current_price:.2f}")
    
    # ═══════════════════════════════════════════════════════════
    # ✅ STEP 2: PRE-VALIDATE FUNDS (Check before submitting)
    # ═══════════════════════════════════════════════════════════
    if side == 'buy':
        # Calculate required cash (with buffer)
        estimated_cost = qty * current_price * (1 + limit_buffer_pct)
        
        print(f"  💰 Cost check:")
        print(f"     Estimated cost: ${estimated_cost:.2f}")
        print(f"     Available cash: ${available_cash:.2f}" if available_cash else "     Available cash: Not provided")
        
        if available_cash is not None and available_cash < estimated_cost:
            raise ValueError(
                f"Insufficient cash: Need ${estimated_cost:.2f}, have ${available_cash:.2f}"
            )
        else:
            print(f"     ✅ Sufficient funds")
    
    elif side == 'sell' and qty < 0:  # Short selling
        # Calculate required buying power
        estimated_cost = abs(qty) * current_price * (1 + limit_buffer_pct)
        
        if buying_power is not None and buying_power < estimated_cost:
            raise ValueError(
                f"Insufficient buying power for short: Need ${estimated_cost:.2f}, have ${buying_power:.2f}"
            )
    
    # ═══════════════════════════════════════════════════════════
    # ✅ STEP 3: GET ORDER PARAMETERS (Session-aware)
    # ═══════════════════════════════════════════════════════════
    try:
        # Get session-aware order parameters (market vs limit)
        params = get_order_params(
            symbol=symbol,
            side=side,
            qty=abs(qty),  # Ensure positive qty
            current_price=current_price,  # ✅ Using refreshed price
            limit_buffer_pct=limit_buffer_pct,
            refresh_price=False  # Already refreshed above
        )
        
        # ═══════════════════════════════════════════════════════════
        # ✅ STEP 4: SUBMIT ORDER TO BROKER
        # ═══════════════════════════════════════════════════════════
        order = api.submit_order(**params)
        
        order_type = params['type']
        limit_price = params.get('limit_price', None)
        
        return order, order_type, limit_price
        
    except Exception as e:
        # Re-raise with more context
        error_msg = str(e)
        if 'insufficient buying power' in error_msg.lower():
            raise ValueError(
                f"Broker rejected order: Insufficient buying power. "
                f"Order: {side.upper()} {qty} {symbol} @ ${current_price:.2f}"
            ) from e
        else:
            raise


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
        account_blocked = account.get('account_blocked', False)
        if account_blocked:
            return False, 0.0
        
        # Check equity (need $2,000+ for margin)
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
        
        # Check if shortable (Alpaca only supports ETB stocks)
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
    2. Identify high-conviction opportunities
    3. Queue appropriate orders based on market session
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
    
    # ✅ NEW: Show market session status
    print_market_status()
    print()
    
    # Cancel old orders first
    print("🧹 Cleaning up old pre-market orders...")
    cancel_existing_premarket_orders()
    
    # Check margin account (for short selling)
    allow_short = PRE_MARKET_ALLOW_SHORT_SELLING
    is_margin, account_equity = check_margin_account()
    
    if allow_short and not is_margin:
        print(f"\n⚠️ Short selling enabled but margin not available")
        if account_equity > 0:
            print(f"   Account equity: ${account_equity:,.2f} (need $2,000+ for margin)")
        print(f"   Will close existing positions but not open new shorts\n")
        allow_short = False
    elif allow_short:
        print(f"\n✅ Margin account enabled - short selling available")
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

    # ✅ NEW: Check for orders reserving cash
    print(f"\n🔍 Checking for pending orders...")
    try:
        open_orders = api.list_orders(status='open')
        
        if open_orders:
            print(f"   Found {len(open_orders)} open order(s):")
            total_reserved = 0
            
            for order in open_orders:
                if order.side == 'buy':
                    limit_price = float(order.limit_price) if order.limit_price else 0
                    reserved = float(order.qty) * limit_price
                    total_reserved += reserved
                    print(f"   • {order.symbol}: BUY {order.qty} @ ${limit_price:.2f} = ${reserved:.2f} reserved")
            
            if total_reserved > 0:
                print(f"\n⚠️ Total cash reserved: ${total_reserved:.2f}")
                available_cash_adjusted = available_cash - total_reserved
                print(f"💵 Adjusted available cash: ${available_cash_adjusted:.2f}")
                
                if available_cash_adjusted < 10:
                    print(f"\n❌ No funds available after accounting for pending orders!")
                    print(f"   Option 1: Cancel pending orders")
                    print(f"   Option 2: Wait for orders to fill")
                    return []
                
                # Use adjusted cash
                available_cash = available_cash_adjusted
        else:
            print(f"   ✅ No open orders")
    
    except Exception as e:
        print(f"   ⚠️ Could not check open orders: {e}")
    
    # ✅ Early exit if no cash
    if available_cash < 10:
        print(f"\n⚠️ Insufficient cash to trade (${available_cash:.2f})")
        print(f"   Need at least $10 to place orders")
        print(f"   Exiting scanner...")
        return []
    
    print(f"\n📊 Scanning {len(symbols)} symbols for high-conviction signals...")

   # ✅ NEW: Early exit if no cash
    if available_cash < 10:
        print(f"\n⚠️ Insufficient cash to trade (${available_cash:.2f})")
        print(f"   Need at least $10 to place orders")
        print(f"   Exiting scanner...")
        return []

    print(f"📊 Scanning {len(symbols)} symbols for high-conviction signals...")
    
    # Show thresholds
    min_sell_prob = PRE_MARKET_MIN_SELL_PROB
    print(f"🎯 BUY Threshold:  Probability >= {PRE_MARKET_MIN_PROB:.0%}")

    print(f"🎯 SELL Threshold: Probability <= {min_sell_prob:.0%}\n")
    

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
                    print(f"     Need: ${price:.2f}")
                    print(f"     Have: ${available_cash:.2f}")
                    continue
                
                # ✅ Calculate actual cost with buffer
                estimated_cost = max_qty * price * (1 + PRE_MARKET_LIMIT_BUFFER)
                
                # ✅ Reduce qty if over budget
                while max_qty > 0 and estimated_cost > available_cash:
                    max_qty -= 1
                    estimated_cost = max_qty * price * (1 + PRE_MARKET_LIMIT_BUFFER)
                
                if max_qty <= 0:
                    print(f"  ⚠️ Insufficient cash after accounting for ±2% buffer")
                    print(f"     Cash: ${available_cash:.2f}")
                    print(f"     Need: ${price * (1 + PRE_MARKET_LIMIT_BUFFER):.2f} per share")
                    continue
                
                print(f"  📋 Preparing BUY order:")
                print(f"     Quantity: {max_qty} shares")
                print(f"     Available cash: ${available_cash:.2f}")
                
                # ✅ NEW: Use session-aware order submission with validation
                try:
                    order, order_type, limit_price = submit_order_smart(
                        symbol=sym,
                        side='buy',
                        qty=max_qty,
                        current_price=price,
                        limit_buffer_pct=PRE_MARKET_LIMIT_BUFFER,
                        available_cash=available_cash,  # ✅ Pass available cash
                        refresh_price=True,
                    )
                    
                    if order_type == 'market':
                        print(f"     Type: MARKET (immediate execution)")
                        estimated_value = max_qty * price
                    else:
                        print(f"     Type: LIMIT @ ${limit_price:.2f}")
                        estimated_value = max_qty * limit_price
                    
                    print(f"     Total Value: ${estimated_value:,.2f}")
                    
                    queued_orders.append({
                        'symbol': sym,
                        'side': 'BUY',
                        'qty': max_qty,
                        'limit_price': limit_price if order_type == 'limit' else None,
                        'order_type': order_type,
                        'probability': prob,
                        'daily_prob': daily_prob,
                        'intraday_prob': intraday_prob,
                        'order_id': order.id,
                        'estimated_value': estimated_value
                    })
                    
                    print(f"  ✅ BUY order placed! Order ID: {order.id}")
                    
                    # ✅ Update available cash (important for multi-symbol scans)
                    available_cash -= estimated_value
                    print(f"  💰 Remaining cash: ${available_cash:.2f}")
                    
                except ValueError as e:
                    # Pre-validation error (caught before API call)
                    print(f"  ⚠️ Order validation failed: {e}")
                    print(f"  💡 Skipping {sym} - insufficient funds")
                    
                except Exception as e:
                    # Other errors (API issues, etc)
                    print(f"  ❌ Order submission failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            # ============================================================
            # SELL SIGNAL (Low probability)
            # ============================================================
            elif prob <= min_sell_prob:
                print(f"\n  🎯 HIGH CONVICTION SELL SIGNAL!")
                
                # Check if we currently hold this stock
                try:
                    position = api.get_position(sym)
                    current_qty = int(float(position.qty))
                    
                    if current_qty > 0:
                        print(f"  📦 Current position: {current_qty} shares")
                        
                        # ✅ PDT CHECK: Don't sell if position opened today
                        try:
                            opened_today = float(get_opened_today_qty(sym) or 0.0)
                            
                            if opened_today >= current_qty:
                                print(f"  🚫 PDT BLOCK: All {current_qty} shares opened today")
                                print(f"     Cannot sell - would create day trade")
                                print(f"     Will hold until tomorrow")
                                continue
                            elif opened_today > 0:
                                # Partial position opened today
                                sellable_qty = int(current_qty - opened_today)
                                print(f"  ⚠️ PDT WARNING: {int(opened_today)} shares opened today")
                                print(f"     Can only sell {sellable_qty} overnight shares")
                                current_qty = sellable_qty
                                
                                if current_qty <= 0:
                                    print(f"     No sellable shares - skipping")
                                    continue
                                    
                        except Exception as e:
                            print(f"  ⚠️ Could not check PDT status: {e}")
                            print(f"     Proceeding with caution...")
                        
                        print(f"  📋 Preparing SELL order (close position):")
                        print(f"     Quantity: {current_qty} shares")
                        
                        # ✅ NEW: Use session-aware order submission
                        try:
                            order, order_type, limit_price = submit_order_smart(
                                symbol=sym,
                                side='sell',
                                qty=current_qty,
                                current_price=price,
                                limit_buffer_pct=PRE_MARKET_LIMIT_BUFFER
                            )
                            
                            if order_type == 'market':
                                print(f"     Type: MARKET (immediate execution)")
                                estimated_value = current_qty * price
                            else:
                                print(f"     Type: LIMIT @ ${limit_price:.2f}")
                                estimated_value = current_qty * limit_price
                            
                            print(f"     Total Value: ${estimated_value:,.2f}")
                            
                            queued_orders.append({
                                'symbol': sym,
                                'side': 'SELL',
                                'qty': current_qty,
                                'limit_price': limit_price if order_type == 'limit' else None,
                                'order_type': order_type,
                                'probability': prob,
                                'daily_prob': daily_prob,
                                'intraday_prob': intraday_prob,
                                'order_id': order.id,
                                'estimated_value': estimated_value,
                                'action': 'CLOSE'
                            })
                            
                            print(f"  ✅ SELL order placed! Order ID: {order.id}")
                            
                        except Exception as e:
                            print(f"  ❌ Order submission failed: {e}")
                            
                            # Check if it's PDT-related
                            if 'pattern day trading' in str(e).lower():
                                print(f"  🚫 PDT PROTECTION: Cannot sell position opened today")
                            else:
                                import traceback
                                traceback.print_exc()
                                
                    else:
                        # No position - try to short if margin enabled
                        if allow_short:
                            if not is_shortable(sym):
                                print(f"  ⚠️ No position in {sym} and not shortable - skipping")
                                continue
                            
                            # Calculate position size for short
                            short_allocation = PRE_MARKET_SHORT_ALLOCATION
                            allocation = buying_power * short_allocation
                            max_qty = int(allocation // price)
                            
                            if max_qty <= 0:
                                print(f"  ⚠️ Insufficient buying power for even 1 share")
                                continue
                            
                            print(f"  📋 Preparing SHORT order:")
                            print(f"     Quantity: {max_qty} shares")
                            
                            try:
                                order, order_type, limit_price = submit_order_smart(
                                    symbol=sym,
                                    side='sell',
                                    qty=max_qty,
                                    current_price=price,
                                    limit_buffer_pct=PRE_MARKET_LIMIT_BUFFER
                                )
                                
                                if order_type == 'market':
                                    estimated_value = max_qty * price
                                else:
                                    estimated_value = max_qty * limit_price
                                
                                print(f"     Total Value: ${estimated_value:,.2f}")
                                
                                queued_orders.append({
                                    'symbol': sym,
                                    'side': 'SELL',
                                    'qty': max_qty,
                                    'limit_price': limit_price if order_type == 'limit' else None,
                                    'order_type': order_type,
                                    'probability': prob,
                                    'daily_prob': daily_prob,
                                    'intraday_prob': intraday_prob,
                                    'order_id': order.id,
                                    'estimated_value': estimated_value,
                                    'action': 'SHORT'
                                })
                                
                                print(f"  ✅ SHORT order placed! Order ID: {order.id}")
                                
                                # Update buying power
                                buying_power -= estimated_value
                                
                            except Exception as e:
                                print(f"  ❌ Order submission failed: {e}")
                                import traceback
                                traceback.print_exc()
                        else:
                            print(f"  ℹ️ No position in {sym} - nothing to sell")
                            print(f"     (Short selling disabled: need $2,000+ equity)")
                
                except Exception:
                    # Position doesn't exist
                    if allow_short:
                        if not is_shortable(sym):
                            print(f"  ℹ️ No position in {sym} and not shortable - skipping")
                            continue
                        
                        # (Same short logic as above - keep your existing code)
                        # ...
                    else:
                        print(f"  ℹ️ No position in {sym} - nothing to sell")
            
            else:
                # Neutral signal
                print(f"  ℹ️ Probability {prob:.1%} in neutral zone ({min_sell_prob:.0%} - {PRE_MARKET_MIN_PROB:.0%}) - skipping")
                
        except Exception as e:
            print(f"  ❌ Error analyzing {sym}: {e}")
            import traceback
            traceback.print_exc()
    
    # ============================================================
    # SUMMARY (update to show order types)
    # ============================================================
    print(f"\n{'='*80}")
    print(f"📊 SCAN COMPLETE")
    print(f"{'='*80}")
    
    if queued_orders:
        # Count by side and action
        buy_orders = [o for o in queued_orders if o['side'] == 'BUY']
        sell_close_orders = [o for o in queued_orders if o['side'] == 'SELL' and o.get('action') == 'CLOSE']
        sell_short_orders = [o for o in queued_orders if o['side'] == 'SELL' and o.get('action') == 'SHORT']
        
        # ✅ NEW: Show order type breakdown
        market_orders = [o for o in queued_orders if o['order_type'] == 'market']
        limit_orders = [o for o in queued_orders if o['order_type'] == 'limit']
        
        print(f"\n✅ Placed {len(queued_orders)} order(s):")
        print(f"   🟢 {len(buy_orders)} BUY order(s)")
        print(f"   🔴 {len(sell_close_orders)} SELL (close position) order(s)")
        if sell_short_orders:
            print(f"   🔴 {len(sell_short_orders)} SELL (short) order(s)")
        print(f"\n   📋 Order types: {len(market_orders)} MARKET, {len(limit_orders)} LIMIT")
        print()
        
        total_buy_value = 0
        total_sell_value = 0
        
        for order in queued_orders:
            daily = order['daily_prob']
            intraday = order['intraday_prob']
            prob_str = f"D:{daily:.1%} I:{intraday:.1%}" if daily and intraday else f"{order['probability']:.1%}"
            
            side_emoji = "🟢" if order['side'] == 'BUY' else "🔴"
            action_str = order.get('action', '')
            action_label = f" ({action_str})" if action_str else ""
            
            # ✅ NEW: Show order type
            order_type = order['order_type'].upper()
            if order_type == 'LIMIT':
                price_str = f"LIMIT @ ${order['limit_price']:.2f}"
            else:
                price_str = "MARKET"
            
            print(f"  {side_emoji} {order['symbol']}")
            print(f"     Signal: {order['side']}{action_label} (Probability: {order['probability']:.1%})")
            print(f"     Breakdown: {prob_str}")
            print(f"     Order: {order['side']} {order['qty']} - {price_str}")
            print(f"     Value: ${order['estimated_value']:,.2f}")
            print(f"     Order ID: {order['order_id']}\n")
            
            if order['side'] == 'BUY':
                total_buy_value += order['estimated_value']
            else:
                total_sell_value += order['estimated_value']
        
        print(f"  💰 Total BUY Value:  ${total_buy_value:,.2f}")
        if total_sell_value > 0:
            print(f"  💰 Total SELL Value: ${total_sell_value:,.2f}")
        
        # ✅ NEW: Session-specific message
        session = get_market_session()
        if session == 'market_hours':
            print(f"\n⚡ MARKET ORDERS executed immediately")
        else:
            print(f"\n📅 LIMIT ORDERS will execute at market open (9:30 AM ET)")
        
        # Send notification
        notification_msg = f"Pre-Market Scanner queued {len(queued_orders)} order(s):\n\n"
        if buy_orders:
            notification_msg += f"🟢 BUY Orders ({len(buy_orders)}):\n"
            for order in buy_orders:
                notification_msg += f"  {order['symbol']}: {order['qty']} @ ${order['limit_price']:.2f} ({order['probability']:.1%})\n"
        
        if sell_close_orders:
            notification_msg += f"\n🔴 SELL (Close) Orders ({len(sell_close_orders)}):\n"
            for order in sell_close_orders:
                notification_msg += f"  {order['symbol']}: {order['qty']} @ ${order['limit_price']:.2f} ({order['probability']:.1%})\n"
        
        if sell_short_orders:
            notification_msg += f"\n🔴 SELL (Short) Orders ({len(sell_short_orders)}):\n"
            for order in sell_short_orders:
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
