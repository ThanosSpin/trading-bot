"""
Order utility functions for pre-market vs market hours trading
"""

from datetime import datetime, time
from config import LIMIT_BUFFER_PCT
import pytz


def is_market_hours(market_tz='America/New_York'):
    """
    Check if currently in regular market hours (9:30 AM - 4:00 PM ET)
    
    Returns:
        bool: True if market is open, False otherwise
    """
    tz = pytz.timezone(market_tz)
    now = datetime.now(tz)
    
    # Check if weekend
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    
    # Market hours: 9:30 AM - 4:00 PM ET
    market_open = time(9, 30)
    market_close = time(16, 0)
    
    current_time = now.time()
    
    return market_open <= current_time <= market_close


def is_pre_market(market_tz='America/New_York'):
    """
    Check if currently in pre-market hours (4:00 AM - 9:30 AM ET)
    
    Returns:
        bool: True if pre-market, False otherwise
    """
    tz = pytz.timezone(market_tz)
    now = datetime.now(tz)
    
    # Check if weekend
    if now.weekday() >= 5:
        return False
    
    # Pre-market hours: 4:00 AM - 9:30 AM ET
    pre_market_open = time(4, 0)
    market_open = time(9, 30)
    
    current_time = now.time()
    
    return pre_market_open <= current_time < market_open


def is_after_hours(market_tz='America/New_York'):
    """
    Check if currently in after-hours (4:00 PM - 8:00 PM ET)
    
    Returns:
        bool: True if after-hours, False otherwise
    """
    tz = pytz.timezone(market_tz)
    now = datetime.now(tz)
    
    # Check if weekend
    if now.weekday() >= 5:
        return False
    
    # After-hours: 4:00 PM - 8:00 PM ET
    market_close = time(16, 0)
    after_hours_close = time(20, 0)
    
    current_time = now.time()
    
    return market_close < current_time <= after_hours_close


def get_order_params(symbol, side, qty, current_price, limit_buffer_pct=LIMIT_BUFFER_PCT, refresh_price=False):
    """
    Generate order parameters based on current market session
    
    Args:
        symbol: Stock symbol
        side: 'buy' or 'sell'
        qty: Number of shares
        current_price: Current stock price
        limit_buffer_pct: Buffer for limit orders (default 2%)
        refresh_price: If True, fetch latest price (default False, handled by caller)
    
    Returns:
        dict: Order parameters ready for api.submit_order()
    """
    
    # ✅ NOTE: refresh_price is handled in submit_order_smart() 
    # This parameter exists for compatibility but isn't used here
    # Price should already be refreshed before calling this function
    
    # Base parameters
    params = {
        'symbol': symbol,
        'qty': qty,
        'side': side,
        'time_in_force': 'day',
    }
    
    # Determine order type based on session
    if is_market_hours():
        # MARKET HOURS: Use market orders for immediate execution
        params['type'] = 'market'
        print(f"[ORDER] {symbol} {side.upper()} {qty} - MARKET ORDER (market hours)")
        print(f"        Current price: ${current_price:.2f}")
        
    elif is_pre_market() or is_after_hours():
        # PRE/AFTER MARKET: Use limit orders with ±buffer
        params['type'] = 'limit'
        params['extended_hours'] = True
        
        # Calculate limit price
        if side == 'buy':
            # Buy: +buffer above current (willing to pay slightly more)
            limit_price = round(current_price * (1 + limit_buffer_pct), 2)
        else:  # sell
            # Sell: -buffer below current (willing to accept slightly less)
            limit_price = round(current_price * (1 - limit_buffer_pct), 2)
        
        params['limit_price'] = limit_price
        
        session = "PRE-MARKET" if is_pre_market() else "AFTER-HOURS"
        print(f"[ORDER] {symbol} {side.upper()} {qty} - LIMIT @ ${limit_price} ({session})")
        print(f"        Current: ${current_price:.2f} | Buffer: {limit_buffer_pct*100:.1f}%")
        
    else:
        # CLOSED: Use limit orders without extended hours
        params['type'] = 'limit'
        params['extended_hours'] = False
        
        if side == 'buy':
            limit_price = round(current_price * (1 + limit_buffer_pct), 2)
        else:
            limit_price = round(current_price * (1 - limit_buffer_pct), 2)
        
        params['limit_price'] = limit_price
        
        print(f"[ORDER] {symbol} {side.upper()} {qty} - LIMIT @ ${limit_price} (QUEUED for open)")
        print(f"        Current: ${current_price:.2f} | Market closed | Will execute at open")
    
    return params




def get_market_session():
    """
    Get current market session name
    
    Returns:
        str: 'pre_market', 'market_hours', 'after_hours', or 'closed'
    """
    if is_pre_market():
        return 'pre_market'
    elif is_market_hours():
        return 'market_hours'
    elif is_after_hours():
        return 'after_hours'
    else:
        return 'closed'


def print_market_status():
    """Print current market session status"""
    session = get_market_session()
    
    tz = pytz.timezone('America/New_York')
    now = datetime.now(tz)
    
    print("="*60)
    print("MARKET SESSION STATUS")
    print("="*60)
    print(f"Current time (ET): {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Session: {session.upper().replace('_', ' ')}")
    
    if session == 'pre_market':
        print("⏰ Pre-market active (4:00 AM - 9:30 AM ET)")
        print("📋 Using: LIMIT orders with ±2% buffer")
    elif session == 'market_hours':
        print("✅ Market OPEN (9:30 AM - 4:00 PM ET)")
        print("📋 Using: MARKET orders for immediate fill")
    elif session == 'after_hours':
        print("🌙 After-hours active (4:00 PM - 8:00 PM ET)")
        print("📋 Using: LIMIT orders with ±2% buffer")
    else:
        print("🔒 Market CLOSED")
        print("📋 Using: LIMIT orders (will execute at open)")
    
    print("="*60)
