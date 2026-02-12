# pdt_tracker.py
# ✅ THREAD-SAFE VERSION - Fixes race condition in file I/O
import json
import os
import threading
from datetime import datetime, timedelta
import pytz


NY_TZ = pytz.timezone("America/New_York")
STATE_FILE = "pdt_state.json"

# ✅ Global lock for thread safety
_state_lock = threading.Lock()


def _today_key():
    """Get today's date key in NY timezone."""
    return datetime.now(NY_TZ).strftime("%Y-%m-%d")


def _load_state_unsafe():
    """Load PDT state from file (NO LOCK - internal use only)."""
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load PDT state: {e}")
        return {}


def _save_state_unsafe(state):
    """Save PDT state to file (NO LOCK - internal use only)."""
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, sort_keys=True)
    except Exception as e:
        print(f"[ERROR] Failed to save PDT state: {e}")


def load_state():
    """Load PDT state from file (thread-safe)."""
    with _state_lock:
        return _load_state_unsafe()


def save_state(state):
    """Save PDT state to file (thread-safe)."""
    with _state_lock:
        _save_state_unsafe(state)


def get_opened_today_qty(symbol: str) -> float:
    """
    Get quantity opened today for a symbol (thread-safe).

    Args:
        symbol: Stock symbol (e.g., "NVDA")

    Returns:
        Number of shares opened today (0.0 if none)
    """
    symbol = str(symbol).upper().strip()

    with _state_lock:
        state = _load_state_unsafe()  # ✅ Use unsafe version inside lock
        day = _today_key()
        return float(state.get(day, {}).get(symbol, 0.0) or 0.0)


def add_opened_today(symbol: str, qty: float):
    """
    Record shares opened today (BUY transaction).

    Args:
        symbol: Stock symbol
        qty: Number of shares bought
    """
    if qty <= 0:
        return

    symbol = str(symbol).upper().strip()
    qty = float(qty)

    with _state_lock:  # ✅ Single lock for entire operation
        state = _load_state_unsafe()
        day = _today_key()

        # Ensure day exists
        if day not in state:
            state[day] = {}

        # Add to existing or create new
        current = float(state[day].get(symbol, 0.0) or 0.0)
        state[day][symbol] = current + qty

        _save_state_unsafe(state)

        print(f"[PDT TRACKER] {symbol}: +{qty:g} shares opened today (total: {state[day][symbol]:g})")


def reduce_opened_today(symbol: str, qty: float):
    """
    Record shares closed today (SELL transaction).

    Args:
        symbol: Stock symbol
        qty: Number of shares sold
    """
    if qty <= 0:
        return

    symbol = str(symbol).upper().strip()
    qty = float(qty)

    with _state_lock:  # ✅ Single lock for entire operation
        state = _load_state_unsafe()
        day = _today_key()

        # Ensure day exists
        if day not in state:
            state[day] = {}

        # Reduce (can't go below 0)
        current = float(state[day].get(symbol, 0.0) or 0.0)
        new_qty = max(0.0, current - qty)
        state[day][symbol] = new_qty

        _save_state_unsafe(state)

        print(f"[PDT TRACKER] {symbol}: -{qty:g} shares closed today (remaining: {new_qty:g})")


def reset_symbol(symbol: str):
    """
    Manually reset opened_today count for a symbol (emergency use).

    Args:
        symbol: Stock symbol to reset
    """
    symbol = str(symbol).upper().strip()

    with _state_lock:
        state = _load_state_unsafe()
        day = _today_key()

        if day in state and symbol in state[day]:
            old_qty = state[day][symbol]
            del state[day][symbol]
            _save_state_unsafe(state)
            print(f"[PDT TRACKER] {symbol}: RESET (was {old_qty:g})")
        else:
            print(f"[PDT TRACKER] {symbol}: Already at 0")


def cleanup_old_days(keep_days: int = 7):
    """
    Remove old date entries from state file (cleanup).

    Args:
        keep_days: Number of recent days to keep (default 7)
    """
    with _state_lock:
        state = _load_state_unsafe()

        today = datetime.now(NY_TZ).date()
        cutoff = (today - timedelta(days=keep_days)).strftime("%Y-%m-%d")

        old_keys = [k for k in state.keys() if k < cutoff]

        if old_keys:
            for key in old_keys:
                del state[key]
            _save_state_unsafe(state)
            print(f"[PDT TRACKER] Cleaned up {len(old_keys)} old day(s)")


def get_all_today() -> dict:
    """
    Get all symbols opened today (debugging/dashboard).

    Returns:
        Dict of {symbol: qty} for today
    """
    with _state_lock:
        state = _load_state_unsafe()
        day = _today_key()
        return state.get(day, {})


# ✅ TESTING FUNCTION
def test_thread_safety():
    """
    Test concurrent operations to verify thread safety.
    Run this after implementing the fix.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor

    print("\n" + "="*60)
    print("THREAD SAFETY TEST")
    print("="*60)

    # Reset test symbol
    test_sym = "TEST"
    reset_symbol(test_sym)

    # Simulate 10 concurrent BUYs of 10 shares each
    def buy_shares():
        add_opened_today(test_sym, 10)
        time.sleep(0.001)  # Simulate processing delay

    print("Running 10 concurrent BUY operations...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(buy_shares) for _ in range(10)]
        for f in futures:
            f.result()

    # Check result
    final_qty = get_opened_today_qty(test_sym)
    expected = 100.0

    if abs(final_qty - expected) < 0.01:
        print(f"✅ PASS: Final quantity = {final_qty} (expected {expected})")
    else:
        print(f"❌ FAIL: Final quantity = {final_qty} (expected {expected})")
        print(f"   Lost {expected - final_qty} shares due to race condition!")

    # Cleanup
    reset_symbol(test_sym)
    print("="*60)


if __name__ == "__main__":
    # Demo usage
    print("\n" + "="*60)
    print("PDT TRACKER DEMO")
    print("="*60)

    # Simulate a trading day
    print("\nSimulating trades...")
    add_opened_today("NVDA", 50)
    add_opened_today("AAPL", 100)
    add_opened_today("NVDA", 25)  # Buy more NVDA

    print(f"\nCurrent opened today:")
    for sym, qty in get_all_today().items():
        print(f"  {sym}: {qty:g} shares")

    # Simulate selling
    print("\nSimulating sells...")
    reduce_opened_today("NVDA", 30)
    reduce_opened_today("AAPL", 50)

    print(f"\nAfter partial sells:")
    for sym, qty in get_all_today().items():
        print(f"  {sym}: {qty:g} shares")

    # Test thread safety
    test_thread_safety()

    print("\n" + "="*60)
    print("✅ ALL TESTS COMPLETE")
    print("="*60)
    