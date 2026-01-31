# account_cache.py
from typing import Optional, Dict
from datetime import datetime, timedelta
from broker import api_market

class AccountCache:
    """Singleton cache for Alpaca account state to reduce API calls."""
    
    _instance = None
    _cache: Optional[Dict] = None
    _cache_time: Optional[datetime] = None
    _ttl_seconds = 5  # Cache expires after 5 seconds
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_account(self, force_refresh: bool = False):
        """Get cached account or fetch fresh if expired."""
        now = datetime.utcnow()
        
        # Return cache if valid
        if (not force_refresh and 
            self._cache is not None and 
            self._cache_time is not None and
            (now - self._cache_time).total_seconds() < self._ttl_seconds):
            return self._cache
        
        # Fetch fresh data
        try:
            account = api_market.get_account()
            positions = api_market.list_positions()
            
            self._cache = {
                "account": account,
                "positions": positions,
                "cash": float(account.cash),
                "equity": float(account.equity),
                "buying_power": float(account.buying_power),
                "daytrade_count": int(account.daytrade_count or 0),
                "pattern_day_trader": bool(getattr(account, "pattern_day_trader", False)),
                "trading_blocked": bool(getattr(account, "trading_blocked", False)),
            }
            self._cache_time = now
            
            print(f"[CACHE] Account refreshed: cash=${self._cache['cash']:.2f}, "
                  f"equity=${self._cache['equity']:.2f}")
            
            return self._cache
            
        except Exception as e:
            print(f"[WARN] Account cache refresh failed: {e}")
            # Return stale cache if available
            return self._cache if self._cache else {}
    
    def invalidate(self):
        """Force refresh on next get_account() call."""
        self._cache_time = None
    
    def get_position(self, symbol: str):
        """Get position for a specific symbol from cache."""
        cache = self.get_account()
        if not cache or "positions" not in cache:
            return None
        
        symbol = symbol.upper()
        for pos in cache["positions"]:
            if str(getattr(pos, "symbol", "")).upper() == symbol:
                return pos
        return None

# Global singleton instance
account_cache = AccountCache()
