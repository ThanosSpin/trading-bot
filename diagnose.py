import sys; sys.path.insert(0, '.')
from update_portfolio_data import trade_log_path
from config import SYMBOL, SPY_SYMBOL
symbols = SYMBOL if isinstance(SYMBOL, list) else [SYMBOL]
symbols = [s.upper() for s in symbols]
symbols.append(SPY_SYMBOL.upper())
import os
for s in symbols:
    p = trade_log_path(s)
    exists = os.path.exists(p)
    size = os.path.getsize(p) if exists else 0
    print(f'{s}: {p} | exists={exists} | size={size}')