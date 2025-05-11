# trader.py
import alpaca_trade_api as tradeapi
from config import API_KEY, API_SECRET, BASE_URL, USE_LIVE_TRADING, SYMBOL
from market import is_market_open

api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')


import alpaca_trade_api as tradeapi
from datetime import datetime
from config import API_KEY, API_SECRET, BASE_URL, USE_LIVE_TRADING, SYMBOL

# Alpaca API setup
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# Function to execute a trade
def execute_trade(action, quantity=1):
    if not is_market_open():
        print("Market is closed. Skipping this trade.")
        return

    if not USE_LIVE_TRADING:
        print(f"[SIMULATION] {action.upper()} {quantity} share(s) of {SYMBOL}")
        return

    # Execute the trade on Alpaca if market is open and live trading is enabled
    if action == "buy":
        api.submit_order(symbol=SYMBOL, qty=quantity, side='buy', type='market', time_in_force='gtc')
    elif action == "sell":
        api.submit_order(symbol=SYMBOL, qty=quantity, side='sell', type='market', time_in_force='gtc')

    print(f"[LIVE] {action.upper()} {quantity} share(s) of {SYMBOL} sent to market")

# Example of running the trade logic
if __name__ == "__main__":
    # Example trade actions
    execute_trade("buy", 1)  # Simulate buying 1 share
    execute_trade("sell", 1)  # Simulate selling 1 share
