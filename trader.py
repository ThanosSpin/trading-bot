# trader.py
import alpaca_trade_api as tradeapi
from config import API_KEY, API_SECRET, BASE_URL, USE_LIVE_TRADING, SYMBOL

api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')


def execute_trade(action, quantity=1):
    if not USE_LIVE_TRADING:
        print(f"[SIMULATION] {action.upper()} {quantity} share(s) of {SYMBOL}")
        return

    if action == "buy":
        api.submit_order(symbol=SYMBOL, qty=quantity, side='buy', type='market', time_in_force='gtc')
    elif action == "sell":
        api.submit_order(symbol=SYMBOL, qty=quantity, side='sell', type='market', time_in_force='gtc')
    print(f"[LIVE] {action.upper()} {quantity} share(s) of {SYMBOL} sent to market")


if __name__ == "__main__":
    execute_trade("buy")
    execute_trade("sell")
