import alpaca_trade_api as tradeapi
from config import API_KEY, API_SECRET, BASE_URL, SYMBOL

api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL)

def place_order(action):
    if action == 'buy':
        api.submit_order(symbol=SYMBOL, qty=1, side='buy', type='market', time_in_force='gtc')
    elif action == 'sell':
        api.submit_order(symbol=SYMBOL, qty=1, side='sell', type='market', time_in_force='gtc')
