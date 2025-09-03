import time
import alpaca_trade_api as tradeapi
from config import API_MARKET_KEY, API_MARKET_SECRET, MARKET_BASE_URL, USE_LIVE_TRADING
from market import is_market_open, is_trading_day
from model import load_model, predict_next
from strategy import should_trade
from portfolio import load_portfolio

# Initialize Alpaca API
api = tradeapi.REST(API_MARKET_KEY, API_MARKET_SECRET, MARKET_BASE_URL, api_version='v2')


def execute_trade(action, quantity, symbol):
    if not is_trading_day() or not is_market_open():
        print(f"⏳ Market is closed or it's a holiday. Skipping trade for {symbol}.")
        return

    if not USE_LIVE_TRADING:
        print(f"[SIMULATION] {action.upper()} {quantity} share(s) of {symbol}")
        return

    try:
        order = api.submit_order(
            symbol=symbol,
            qty=quantity,
            side=action,
            type='market',
            time_in_force='gtc'
        )

        print(f"[LIVE] {action.upper()} order submitted for {symbol}: ID={order.id}, status={order.status}")
        time.sleep(2)
        order_result = api.get_order(order.id)

        print(f"[LIVE] Order status for {symbol}: {order_result.status}")
        if order_result.status == "filled":
            print(f"✅ Order filled at ${order_result.filled_avg_price} "
                  f"for {order_result.filled_qty} share(s) of {symbol}.")
        elif order_result.status == "rejected":
            print(f"❌ Order rejected for {symbol}. Reason: {order_result.fail_reason or 'Not specified'}")
        else:
            print(f"ℹ️ Order for {symbol} still pending or partially filled.")
    except Exception as e:
        print(f"[ERROR] Failed to execute trade for {symbol}: {e}")


def run_trading_bot(symbols):
    """
    symbols: list of stock symbols (or single symbol as string)
    """
    if isinstance(symbols, str):
        symbols = [symbols]

    for symbol in symbols:
        # Load portfolio
        portfolio = load_portfolio(symbol)

        # Load model
        model = load_model(symbol)
        if model is None:
            print(f"[ERROR] No model found for {symbol}. Skipping.")
            continue

        # Load historical data and predict
        from data_loader import fetch_historical_data  # import locally to avoid circular
        df = fetch_historical_data(symbol)
        if df is None or df.empty:
            print(f"[ERROR] No historical data for {symbol}. Skipping.")
            continue

        prob_up = predict_next(df, model)
        if prob_up is None:
            print(f"[INFO] Invalid prediction for {symbol}. Skipping.")
            continue

        # Decide action
        action, quantity = should_trade(symbol, prob_up, total_symbols=len(symbols))
        print(f"{symbol} → Prediction: {prob_up:.2f}, Action: {action.upper()} {quantity}")

        if action in ["buy", "sell"] and quantity > 0:
            execute_trade(action, quantity, symbol)
        else:
            print(f"[INFO] No trade executed for {symbol}.")


if __name__ == "__main__":
    # Example usage with a single or multiple symbols
    from config import SYMBOL
    run_trading_bot(SYMBOL)