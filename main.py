# main.py
from data_loader import fetch_historical_data, fetch_latest_price
from model_xgb import load_model, predict_next
from strategy import should_trade
from portfolio import PortfolioManager
from trader import execute_trade, is_market_open, get_pdt_status
from config import SYMBOL  # Can be string or list
from config import BUY_THRESHOLD, SELL_THRESHOLD

PREDICTION_PERIOD = "6mo"
PREDICTION_INTERVAL = "1d"


def process_all_symbols(symbols):
    """
    Multi-symbol orchestration using PortfolioManager directly.
    """

    # Step 1: gather predictions
    probs = {}
    models = {}
    for sym in symbols:
        df_recent = fetch_historical_data(sym, period=PREDICTION_PERIOD, interval=PREDICTION_INTERVAL)
        if df_recent is None or df_recent.empty:
            print(f"[WARN] Recent data missing for {sym}. Skipping prediction.")
            continue

        model = load_model(sym)
        if model is None:
            print(f"[ERROR] No model found for {sym}. Skipping.")
            continue

        prob_up = predict_next(df_recent, model)
        if prob_up is None:
            print(f"[WARN] Prediction invalid for {sym}. Skipping.")
            continue

        probs[sym] = prob_up
        models[sym] = model

    if not probs:
        print("[INFO] No valid predictions available.")
        return

    # Step 2: determine buy/sell candidates
    buy_candidates = [s for s, p in probs.items() if p >= BUY_THRESHOLD]
    sell_candidates = [s for s, p in probs.items() if p <= SELL_THRESHOLD]

    total_symbols = len(symbols)
    concurrent_buys = len(buy_candidates)

    # Step 3: process each symbol
    for sym, prob_up in probs.items():
        pm = PortfolioManager(sym)  # Direct class usage
        action, quantity = should_trade(sym, prob_up, total_symbols=total_symbols, concurrent_buys=concurrent_buys)

        print(f"{sym} → Prediction: {prob_up:.2f}, Action: {action.upper()} {quantity}")
        print(f"[INFO] Current Portfolio for {sym}: Cash=${pm.data['cash']:.2f}, Shares={pm.data['shares']}, Value=${pm.value():.2f}")

        # Fetch latest price
        price = fetch_latest_price(sym)
        if price is None:
            print(f"[WARN] Could not fetch latest price for {sym}. Skipping trade.")
            continue

        # Execute trade if requested
        if action in ["buy", "sell"] and quantity > 0:
            filled_qty, filled_price = execute_trade(action, quantity, sym)
            if filled_qty and filled_price:
                pm.update(action, filled_price, filled_qty)
                print(f"✅ Updated Portfolio Value for {sym}: ${pm.value():.2f}")
            else:
                print(f"[WARN] Order for {sym} did not fill immediately (or was blocked).")
        else:
            print(f"[INFO] No action taken for {sym}.")


def main():
    if not is_market_open():
        print("⏳ Market is closed. Skipping trades.")
        return

    symbols = SYMBOL if isinstance(SYMBOL, list) else [SYMBOL]

    # Print PDT status
    try:
        pdt_info = get_pdt_status()
        if pdt_info:
            print("\n📊 PDT Account Status:")
            print(f"   Equity: ${pdt_info['equity']:.2f}")
            print(f"   Day Trades (5-day window): {pdt_info['daytrade_count']}")
            print(f"   Remaining Day Trades: {pdt_info['remaining']}")
            print("-" * 40)
    except Exception:
        pass

    process_all_symbols(symbols)


if __name__ == "__main__":
    main()