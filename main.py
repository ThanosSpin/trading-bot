# main.py (updated to coordinate multi-symbol signals before deciding allocations)
from data_loader import fetch_historical_data, fetch_latest_price
from model_xgb import load_model, predict_next
from strategy import should_trade
from portfolio import load_portfolio, update_portfolio, save_portfolio, portfolio_value, get_live_portfolio
from trader import execute_trade, is_market_open, get_pdt_status
from config import SYMBOL  # SYMBOL can be a string or list

PREDICTION_PERIOD = "6mo"
PREDICTION_INTERVAL = "1d"


def process_all_symbols(symbols):
    """
    For multi-symbol support: compute probabilities for all symbols first,
    decide which symbols are BUY candidates, then call should_trade with
    concurrent_buys so allocation rules can behave as you requested.
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

    # Step 2: determine buy candidates (based on BUY_THRESHOLD in strategy via should_trade)
    # We'll do a cheap pre-filter: consider candidate a buy if prob_up >= BUY_THRESHOLD
    # Import BUY_THRESHOLD dynamically to avoid circular import issues
    from config import BUY_THRESHOLD, SELL_THRESHOLD

    buy_candidates = [s for s, p in probs.items() if p >= BUY_THRESHOLD]
    sell_candidates = [s for s, p in probs.items() if p <= SELL_THRESHOLD]

    total_symbols = len(symbols)
    concurrent_buys = len(buy_candidates)

    # Step 3: run should_trade for each symbol with the concurrent_buys info
    for sym, prob_up in probs.items():
        action, quantity = should_trade(sym, prob_up, total_symbols=total_symbols, concurrent_buys=concurrent_buys)
        print(f"{sym} → Prediction: {prob_up:.2f}, Action: {action.upper()} {quantity}")

        # Show current live portfolio for debugging
        try:
            live = get_live_portfolio(sym)
            print(f"[INFO] Current Portfolio for {sym}: Cash=${live['cash']:.2f}, Shares={live['shares']}, Value=${live['cash'] + live['shares']*live['last_price']:.2f}")
        except Exception as e:
            print(f"[WARN] Could not fetch live portfolio for {sym}: {e}")
            live = load_portfolio(sym)

        # Fetch latest price
        price = fetch_latest_price(sym)
        if price is None:
            print(f"[WARN] Could not fetch latest price for {sym}. Skipping trade.")
            continue

        # Execute trade if requested
        if action in ["buy", "sell"] and quantity > 0:
            filled_qty, filled_price = execute_trade(action, quantity, sym)
            if filled_qty and filled_price:
                # update_portfolio keeps backward compatibility: it expects action, price, portfolio, symbol
                # We pass the live/local portfolio we used before
                # Note: update_portfolio may assume qty=1 per trade in your code; if you're trading different qtys,
                # you may need to extend update_portfolio to accept qty and update cash/shares appropriately.
                portfolio = load_portfolio(sym)
                portfolio = update_portfolio(action, filled_price, portfolio, sym)
                save_portfolio(portfolio, sym)
                print(f"✅ Updated Portfolio Value for {sym}: ${portfolio_value(portfolio):.2f}")
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