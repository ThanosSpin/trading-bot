# main.py
import time
from data_loader import fetch_historical_data, fetch_latest_price
from model_xgb import load_model, predict_next
from strategy import compute_strategy_decisions
from portfolio import PortfolioManager
from trader import execute_trade, is_market_open, get_pdt_status
from config import SYMBOL, BUY_THRESHOLD, SELL_THRESHOLD

PREDICTION_PERIOD = "6mo"
PREDICTION_INTERVAL = "1d"


# ===============================================================
# Fetch Predictions for All Symbols
# ===============================================================
def get_predictions(symbols):
    predictions = {}

    for sym in symbols:
        df_recent = fetch_historical_data(sym, period=PREDICTION_PERIOD, interval=PREDICTION_INTERVAL)
        if df_recent is None or df_recent.empty:
            print(f"[WARN] No recent data for {sym}, skipping.")
            continue

        model = load_model(sym)
        if model is None:
            print(f"[ERROR] No model found for {sym}, skipping.")
            continue

        prob_up = predict_next(df_recent, model)
        if prob_up is None:
            print(f"[WARN] Invalid prediction for {sym}, skipping.")
            continue

        predictions[sym] = prob_up

    return predictions


# ===============================================================
# Execute the Computed Decisions
# ===============================================================
def execute_decisions(decisions):
    """
    decisions = {
        "NVDA": ("sell", 10),
        "AAPL": ("buy", 12)
    }
    """

    for sym, (action, qty) in decisions.items():
        pm = PortfolioManager(sym)
        price = fetch_latest_price(sym)

        print(f"\n--- {sym} Decision ---")
        print(f"Action: {action.upper()} | Qty: {qty} | Price: {price}")

        if price is None:
            print(f"[WARN] No price for {sym}, skipping.")
            continue

        # Skip HOLD
        if action == "hold" or qty <= 0:
            print("[INFO] Action: HOLD")
            continue

        # Execute order via Alpaca
        filled_qty, filled_price = execute_trade(action, qty, sym)

        if not filled_qty:
            print(f"[WARN] {sym} trade not filled.")
            continue

        # Update local portfolio tracking
        pm.refresh_live()  # Sync fresh balance before applying local update
        pm._apply(action, filled_price, filled_qty)

        print(f"Updated Portfolio Value for {sym}: ${pm.value():.2f}")


# ===============================================================
# Main Orchestration Logic
# ===============================================================
def process_all_symbols(symbols):
    # Step 1: Get predictions
    predictions = get_predictions(symbols)

    if not predictions:
        print("[INFO] No valid predictions available.")
        return

    # Step 2: Compute strategy decisions
    decisions = compute_strategy_decisions(predictions)

    # Print summary
    print("\n================== DECISIONS ==================")
    for sym, (action, qty) in decisions.items():
        print(f"{sym}: {action.upper()} {qty}")
    print("================================================\n")

    # Step 3: Execute trades
    execute_decisions(decisions)


# ===============================================================
# Program Entry Point
# ===============================================================
def main():
    if not is_market_open():
        print("⏳ Market is closed. Exiting.")
        return

    symbols = SYMBOL if isinstance(SYMBOL, list) else [SYMBOL]

    # Print PDT status
    try:
        pdt = get_pdt_status()
        if pdt:
            print("\n📊 PDT Account Status:")
            print(f"Equity: ${pdt['equity']:.2f}")
            print(f"Day Trades (5-day): {pdt['daytrade_count']}")
            print(f"Remaining: {pdt['remaining']}")
            print("--------------------------------------")
    except:
        pass

    process_all_symbols(symbols)


if __name__ == "__main__":
    main()