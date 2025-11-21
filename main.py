# main.py
import time
from data_loader import fetch_historical_data, fetch_latest_price
from market import is_market_open, debug_market
from model_xgb import compute_signals
from strategy import compute_strategy_decisions
from portfolio import PortfolioManager
from trader import execute_trade, get_pdt_status
from config import SYMBOL, BUY_THRESHOLD, SELL_THRESHOLD, INTRADAY_WEIGHT
from market import is_market_open, is_trading_day
import os

os.environ["TZ"] = "America/New_York"
time.tzset()

# ===============================================================
# Fetch Predictions for All Symbols (USING compute_signals)
# ===============================================================
def get_predictions(symbols, debug=True):
    predictions = {}

    for sym in symbols:
        print(f"\n🔍 Fetching signals for {sym}...")

        sig = compute_signals(
            sym,
            lookback_minutes=180,
            intraday_weight=INTRADAY_WEIGHT,
            resample_to="5min",
        )

        if debug:
            print(f"\n[DEBUG] {sym} Signals Summary")
            print("------------------------------------")
            print(f"Daily rows:         {sig.get('daily_rows')}")
            print(f"Intraday rows:      {sig.get('intraday_rows')}  "
                  f"(before trim: {sig.get('intraday_before')})")
            print(f"Daily prob:         {sig.get('daily_prob')}")
            print(f"Intraday prob:      {sig.get('intraday_prob')}")
            print(f"Final combined prob:{sig.get('final_prob')}")
            print("------------------------------------\n")

        final_prob = sig.get("final_prob")
        if final_prob is None:
            print(f"[WARN] Invalid prediction for {sym}, skipping.")
            continue

        predictions[sym] = final_prob

    return predictions


# ===============================================================
# Execute decisions (clean & safe)
# ===============================================================
def execute_decisions(decisions):
    """
    decisions = {
        "AAPL": {"action": "buy", "qty": 3, "explain": "..."},
        "NVDA": {"action": "sell", "qty": 1, "explain": "..."}
    }
    """

    for sym, decision in decisions.items():

        action = decision.get("action", "hold")
        qty = decision.get("qty", 0)
        explain = decision.get("explain", "")
        pm = PortfolioManager(sym)
        price = fetch_latest_price(sym)

        print(f"\n--- {sym} Decision ---")
        print(f"Action: {action.upper()} | Qty: {qty} | Price: {price}")
        print(f"Reason: {explain}")

        if price is None:
            print(f"[WARN] No price for {sym}, skipping.")
            continue

        if action == "hold" or qty <= 0:
            print("[INFO] HOLD — No trade executed.")
            continue

        # Execute trade (LIVE or SIM depending on config)
        filled_qty, filled_price = execute_trade(action, qty, sym)

        if not filled_qty:
            print(f"[WARN] {sym} trade NOT filled.")
            continue

        # Sync portfolio and apply changes
        pm.refresh_live()
        pm._apply(action, filled_price, filled_qty)

        print(f"Updated Portfolio Value for {sym}: ${pm.value():.2f}")


# ===============================================================
# Orchestrator
# ===============================================================
def process_all_symbols(symbols):
    # Step 1: Predictions
    predictions = get_predictions(symbols, debug=True)

    if not predictions:
        print("[INFO] No valid predictions available. Stopping.")
        return

    # Step 2: Strategy logic
    decisions = compute_strategy_decisions(predictions)

    print("\n================== DECISIONS ==================")
    for sym, d in decisions.items():
        print(f"{sym}: {d['action'].upper()} {d['qty']} — {d['explain']}")
    print("================================================\n")

    # Step 3: Execute trades
    execute_decisions(decisions)


# ===============================================================
# Entry Point
# ===============================================================
def main():
    symbols = SYMBOL if isinstance(SYMBOL, list) else [SYMBOL]

    # Always show market diagnostics first
    print("\n🔧 Running market diagnostics...")
    debug_market()

    # Optional market-hours guard
    if not is_market_open():
        print("⏳ Market is closed. Exiting.")
        return

    # PDT Display
    try:
        pdt = get_pdt_status()
        if pdt:
            print("\n📊 PDT Account Status:")
            print(f"Equity: ${pdt['equity']:.2f}")
            print(f"Day Trades (5-day): {pdt['daytrade_count']}")
            print(f"Remaining: {pdt['remaining']}")
            print(f"Flagged: {pdt['is_pdt']}")
            print("--------------------------------------")
    except Exception:
        pass

    process_all_symbols(symbols)


if __name__ == "__main__":
    main()