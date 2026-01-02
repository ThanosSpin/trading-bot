# main.py
import time
from data_loader import fetch_historical_data, fetch_latest_price
from market import is_market_open, debug_market
from model_xgb import compute_signals
from strategy import compute_strategy_decisions
from portfolio import PortfolioManager
from trader import execute_trade, get_pdt_status
from config import (
    SYMBOL, BUY_THRESHOLD, SELL_THRESHOLD, INTRADAY_WEIGHT,
    SPY_SYMBOL, WEAK_PROB_THRESHOLD, WEAK_RATIO_THRESHOLD,
    SPY_ENTRY_THRESHOLD, SPY_EXIT_THRESHOLD, SPY_MUTUAL_EXCLUSIVE
)
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
            lookback_minutes=900,
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
    Execute SELLs first (to free capital), then BUYs.

    Supports flags from strategy.py:
      - recalc_all_in=True        -> recompute qty from live cash after SELLs
      - recalc_after_sells=True   -> recompute qty using remaining cash after higher-priority buys
      - priority_rank=int         -> lower rank executes first (1 before 2)
    """

    def _get_live_account_cash(proxy_symbol: str = "NVDA") -> float:
        # cash is account-level; any symbol PM works, proxy_symbol is just a handle
        try:
            pm_cash = PortfolioManager(proxy_symbol)
            pm_cash.refresh_live()
            return float(pm_cash.data.get("cash", 0.0))
        except Exception:
            # fallback to SPY proxy
            try:
                pm_cash = PortfolioManager("SPY")
                pm_cash.refresh_live()
                return float(pm_cash.data.get("cash", 0.0))
            except Exception:
                return 0.0

    sell_syms = [s for s, d in decisions.items()
                if d.get("action") == "sell" and int(d.get("qty", 0)) > 0]

    buy_syms = [s for s, d in decisions.items()
               if d.get("action") == "buy" and int(d.get("qty", 0)) > 0]

    hold_syms = [s for s, d in decisions.items()
                if d.get("action") not in ("buy", "sell") or int(d.get("qty", 0)) <= 0]

    any_sell_filled = False
    sell_failed = set()

    # -------------------------
    # PASS 1: SELLs
    # -------------------------
    for sym in sell_syms:
        decision = decisions[sym]
        qty = int(decision.get("qty", 0))
        explain = decision.get("explain", "")

        pm = PortfolioManager(sym)
        price = fetch_latest_price(sym)

        print(f"\n--- {sym} Decision ---")
        print(f"Action: SELL | Qty: {qty} | Price: {price}")
        print(f"Reason: {explain}")

        if price is None or qty <= 0:
            print(f"[WARN] {sym} invalid sell input, skipping.")
            sell_failed.add(sym)
            continue

        filled_qty, filled_price = execute_trade("sell", qty, sym)

        if not filled_qty:
            print(f"[WARN] {sym} SELL not filled.")
            sell_failed.add(sym)
            continue

        any_sell_filled = True
        pm.refresh_live()
        pm._apply("sell", filled_price, filled_qty)
        print(f"Updated Portfolio Value for {sym}: ${pm.value():.2f}")

    # -------------------------
    # Refresh cash after SELLs
    # -------------------------
    global_cash_after_sells = _get_live_account_cash(proxy_symbol="NVDA")
    remaining_cash = global_cash_after_sells

    # If strategy required SPY liquidation to fund core and it failed, we should not "all-in" core.
    spy_required_sell = ("SPY" in decisions and decisions["SPY"].get("action") == "sell")
    spy_sell_failed = (spy_required_sell and "SPY" in sell_failed)

    # -------------------------
    # PASS 2: BUYs (priority order)
    # -------------------------
    def _buy_sort_key(sym: str):
        d = decisions.get(sym, {})
        # priority_rank: 1 executes before 2, etc.
        # if missing, default to 999 so it runs after priority buys
        return int(d.get("priority_rank", 999))

    buy_syms_sorted = sorted(buy_syms, key=_buy_sort_key)

    for sym in buy_syms_sorted:
        decision = decisions[sym]
        explain = (decision.get("explain", "") or "")
        price = fetch_latest_price(sym)

        print(f"\n--- {sym} Decision ---")
        print(f"Action: BUY | Price: {price}")
        print(f"Reason: {explain}")

        if price is None or price <= 0:
            print(f"[WARN] No valid price for {sym}, skipping.")
            continue

        qty = int(decision.get("qty", 0))

        # Flag-based recompute (preferred, no string matching)
        recalc_all_in = bool(decision.get("recalc_all_in", False))
        recalc_after_sells = bool(decision.get("recalc_after_sells", False))

        # If strategy expected SPY to be sold first but it didn't fill, skip flagged buys.
        if (recalc_all_in or recalc_after_sells) and spy_sell_failed:
            print("[INFO] BUY skipped because required SPY SELL did not fill (cannot fund priority buy).")
            continue

        # Recompute qty from remaining cash if flagged
        if recalc_all_in:
            # refresh live cash in case SELL filled changed it (and to avoid stale state)
            remaining_cash = _get_live_account_cash(proxy_symbol=sym)
            qty = int(remaining_cash // price)
            print(f"[INFO] {sym} recalc_all_in: cash=${remaining_cash:.2f} price=${price:.2f} -> qty={qty}")

        elif recalc_after_sells:
            # use the cash remaining after earlier buys in this same cycle
            qty = int(remaining_cash // price)
            print(f"[INFO] {sym} recalc_after_sells: remaining_cash=${remaining_cash:.2f} price=${price:.2f} -> qty={qty}")

        if qty <= 0:
            print("[INFO] BUY skipped — insufficient cash.")
            continue

        pm = PortfolioManager(sym)
        filled_qty, filled_price = execute_trade("buy", qty, sym)

        if not filled_qty:
            print(f"[WARN] {sym} BUY not filled.")
            continue

        pm.refresh_live()
        pm._apply("buy", filled_price, filled_qty)
        print(f"Updated Portfolio Value for {sym}: ${pm.value():.2f}")

        # update remaining cash after successful buy
        remaining_cash = _get_live_account_cash(proxy_symbol=sym)

    # -------------------------
    # PASS 3: HOLD prints
    # -------------------------
    for sym in hold_syms:
        decision = decisions[sym]
        price = fetch_latest_price(sym)
        print(f"\n--- {sym} Decision ---")
        print(f"Action: HOLD | Qty: 0 | Price: {price}")
        print(f"Reason: {decision.get('explain','')}")
        print("[INFO] HOLD — No trade executed.")

# ===============================================================
# Orchestrator
# ===============================================================
def process_all_symbols(symbols):
    symbols = symbols if isinstance(symbols, list) else [symbols]
    symbols = [s.upper() for s in symbols]

    spy_sym = SPY_SYMBOL.upper()

    # Core universe = configured symbols WITHOUT SPY
    core_symbols = [s for s in symbols if s != spy_sym]

    # ----------------------------
    # Step 1: Predictions for core stocks
    # ----------------------------
    predictions = get_predictions(core_symbols, debug=True)

    if not predictions:
        print("[INFO] No valid predictions available. Stopping.")
        return

    # ----------------------------
    # Step 1b: ALWAYS fetch SPY too
    # (so strategy can SELL/EXIT SPY when NVDA/AAPL opportunity appears)
    # ----------------------------
    spy_preds = get_predictions([spy_sym], debug=True)
    spy_prob = spy_preds.get(spy_sym)

    if spy_prob is not None:
        predictions[spy_sym] = spy_prob
    else:
        print(f"[WARN] Could not get valid prediction for {spy_sym}. SPY will be ignored this cycle.")

    # ----------------------------
    # Step 2: Strategy logic
    # ----------------------------
    symbols_for_strategy = list(predictions.keys())
    decisions = compute_strategy_decisions(predictions, symbols=symbols_for_strategy)

    print("\n================== DECISIONS ==================")
    for sym, d in decisions.items():
        print(f"{sym}: {d['action'].upper()} {d['qty']} — {d['explain']}")
    print("================================================\n")

    # ----------------------------
    # Step 3: Execute trades
    # ----------------------------
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