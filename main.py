# main.py
import time
from data_loader import fetch_historical_data, fetch_latest_price
from market import is_market_open, debug_market
from model_xgb import compute_signals
from strategy import compute_strategy_decisions
from portfolio import PortfolioManager
from trader import execute_trade, get_pdt_status
from strategy import apply_position_limits
from account_cache import account_cache
from config import (
    SYMBOL, BUY_THRESHOLD, SELL_THRESHOLD, INTRADAY_WEIGHT,
    SPY_SYMBOL, WEAK_PROB_THRESHOLD, WEAK_RATIO_THRESHOLD,
    SPY_ENTRY_THRESHOLD, SPY_EXIT_THRESHOLD, SPY_MUTUAL_EXCLUSIVE
)
from market import is_market_open, is_trading_day
import os, csv
import pandas as pd
from datetime import datetime, timezone


os.environ["TZ"] = "America/New_York"
time.tzset()


BOT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BOT_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)


# ===============================================================
# ✅ FIXED: Signal history logger with new schema
# ===============================================================
def log_signal_snapshot(sym: str, sig: dict) -> None:
    """
    Log signal snapshot with flexible schema handling.
    Auto-creates headers based on signal dict keys.
    """
    try:
        os.makedirs(LOGS_DIR, exist_ok=True)
        symU = str(sym).upper().strip()
        path = os.path.join(LOGS_DIR, f"signals_{symU}.csv")
        
        # Define expected columns in order
        columns = [
            "timestamp",
            "symbol", 
            "price",
            "dailyprob",
            "intradayprob",
            "finalprob",
            "intradayweight",
            "intradaymodelused",
            "intradayqualityscore",
            "intradayvol",
            "intradaymom",
            "intradayregime",  # ✅ NEW field
            "allowintraday",   # ✅ NEW field
        ]
        
        # Check if file exists
        file_exists = os.path.exists(path)
        
        # Prepare row data
        row_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symU,
            "price": sig.get("price"),
            "dailyprob": sig.get("daily_prob"),
            "intradayprob": sig.get("intraday_prob"),
            "finalprob": sig.get("final_prob"),
            "intradayweight": sig.get("intraday_weight"),
            "intradaymodelused": sig.get("intraday_model_used"),
            "intradayqualityscore": sig.get("intraday_quality_score"),
            "intradayvol": sig.get("intraday_vol"),
            "intradaymom": sig.get("intraday_mom"),
            "intradayregime": sig.get("intraday_regime"),  # ✅ NEW
            "allowintraday": sig.get("allow_intraday"),    # ✅ NEW
        }
        
        # Write to CSV
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            
            # Write header if new file
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(row_data)
    except Exception as e:
        print(f"[WARN] log_signal_snapshot failed for {sym}: {e}")


# ===============================================================
# Fetch Predictions for All Symbols (USING compute_signals)
# ===============================================================
def get_predictions(symbols, debug=True):
    predictions = {}
    diagnostics = {}


    for sym in symbols:
        print(f"\n🔍 Fetching signals for {sym}...")


        sig = compute_signals(
            sym,
            lookback_minutes=2400,
            intraday_weight=INTRADAY_WEIGHT,
            resample_to="15min",
        )
        # ✅ DEBUG
        print(f"[MAIN] {sym} regime={sig.get('intraday_regime')} model={sig.get('intraday_model_used')} ip={sig.get('intraday_prob')}")
        # print(f"[DIAG] NVDA debug ")
        # print(f"  intraday_rows: {sig.get('intraday_rows')}")
        # print(f"  intraday_quality_score: {sig.get('intraday_quality_score')}")
        # print(f"  allow_intraday: {sig.get('allow_intraday')}")
        # print(f"  intraday_vol: {sig.get('intraday_vol')}")
        # print("  ---")

        # ✅ log signal history for dashboard
        log_signal_snapshot(sym, sig)


        if debug:
            print(f"\n[DEBUG] {sym} Signals Summary")
            print("------------------------------------")
            print(f"Daily rows:         {sig.get('daily_rows')}")
            print(f"Intraday rows:      {sig.get('intraday_rows')}  "
                  f"(before trim: {sig.get('intraday_before')})")
            print(f"Daily prob:         {sig.get('daily_prob')}")
            print(f"Intraday prob:      {sig.get('intraday_prob')}")
            print(f"Final combined prob:{sig.get('final_prob')}")
            print(f"Intraday regime:    {sig.get('intraday_regime')}")  # ✅ NEW
            print(f"Allow intraday:     {sig.get('allow_intraday')}")   # ✅ NEW
            print("------------------------------------\n")


        final_prob = sig.get("final_prob")
        if final_prob is None:
            print(f"[WARN] Invalid prediction for {sym}, skipping.")
            continue


        predictions[sym] = final_prob


        diagnostics[sym] = {
            "daily_prob": sig.get("daily_prob"),
            "intraday_prob": sig.get("intraday_prob"),
            "final_prob": sig.get("final_prob"),
            "intraday_weight": sig.get("intraday_weight"),
            "intraday_model_used": sig.get("intraday_model_used"),
            "intraday_quality_score": sig.get("intraday_quality_score"),
            "intraday_vol": sig.get("intraday_vol"),
            "intraday_mom": sig.get("intraday_mom"),
            "intraday_regime": sig.get("intraday_regime"),      # ✅ NEW
            "allow_intraday": sig.get("allow_intraday"),        # ✅ NEW
            "intraday_volume": sig.get("intraday_volume"),              # ✅ NEW
            "intraday_volume_ratio": sig.get("intraday_volume_ratio"),  # ✅ NEW
            "price": sig.get("price"),
        }


    return predictions, diagnostics


# ===============================================================
# Cycle Summary
# ===============================================================
def print_cycle_summary(decisions):
    buys = [s for s, d in decisions.items() if d.get("action") == "buy"]
    sells = [s for s, d in decisions.items() if d.get("action") == "sell"]
    holds = [s for s, d in decisions.items() if d.get("action") == "hold"]
    
    core_buy = next((s for s in buys if s in ("NVDA", "AAPL")), None)
    
    # ✅ Use cache instead of creating new PM
    cash = account_cache.get_account().get("cash", 0.0)
    
    print(
        f"📊 Cycle Summary | "
        f"BUY: {len(buys)} | "
        f"SELL: {len(sells)} | "
        f"HOLD: {len(holds)} | "
        f"Core BUY: {core_buy or '-'} | "
        f"Cash: ${cash:,.2f}"
    )


def print_signal_diagnostics(decisions, diagnostics):
    print("🔍 SIGNAL DIAGNOSTICS")
    
    def fmt(x, n=3):
        try:
            return "NA" if x is None else f"{float(x):.{n}f}"
        except:
            return "NA"
    
    def fmt_div(x):
        try:
            return "NA" if x is None else f"{float(x):.3f}"
        except:
            return "NA"
    
    decisions = decisions or {}
    diagnostics = diagnostics or {}
    
    for sym, d in decisions.items():
        sig = diagnostics.get(sym, {})
        dp, ip, fp = sig.get("daily_prob"), sig.get("intraday_prob"), sig.get("final_prob")
        w, model_used = sig.get("intraday_weight"), sig.get("intraday_model_used") or sig.get("model")
        vol, mom, q = sig.get("intraday_vol"), sig.get("intraday_mom"), sig.get("intraday_quality_score")
        
        div = fmt_div(ip - dp) if dp is not None and ip is not None else "NA"
        action = d.get("action", "hold").upper()
        
        # 🔥 VR FIX: intraday_vol / daily_vol
        vr = "NA"
        if vol is not None:
            try:
                df_daily = fetch_historical_data(sym, period="1mo", interval="1d")
                if df_daily is not None and len(df_daily) >= 20:
                    daily_close = df_daily["Close"].iloc[:, 0] if isinstance(df_daily["Close"], pd.DataFrame) else df_daily["Close"]
                    daily_rets = daily_close.pct_change().dropna()
                    daily_vol = float(daily_rets.std()) if len(daily_rets) >= 5 else 0.0024
                    vr = f"{vol/daily_vol:.2f}" if daily_vol > 0 else "NA"
            except:
                vr = f"{vol/0.0024:.1f}"  # fallback
        
        print(f"  {sym:<5} D={fmt(dp)} I={fmt(ip)} Δ={div} W={fmt(w):>5} → F={fmt(fp)} | q={fmt(q):>4} vol={fmt(vol,5)} mom={fmt(mom,4)} vr={vr} | regime={sig.get('intraday_regime')} | model={model_used} | {action}")


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


    # ✅ Fetch account state ONCE at the beginning
    account_cache.invalidate()  # Force fresh data for this cycle
    account_state = account_cache.get_account()
    remaining_cash = account_state.get("cash", 0.0)


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


        filled_qty, filled_price = execute_trade("sell", qty, sym, decision=decision)


        if not filled_qty:
            print(f"[WARN] {sym} SELL not filled.")
            sell_failed.add(sym)
            continue


        any_sell_filled = True
        pm.refresh_live()
        account_cash = _get_live_account_cash(proxy_symbol=sym)
        pm._apply("sell", filled_price, filled_qty, account_cash=account_cash)
        print(f"Updated Snapshot (cash + {sym} position): ${pm.value():.2f}")


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


            # ✅ Apply limits
            qty = apply_position_limits(qty, price, remaining_cash, sym)


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
        account_cash = _get_live_account_cash(proxy_symbol=sym)
        pm._apply("buy", filled_price, filled_qty, account_cash=account_cash)
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
    predictions, diagnostics = get_predictions(core_symbols, debug=False)


    if not predictions:
        print("[INFO] No valid predictions available. Stopping.")
        return


    # ----------------------------
    # Step 1b: ALWAYS fetch SPY too
    # (so strategy can SELL/EXIT SPY when NVDA/AAPL opportunity appears)
    # ----------------------------
    spy_preds, spy_diag = get_predictions([spy_sym], debug=False)
    spy_prob = spy_preds.get(spy_sym)


    if spy_prob is not None:
        predictions[spy_sym] = spy_prob
        diagnostics.update(spy_diag)   # merge SPY diagnostics into main diagnostics
    else:
        print(f"[WARN] Could not get valid prediction for {spy_sym}. SPY will be ignored this cycle.")


    # ----------------------------
    # Step 2: Strategy logic
    # ----------------------------
    symbols_for_strategy = list(predictions.keys())
    decisions = compute_strategy_decisions(predictions, symbols=symbols_for_strategy, diagnostics=diagnostics)


    print("\n================== DECISIONS ==================")
    for sym, d in decisions.items():
        print(f"{sym}: {d['action'].upper()} {d['qty']} — {d['explain']}")
    print("================================================\n")
    
    # ----------------------------
    # Step 3: Execute trades
    # ----------------------------
    execute_decisions(decisions)
    
    # ----------------------------
    # Step 4: Execute Cycle Summary
    # ----------------------------
    print("\n================== CYCLE SUMMARY ==================")
    print_cycle_summary(decisions)
    print("================================================\n")
    
    # ----------------------------
    # Step 5: Execute Signal Diagnostics
    # ----------------------------
    print_signal_diagnostics(decisions, diagnostics)


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
