from typing import Dict, List
from config import BUY_THRESHOLD, SELL_THRESHOLD, STOP_LOSS, TAKE_PROFIT, RISK_FRACTION
from portfolio import PortfolioManager
from data_loader import fetch_latest_price


# ---------------------------------------------------------
# Helper to build clean decision dicts
# ---------------------------------------------------------
def make_decision(action: str, qty: int, explain: str):
    return {"action": action, "qty": int(qty), "explain": explain.strip()}


# ---------------------------------------------------------
# Evaluate STOP-LOSS / TAKE-PROFIT (universal)
# ---------------------------------------------------------
def check_stop_tp(symbol, price, pm):
    shares = pm.data.get("shares", 0.0)
    if shares <= 0:
        return None

    # Use weighted average price if present
    entry_price = pm.data.get("avg_price", pm.data.get("last_price", 0))

    if entry_price <= 0:
        return None

    # STOP-LOSS
    if price <= entry_price * STOP_LOSS:
        return make_decision(
            "sell",
            int(shares),
            f"{symbol}: STOP-LOSS hit {price:.2f} <= {STOP_LOSS*100:.1f}% of entry {entry_price:.2f}"
        )

    # TAKE-PROFIT
    if price >= entry_price * TAKE_PROFIT:
        return make_decision(
            "sell",
            int(shares),
            f"{symbol}: TAKE-PROFIT hit {price:.2f} >= {TAKE_PROFIT*100:.1f}% of entry {entry_price:.2f}"
        )

    return None


# ---------------------------------------------------------
# Per-symbol trading logic (baseline)
# ---------------------------------------------------------
def should_trade(symbol: str, prob_up: float, total_symbols: int = 1,
                 concurrent_buys: int = 1, available_cash: float = None):

    pm = PortfolioManager(symbol)
    try:
        pm.refresh_live()
    except:
        pass

    shares = float(pm.data.get("shares", 0.0))
    cash = float(pm.data.get("cash", 0.0))
    price = fetch_latest_price(symbol)

    if price is None or price <= 0:
        return make_decision("hold", 0, f"{symbol}: no valid price.")

    if available_cash is not None:
        cash = float(available_cash)

    # Allocation logic
    if total_symbols == 1:
        max_invest = cash
    elif total_symbols == 2:
        max_invest = cash * RISK_FRACTION if concurrent_buys == 2 else cash
    else:
        max_invest = cash * RISK_FRACTION

    explain = f"{symbol}: prob_up={prob_up:.3f}. "

    # ---------------------------
    # BUY
    # ---------------------------
    if prob_up >= BUY_THRESHOLD:
        affordable = int(cash // price)
        target = int(max_invest // price)
        qty = min(affordable, target)

        if qty > 0:
            return make_decision("buy", qty,
                                 explain + f"BUY (≥{BUY_THRESHOLD}). qty={qty}")
        return make_decision("hold", 0, explain + "BUY signal but insufficient cash.")

    # ---------------------------
    # SELL
    # ---------------------------
    if prob_up <= SELL_THRESHOLD and shares > 0:
        return make_decision("sell", int(shares),
                             explain + f"SELL (≤{SELL_THRESHOLD}). Clearing position.")

    # HOLD
    return make_decision("hold", 0, explain + "Within thresholds → HOLD.")


# ---------------------------------------------------------
# NVDA-priority coordinator
# ---------------------------------------------------------
def compute_strategy_decisions(predictions: Dict[str, float], symbols: List[str] = None):

    if symbols is None:
        symbols = list(predictions.keys())

    symbols = [s.upper() for s in symbols]
    preds = {k.upper(): v for k, v in predictions.items()}

    # Default decisions
    decisions = {s: make_decision("hold", 0, f"{s}: default HOLD") for s in symbols}

    # Fetch live state
    pms = {}
    prices = {}
    for sym in symbols:
        pm = PortfolioManager(sym)
        try:
            pm.refresh_live()
        except:
            pass
        pms[sym] = pm
        prices[sym] = fetch_latest_price(sym) or 0.0

    # First pass — ALWAYS apply STOP-LOSS / TAKE-PROFIT
    sl_tp_decisions = {}
    for sym in symbols:
        price = prices.get(sym, 0)
        if price > 0:
            sltp = check_stop_tp(sym, price, pms[sym])
            if sltp:
                sl_tp_decisions[sym] = sltp

    # If STOP/TP triggered for any symbol, apply ONLY these
    if sl_tp_decisions:
        for sym in symbols:
            if sym in sl_tp_decisions:
                decisions[sym] = sl_tp_decisions[sym]
            else:
                decisions[sym] = make_decision("hold", 0, f"{sym}: HOLD during SL/TP event.")
        return decisions

    # ---------------------------------------------------------
    # GLOBAL SELL — corrected logic
    # ---------------------------------------------------------
    valid_preds = [preds[s] for s in symbols if s in preds]

    if valid_preds and all(p <= SELL_THRESHOLD for p in valid_preds):
        for sym in symbols:
            shares = pms[sym].data.get("shares", 0)
            if shares > 0:
                decisions[sym] = make_decision("sell", int(shares),
                                               f"{sym}: Global SELL — liquidating.")
            else:
                decisions[sym] = make_decision("hold", 0, f"{sym}: Global SELL — no position.")
        return decisions

    # ---------------------------------------------------------
    # If NVDA not traded → regular logic
    # ---------------------------------------------------------
    if "NVDA" not in symbols:
        concurrent_buys = sum(1 for s in symbols if preds.get(s, 0) >= BUY_THRESHOLD)
        for sym in symbols:
            decisions[sym] = should_trade(sym, preds.get(sym, 0),
                                          total_symbols=len(symbols),
                                          concurrent_buys=concurrent_buys)
        return decisions

    # ---------------------------------------------------------
    # NVDA PRIORITY LOGIC
    # ---------------------------------------------------------
    def live_shares(sym): return float(pms[sym].data.get("shares", 0))
    def entry_price(sym): return float(pms[sym].data.get("avg_price",
                                 pms[sym].data.get("last_price", 0)))

    nvda_prob = preds.get("NVDA", 0)
    concurrent_buys = sum(1 for s in symbols if preds.get(s, 0) >= BUY_THRESHOLD)
    nvda_base = should_trade("NVDA", nvda_prob, len(symbols), concurrent_buys)
    nvda_action = nvda_base["action"]

    # ---------------------------------------------------------
    # NVDA BUY PRIORITY
    # ---------------------------------------------------------
    if nvda_action == "buy":
        # Use real global cash
        global_cash = float(pms[symbols[0]].data.get("cash", 0))
        nvda_price = prices["NVDA"]

        if nvda_price <= 0:
            return decisions  # bad price

        sim_cash = global_cash

        # NVDA BUY → liquidate other positions if needed
        for sym in symbols:
            if sym == "NVDA":
                continue

            sh = live_shares(sym)
            pr = prices[sym]

            if sh > 0 and pr > 0:
                sim_cash += sh * pr
                decisions[sym] = make_decision("sell", int(sh),
                                               f"{sym}: Sold to fund NVDA priority buy.")

        max_shares = int(sim_cash // nvda_price)

        if max_shares >= 1:
            decisions["NVDA"] = make_decision(
                "buy",
                max_shares,
                f"NVDA BUY priority — capital ${sim_cash:.2f}, qty={max_shares}"
            )
        else:
            decisions["NVDA"] = make_decision(
                "hold",
                0,
                "NVDA BUY priority but insufficient capital."
            )

        return decisions

    # ---------------------------------------------------------
    # NVDA SELL PRIORITY — rotate into strongest BUY
    # ---------------------------------------------------------
    if nvda_action == "sell":
        sh = live_shares("NVDA")
        pr = prices["NVDA"]

        if sh > 0:
            decisions["NVDA"] = make_decision("sell", int(sh), "NVDA SELL signal.")
        else:
            decisions["NVDA"] = make_decision("hold", 0, "NVDA SELL but no position.")

        # Rotate into strongest BUY alternative
        buyers = [s for s in symbols if s != "NVDA" and preds.get(s, 0) >= BUY_THRESHOLD]

        if buyers:
            strongest = max(buyers, key=lambda x: preds.get(x, 0))
            sim_cash = float(pms[symbols[0]].data.get("cash", 0)) + sh * pr
            tgt_price = prices[strongest]
            qty = int(sim_cash // tgt_price)

            if qty >= 1:
                decisions[strongest] = make_decision(
                    "buy", qty,
                    f"{strongest}: Rotated from NVDA sell, qty={qty}"
                )

        return decisions

    # ---------------------------------------------------------
    # NVDA HOLD — normal trading for all symbols
    # ---------------------------------------------------------
    for sym in symbols:
        decisions[sym] = should_trade(sym, preds.get(sym, 0),
                                      total_symbols=len(symbols),
                                      concurrent_buys=concurrent_buys)

    return decisions