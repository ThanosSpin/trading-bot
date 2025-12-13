from typing import Dict, List
from config import (
    BUY_THRESHOLD, SELL_THRESHOLD, STOP_LOSS, TAKE_PROFIT, RISK_FRACTION,
    SPY_SYMBOL, WEAK_PROB_THRESHOLD, WEAK_RATIO_THRESHOLD,
    SPY_ENTRY_THRESHOLD, SPY_EXIT_THRESHOLD, SPY_MUTUAL_EXCLUSIVE
)
from portfolio import PortfolioManager
from data_loader import fetch_latest_price


# ---------------------------------------------------------
# Helper to build clean decision dicts
# ---------------------------------------------------------
def make_decision(action: str, qty: int, explain: str):
    return {"action": action, "qty": int(qty), "explain": explain.strip()}

# ---------------------------------------------------------
# Helper for weak market
# ---------------------------------------------------------
def _weak_market(symbols: List[str], preds: Dict[str, float]) -> bool:
    """Return True if enough of the non-SPY symbols are weak."""
    universe = [s for s in symbols if s != SPY_SYMBOL]
    if not universe:
        return False

    weak = [s for s in universe if preds.get(s, 1.0) <= WEAK_PROB_THRESHOLD]
    ratio = len(weak) / max(len(universe), 1)
    return ratio >= WEAK_RATIO_THRESHOLD


def _any_stock_trade(decisions: Dict[str, dict], symbols: List[str]) -> bool:
    """True if any non-SPY symbol has buy/sell decision."""
    for s in symbols:
        if s == SPY_SYMBOL:
            continue
        a = (decisions.get(s) or {}).get("action", "hold")
        if a in ("buy", "sell"):
            return True
    return False


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

    spy_sym = SPY_SYMBOL.upper()

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

    # ---------------------------------------------------------
    # First pass — ALWAYS apply STOP-LOSS / TAKE-PROFIT
    # ---------------------------------------------------------
    sl_tp_decisions = {}
    for sym in symbols:
        price = prices.get(sym, 0)
        if price > 0:
            sltp = check_stop_tp(sym, price, pms[sym])
            if sltp:
                sl_tp_decisions[sym] = sltp

    if sl_tp_decisions:
        for sym in symbols:
            if sym in sl_tp_decisions:
                decisions[sym] = sl_tp_decisions[sym]
            else:
                decisions[sym] = make_decision("hold", 0, f"{sym}: HOLD during SL/TP event.")
        return decisions

    # ---------------------------------------------------------
    # Prepare SPY candidate (apply later so mutual-excl works)
    # ---------------------------------------------------------
    spy_candidate = None
    spy_prob = preds.get(spy_sym)

    if spy_prob is not None:
        market_is_weak = _weak_market(symbols, preds)  # already excludes SPY_SYMBOL in your helper
        if market_is_weak:
            spy_pm = PortfolioManager(spy_sym)
            try:
                spy_pm.refresh_live()
            except:
                pass

            spy_shares = float(spy_pm.data.get("shares", 0.0))
            spy_cash = float(spy_pm.data.get("cash", 0.0))
            spy_price = fetch_latest_price(spy_sym) or 0.0

            if spy_price > 0:
                if spy_prob >= SPY_ENTRY_THRESHOLD:
                    qty = int((spy_cash * RISK_FRACTION) // spy_price)
                    spy_candidate = make_decision(
                        "buy",
                        max(qty, 0),
                        f"{spy_sym}: SPY fallback BUY — market weak and spy_prob={spy_prob:.3f} ≥ {SPY_ENTRY_THRESHOLD}"
                    )
                elif spy_prob <= SPY_EXIT_THRESHOLD and spy_shares > 0:
                    spy_candidate = make_decision(
                        "sell",
                        int(spy_shares),
                        f"{spy_sym}: SPY fallback SELL — spy_prob={spy_prob:.3f} ≤ {SPY_EXIT_THRESHOLD}"
                    )
                else:
                    spy_candidate = make_decision(
                        "hold", 0,
                        f"{spy_sym}: SPY fallback HOLD — spy_prob={spy_prob:.3f}"
                    )

    # ---------------------------------------------------------
    # GLOBAL SELL — exclude SPY from this check
    # ---------------------------------------------------------
    core_symbols = [s for s in symbols if s != spy_sym]
    valid_core_preds = [preds[s] for s in core_symbols if s in preds]

    if valid_core_preds and all(p <= SELL_THRESHOLD for p in valid_core_preds):
        for sym in symbols:
            sh = float(pms[sym].data.get("shares", 0.0))
            if sh > 0:
                decisions[sym] = make_decision("sell", int(sh), f"{sym}: Global SELL — liquidating.")
            else:
                decisions[sym] = make_decision("hold", 0, f"{sym}: Global SELL — no position.")
        return decisions

    # ---------------------------------------------------------
    # If NVDA not present → regular logic (exclude SPY from concurrent buys)
    # ---------------------------------------------------------
    if "NVDA" not in core_symbols:
        concurrent_buys = sum(1 for s in core_symbols if preds.get(s, 0) >= BUY_THRESHOLD)

        for sym in core_symbols:
            decisions[sym] = should_trade(
                sym, preds.get(sym, 0),
                total_symbols=len(core_symbols),
                concurrent_buys=concurrent_buys
            )

        # Apply SPY after regular decisions
        if spy_candidate is not None:
            if (not SPY_MUTUAL_EXCLUSIVE) or (not _any_stock_trade(decisions, core_symbols)):
                decisions[spy_sym] = spy_candidate
            else:
                decisions[spy_sym] = make_decision("hold", 0, f"{spy_sym}: Mutual-exclusive → skipping SPY this cycle.")

        return decisions

    # ---------------------------------------------------------
    # NVDA PRIORITY LOGIC (operate on core symbols only)
    # ---------------------------------------------------------
    def live_shares(sym): return float(pms[sym].data.get("shares", 0.0))

    nvda_prob = preds.get("NVDA", 0)
    concurrent_buys = sum(1 for s in core_symbols if preds.get(s, 0) >= BUY_THRESHOLD)
    nvda_base = should_trade("NVDA", nvda_prob, len(core_symbols), concurrent_buys)
    nvda_action = nvda_base["action"]

    # NVDA BUY PRIORITY
    if nvda_action == "buy":
        global_cash = float(pms[core_symbols[0]].data.get("cash", 0))
        nvda_price = prices.get("NVDA", 0.0)
        if nvda_price <= 0:
            return decisions

        sim_cash = global_cash

        for sym in core_symbols:
            if sym == "NVDA":
                continue
            sh = live_shares(sym)
            pr = prices.get(sym, 0.0)
            if sh > 0 and pr > 0:
                sim_cash += sh * pr
                decisions[sym] = make_decision("sell", int(sh), f"{sym}: Sold to fund NVDA priority buy.")

        max_shares = int(sim_cash // nvda_price)
        if max_shares >= 1:
            decisions["NVDA"] = make_decision("buy", max_shares, f"NVDA BUY priority — capital ${sim_cash:.2f}, qty={max_shares}")
        else:
            decisions["NVDA"] = make_decision("hold", 0, "NVDA BUY priority but insufficient capital.")

        # Apply SPY last
        if spy_candidate is not None:
            if (not SPY_MUTUAL_EXCLUSIVE) or (not _any_stock_trade(decisions, core_symbols)):
                decisions[spy_sym] = spy_candidate
            else:
                decisions[spy_sym] = make_decision("hold", 0, f"{spy_sym}: Mutual-exclusive → skipping SPY this cycle.")

        return decisions

    # NVDA SELL PRIORITY — rotate into strongest BUY
    if nvda_action == "sell":
        sh = live_shares("NVDA")
        if sh > 0:
            decisions["NVDA"] = make_decision("sell", int(sh), "NVDA SELL signal.")
        else:
            decisions["NVDA"] = make_decision("hold", 0, "NVDA SELL but no position.")

        buyers = [s for s in core_symbols if s != "NVDA" and preds.get(s, 0) >= BUY_THRESHOLD]
        if buyers:
            strongest = max(buyers, key=lambda x: preds.get(x, 0))
            sim_cash = float(pms[core_symbols[0]].data.get("cash", 0.0)) + sh * (prices.get("NVDA", 0.0))
            tgt_price = prices.get(strongest, 0.0)
            qty = int(sim_cash // tgt_price) if tgt_price > 0 else 0
            if qty >= 1:
                decisions[strongest] = make_decision("buy", qty, f"{strongest}: Rotated from NVDA sell, qty={qty}")

        # Apply SPY last
        if spy_candidate is not None:
            if (not SPY_MUTUAL_EXCLUSIVE) or (not _any_stock_trade(decisions, core_symbols)):
                decisions[spy_sym] = spy_candidate
            else:
                decisions[spy_sym] = make_decision("hold", 0, f"{spy_sym}: Mutual-exclusive → skipping SPY this cycle.")

        return decisions

    # NVDA HOLD — normal trading for core symbols
    for sym in core_symbols:
        decisions[sym] = should_trade(
            sym, preds.get(sym, 0),
            total_symbols=len(core_symbols),
            concurrent_buys=concurrent_buys
        )

    # Apply SPY last
    if spy_candidate is not None:
        if (not SPY_MUTUAL_EXCLUSIVE) or (not _any_stock_trade(decisions, core_symbols)):
            decisions[spy_sym] = spy_candidate
        else:
            decisions[spy_sym] = make_decision("hold", 0, f"{spy_sym}: Mutual-exclusive → skipping SPY this cycle.")

    return decisions