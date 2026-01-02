from typing import Dict, List
from config import (
    BUY_THRESHOLD, SELL_THRESHOLD, STOP_LOSS, TAKE_PROFIT, RISK_FRACTION,
    SPY_SYMBOL, WEAK_PROB_THRESHOLD, WEAK_RATIO_THRESHOLD,
    SPY_ENTRY_THRESHOLD, SPY_EXIT_THRESHOLD, SPY_MUTUAL_EXCLUSIVE, SPY_RISK_FRACTION
)
from portfolio import PortfolioManager
from data_loader import fetch_latest_price


# ---------------------------------------------------------
# Helper to build clean decision dicts
# ---------------------------------------------------------
def make_decision(action: str, qty: int, explain: str, **meta):
    d = {"action": action, "qty": int(qty), "explain": explain.strip()}
    d.update(meta)  # allow flags like recalc_all_in=True
    return d

def _core_buy_intent(preds: Dict[str, float], core_symbols: List[str]) -> List[str]:
    """Symbols that WANT to buy based on prob only (ignores cash/qty)."""
    return [s for s in core_symbols if preds.get(s, 0.0) >= BUY_THRESHOLD]

def _any_core_buy(decisions: Dict[str, dict], core_symbols: List[str]) -> bool:
    """True if any core symbol has a BUY decision."""
    return any((decisions.get(s) or {}).get("action") == "buy" for s in core_symbols)

def _force_spy_exit_if_core_buy(decisions: Dict[str, dict], spy_sym: str, explain_suffix: str = ""):
    """If core has a BUY, sell SPY if held; otherwise hold SPY."""
    spy_pm = PortfolioManager(spy_sym)
    try:
        spy_pm.refresh_live()
    except:
        pass
    spy_shares = float(spy_pm.data.get("shares", 0.0))

    if spy_shares > 0:
        decisions[spy_sym] = make_decision(
            "sell",
            int(spy_shares),
            f"{spy_sym}: SELL (rotate into core) — core BUY opportunity detected. {explain_suffix}".strip()
        )
    else:
        decisions[spy_sym] = make_decision(
            "hold",
            0,
            f"{spy_sym}: HOLD — no position, core BUY opportunity detected. {explain_suffix}".strip()
        )

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
    core_symbols = [s for s in symbols if s != spy_sym]

    # Default decisions
    decisions = {s: make_decision("hold", 0, f"{s}: default HOLD") for s in symbols}

    # ---------------------------------------------------------
    # Fetch live state + prices (for stop/tp + funding calcs)
    # ---------------------------------------------------------
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

    # Ensure SPY state exists even if not in symbols (safe)
    if spy_sym not in pms:
        pm = PortfolioManager(spy_sym)
        try:
            pm.refresh_live()
        except:
            pass
        pms[spy_sym] = pm
        prices[spy_sym] = fetch_latest_price(spy_sym) or 0.0

    def spy_shares() -> float:
        return float(pms[spy_sym].data.get("shares", 0.0))

    def spy_price() -> float:
        return float(prices.get(spy_sym, 0.0) or 0.0)

    # ---------------------------------------------------------
    # 1) STOP-LOSS / TAKE-PROFIT always first (including SPY)
    # ---------------------------------------------------------
    sl_tp_decisions = {}
    for sym in symbols:
        price = prices.get(sym, 0.0) or 0.0
        if price > 0:
            d = check_stop_tp(sym, price, pms[sym])
            if d:
                sl_tp_decisions[sym] = d

    if sl_tp_decisions:
        for sym in symbols:
            if sym in sl_tp_decisions:
                decisions[sym] = sl_tp_decisions[sym]
            else:
                decisions[sym] = make_decision("hold", 0, f"{sym}: HOLD during SL/TP event.")
        return decisions

    # ---------------------------------------------------------
    # 2) SPY candidate (ONLY for entering SPY on weak market)
    # ---------------------------------------------------------
    spy_candidate = None
    spy_prob = preds.get(spy_sym)

    if spy_prob is not None:
        market_is_weak = _weak_market(symbols, preds)  # your helper excludes SPY
        if market_is_weak:
            sh = spy_shares()
            cash = float(pms[spy_sym].data.get("cash", 0.0))
            px = spy_price()

            if px > 0:
                if spy_prob >= SPY_ENTRY_THRESHOLD:
                    qty = int((cash * SPY_RISK_FRACTION) // px)
                    spy_candidate = make_decision(
                        "buy",
                        max(qty, 0),
                        f"{spy_sym}: SPY fallback BUY — market weak and spy_prob={spy_prob:.3f} ≥ {SPY_ENTRY_THRESHOLD}"
                    )
                elif spy_prob <= SPY_EXIT_THRESHOLD and sh > 0:
                    spy_candidate = make_decision(
                        "sell",
                        int(sh),
                        f"{spy_sym}: SPY fallback SELL — spy_prob={spy_prob:.3f} ≤ {SPY_EXIT_THRESHOLD}"
                    )
                else:
                    spy_candidate = make_decision(
                        "hold", 0,
                        f"{spy_sym}: SPY fallback HOLD — spy_prob={spy_prob:.3f}"
                    )

    # ---------------------------------------------------------
    # 3) GLOBAL SELL (core only)
    # ---------------------------------------------------------
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
    # 4) Core strategy with NVDA priority (core only)
    # ---------------------------------------------------------
    if "NVDA" not in core_symbols:
        concurrent_buys = sum(1 for s in core_symbols if preds.get(s, 0.0) >= BUY_THRESHOLD)
        for sym in core_symbols:
            decisions[sym] = should_trade(
                sym, preds.get(sym, 0.0),
                total_symbols=len(core_symbols),
                concurrent_buys=concurrent_buys
            )
    else:
        def live_shares(sym): return float(pms[sym].data.get("shares", 0.0))

        concurrent_buys = sum(1 for s in core_symbols if preds.get(s, 0.0) >= BUY_THRESHOLD)

        nvda_prob = preds.get("NVDA", 0.0)
        nvda_base = should_trade("NVDA", nvda_prob, len(core_symbols), concurrent_buys)
        nvda_action = nvda_base["action"]

        # ---- NVDA BUY priority: sell other core positions (funding) + plan big buy
        if nvda_action == "buy":
            sim_cash = float(pms[core_symbols[0]].data.get("cash", 0.0))
            nvda_px = float(prices.get("NVDA", 0.0) or 0.0)
            if nvda_px <= 0:
                return decisions

            # fund with other core positions (excluding NVDA)
            for sym in core_symbols:
                if sym == "NVDA":
                    continue
                sh = live_shares(sym)
                px = float(prices.get(sym, 0.0) or 0.0)
                if sh > 0 and px > 0:
                    sim_cash += sh * px
                    decisions[sym] = make_decision("sell", int(sh), f"{sym}: Sold to fund NVDA priority buy.")

            max_shares = int(sim_cash // nvda_px)
            if max_shares >= 1:
                decisions["NVDA"] = make_decision(
                    "buy",
                    max_shares,
                    f"NVDA BUY priority — capital ${sim_cash:.2f}, qty={max_shares}"
                )
            else:
                decisions["NVDA"] = make_decision("hold", 0, "NVDA BUY priority but insufficient capital.")

            # NOTE: do NOT apply SPY here; final enforcement below will handle it.
        elif nvda_action == "sell":
            sh = live_shares("NVDA")
            if sh > 0:
                decisions["NVDA"] = make_decision("sell", int(sh), "NVDA SELL signal.")
            else:
                decisions["NVDA"] = make_decision("hold", 0, "NVDA SELL but no position.")

            # rotate into strongest other BUY
            buyers = [s for s in core_symbols if s != "NVDA" and preds.get(s, 0.0) >= BUY_THRESHOLD]
            if buyers:
                strongest = max(buyers, key=lambda x: preds.get(x, 0.0))
                cash_now = float(pms[core_symbols[0]].data.get("cash", 0.0)) + sh * float(prices.get("NVDA", 0.0) or 0.0)
                tgt_px = float(prices.get(strongest, 0.0) or 0.0)
                qty = int(cash_now // tgt_px) if tgt_px > 0 else 0
                if qty >= 1:
                    decisions[strongest] = make_decision("buy", qty, f"{strongest}: Rotated from NVDA sell, qty={qty}")
        else:
            # NVDA HOLD -> normal core trading
            for sym in core_symbols:
                decisions[sym] = should_trade(
                    sym, preds.get(sym, 0.0),
                    total_symbols=len(core_symbols),
                    concurrent_buys=concurrent_buys
                )

    # ---------------------------------------------------------
    # 5) FINAL ENFORCEMENT: if NVDA/AAPL has BUY INTENT => SELL SPY + recalc flags
    # (prob-based intent, not cash-based)
    # ---------------------------------------------------------
    wanted = []
    if "NVDA" in core_symbols and preds.get("NVDA", 0.0) >= BUY_THRESHOLD:
        wanted.append("NVDA")
    if "AAPL" in core_symbols and preds.get("AAPL", 0.0) >= BUY_THRESHOLD:
        wanted.append("AAPL")

    if wanted:
        sh = spy_shares()
        if sh > 0:
            decisions[spy_sym] = make_decision(
                "sell",
                int(sh),
                f"{spy_sym}: SELL (rotate into core) — core BUY intent: {', '.join(wanted)}."
            )

        # mark priority buys for main.py to recalc AFTER sells
        if "NVDA" in wanted:
            decisions["NVDA"] = make_decision(
                "buy", 1,
                "NVDA PRIORITY BUY — recalc all-in after sells (SPY liquidation).",
                recalc_all_in=True,
                priority_rank=1,
            )
            if "AAPL" in wanted:
                decisions["AAPL"] = make_decision(
                    "buy", 1,
                    "AAPL BUY intent — secondary to NVDA (after sells).",
                    recalc_after_sells=True,
                    priority_rank=2,
                )
        else:
            decisions["AAPL"] = make_decision(
                "buy", 1,
                "AAPL PRIORITY BUY — recalc all-in after sells (SPY liquidation).",
                recalc_all_in=True,
                priority_rank=1,
            )

        # When core wants to buy, we do NOT allow SPY buy this cycle
        return decisions

    # ---------------------------------------------------------
    # 6) If no core buy intent, allow SPY candidate (entry/exit) with mutual exclusive rules
    # ---------------------------------------------------------
    if spy_candidate is not None:
        if (not SPY_MUTUAL_EXCLUSIVE) or (not _any_stock_trade(decisions, core_symbols)):
            decisions[spy_sym] = spy_candidate
        else:
            decisions[spy_sym] = make_decision("hold", 0, f"{spy_sym}: Mutual-exclusive → skipping SPY this cycle.")

    return decisions