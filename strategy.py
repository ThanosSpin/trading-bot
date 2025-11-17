# strategy.py
from typing import Dict, List
from config import BUY_THRESHOLD, SELL_THRESHOLD, STOP_LOSS, TAKE_PROFIT, RISK_FRACTION
from portfolio import PortfolioManager
from data_loader import fetch_latest_price

# --------------------------
# small helper to build decision dicts
# --------------------------
def make_decision(action: str, qty: int, explain: str):
    return {"action": action, "qty": int(qty), "explain": explain.strip()}

# --------------------------
# per-symbol logic (returns dict)
# --------------------------
def should_trade(symbol: str, prob_up: float, total_symbols: int = 1,
                 concurrent_buys: int = 1, available_cash: float = None):
    """
    Per-symbol decision using your original logic plus STOP/TP.
    Returns: {"action","qty","explain"}
    """
    pm = PortfolioManager(symbol)
    try:
        pm.refresh_live()
    except Exception:
        pass

    shares = float(pm.data.get("shares", 0.0))
    cash = float(pm.data.get("cash", 0.0))
    last_price = float(pm.data.get("last_price", 0.0))
    price = fetch_latest_price(symbol)

    if price is None or price <= 0:
        return make_decision("hold", 0, f"{symbol}: no valid market price.")

    if available_cash is not None:
        cash = float(available_cash)

    # allocation logic
    if total_symbols == 1:
        max_invest = cash
    elif total_symbols == 2:
        max_invest = cash * RISK_FRACTION if concurrent_buys == 2 else cash
    else:
        max_invest = cash * RISK_FRACTION

    explain = f"{symbol}: prob_up={prob_up:.3f}. "

    # Stop-loss / Take-profit (based on last recorded buy price)
    if shares > 0 and last_price > 0:
        if price <= last_price * STOP_LOSS:
            return make_decision("sell", int(shares),
                                 explain + f"STOP-LOSS hit ({price:.2f} <= {STOP_LOSS*100:.1f}% of {last_price:.2f}).")
        if price >= last_price * TAKE_PROFIT:
            return make_decision("sell", int(shares),
                                 explain + f"TAKE-PROFIT hit ({price:.2f} >= {TAKE_PROFIT*100:.1f}% of {last_price:.2f}).")

    # BUY logic
    if prob_up >= BUY_THRESHOLD:
        affordable = int(cash // price)
        target = int(max_invest // price)
        qty = min(affordable, target)
        if qty > 0:
            return make_decision("buy", qty,
                                 explain + f"BUY signal (≥{BUY_THRESHOLD}). cash={cash:.2f}, price={price:.2f}, qty={qty}.")
        return make_decision("hold", 0, explain + "BUY signal detected but insufficient cash.")

    # SELL logic
    if prob_up <= SELL_THRESHOLD and shares > 0:
        return make_decision("sell", int(shares),
                             explain + f"SELL signal (≤{SELL_THRESHOLD}). Selling all {int(shares)} shares.")

    return make_decision("hold", 0, explain + "Within thresholds → HOLD.")


# --------------------------
# NVDA-priority coordinator implementing Option2+soft fallback (A)
# --------------------------
def compute_strategy_decisions(predictions: Dict[str, float], symbols: List[str] = None) -> Dict[str, Dict]:
    """
    Returns per-symbol decisions as dict:
    { "AAPL": {"action":"buy","qty":3,"explain":"..."} , ... }
    NVDA-priority (soft): try NVDA cash first, apply SL/TP sells on others,
    then only sell other positions (least promising first) if still needed to buy at least 1 NVDA share.
    """
    if symbols is None:
        symbols = list(predictions.keys())

    symbols = [s.upper() for s in symbols]
    preds = {k.upper(): v for k, v in predictions.items()}

    # default hold decisions
    decisions: Dict[str, Dict] = {s: make_decision("hold", 0, f"{s}: default HOLD") for s in symbols}

    # if NVDA not present, do normal per-symbol decisions
    if "NVDA" not in symbols:
        concurrent_buys = sum(1 for s in symbols if preds.get(s, 0) >= BUY_THRESHOLD)
        for s in symbols:
            decisions[s] = should_trade(s, preds.get(s, 0),
                                        total_symbols=len(symbols),
                                        concurrent_buys=concurrent_buys)
        return decisions

    # instantiate PortfolioManagers and refresh live state
    pms = {}
    for s in symbols:
        pm = PortfolioManager(s)
        try:
            pm.refresh_live()
        except Exception:
            pass
        pms[s] = pm

    def live_shares(sym): return float(pms[sym].data.get("shares", 0.0))
    def live_cash(sym): return float(pms[sym].data.get("cash", 0.0))
    def last_buy_price(sym): return float(pms[sym].data.get("last_price", 0.0))

    # fetch current market prices
    prices = {}
    for s in symbols:
        prices[s] = fetch_latest_price(s) or 0.0

    # base NVDA decision using should_trade (gives us stop/take logic for NVDA as well)
    nvda_prob = preds.get("NVDA", 0.0)
    nvda_base = should_trade("NVDA", nvda_prob, total_symbols=len(symbols),
                             concurrent_buys=sum(1 for s in symbols if preds.get(s, 0) >= BUY_THRESHOLD))
    nvda_action = nvda_base["action"]

    # -------------------------
    # NVDA BUY priority (soft flow A)
    # -------------------------
    if nvda_action == "buy":
        nvda_price = prices.get("NVDA", 0.0)
        if nvda_price <= 0:
            # can't trade NVDA without price; fallback to normal decisions
            concurrent_buys = sum(1 for s in symbols if preds.get(s, 0) >= BUY_THRESHOLD)
            for s in symbols:
                decisions[s] = should_trade(s, preds.get(s, 0),
                                            total_symbols=len(symbols),
                                            concurrent_buys=concurrent_buys)
            return decisions

        # 1) start with NVDA's live cash
        sim_cash = live_cash("NVDA")

        # 2) apply SL/TP sells on other symbols first (preserve risk rules)
        for s in symbols:
            if s == "NVDA":
                continue
            s_shares = live_shares(s)
            s_price = prices.get(s, 0.0)
            s_last = last_buy_price(s)

            if s_shares > 0 and s_price > 0 and s_last > 0:
                # SL
                if s_price <= s_last * STOP_LOSS:
                    sim_cash += s_shares * s_price
                    decisions[s] = make_decision("sell", int(s_shares),
                                                 f"{s}: STOP-LOSS sell executed under NVDA priority (price {s_price:.2f}).")
                    continue
                # TP
                if s_price >= s_last * TAKE_PROFIT:
                    sim_cash += s_shares * s_price
                    decisions[s] = make_decision("sell", int(s_shares),
                                                 f"{s}: TAKE-PROFIT sell executed under NVDA priority (price {s_price:.2f}).")
                    continue
            # default for now: hold until we determine if more cash is needed
            decisions[s] = make_decision("hold", 0, f"{s}: held for now under NVDA priority.")

        # 3) check if we already can buy at least 1 NVDA share
        if int(sim_cash // nvda_price) >= 1:
            qty = int(sim_cash // nvda_price)
            decisions["NVDA"] = make_decision("buy", qty,
                                              f"NVDA BUY priority — used NVDA cash + SL/TP proceeds. Allocated ${sim_cash:.2f}. qty={qty}.")
            return decisions

        # 4) Need more cash: sell other positions (least promising first) until >=1 NVDA share
        # Build list of sellable positions with shares > 0 (exclude NVDA)
        sellables = []
        for s in symbols:
            if s == "NVDA":
                continue
            s_shares = live_shares(s)
            s_price = prices.get(s, 0.0)
            if s_shares > 0 and s_price > 0:
                sellables.append({
                    "symbol": s,
                    "shares": s_shares,
                    "price": s_price,
                    "value": s_shares * s_price,
                    "prob": preds.get(s, 0.0),
                })
        # Sort by ascending prob (least promising first), tie-breaker: largest value first
        sellables.sort(key=lambda x: (x["prob"], -x["value"]))

        # Sell iteratively until enough cash or exhausted
        for item in sellables:
            if int(sim_cash // nvda_price) >= 1:
                break
            s = item["symbol"]
            proceeds = item["value"]
            sim_cash += proceeds
            decisions[s] = make_decision("sell", int(item["shares"]),
                                         f"{s}: sold to fund NVDA priority buy (proceeds ${proceeds:.2f}).")

        # final check
        if int(sim_cash // nvda_price) >= 1:
            qty = int(sim_cash // nvda_price)
            decisions["NVDA"] = make_decision("buy", qty,
                                              f"NVDA BUY priority — liquidated lower-priority positions to raise ${sim_cash:.2f}; qty={qty}.")
        else:
            # still cannot afford 1 share, so do NOT force sell everything; fallback: hold NVDA and keep prior sells (SL/TP) only
            decisions["NVDA"] = make_decision("hold", 0,
                                              f"NVDA BUY priority but insufficient capital after SL/TP and selective sells (available ${sim_cash:.2f}).")
        return decisions

    # -------------------------
    # NVDA SELL: rotate into other buys if any, otherwise normal logic
    # -------------------------
    if nvda_action == "sell":
        # find other buy candidates
        other_buyers = [s for s in symbols if s != "NVDA" and preds.get(s, 0) >= BUY_THRESHOLD]
        nvda_sh = live_shares("NVDA")
        if nvda_sh > 0:
            decisions["NVDA"] = make_decision("sell", int(nvda_sh), "NVDA SELL signal — selling NVDA first.")
        else:
            decisions["NVDA"] = make_decision("hold", 0, "NVDA SELL signal but no holdings.")

        if other_buyers:
            # pick strongest buy candidate
            strongest = max(other_buyers, key=lambda s: preds.get(s, 0))
            nvda_price = prices.get("NVDA", 0.0)
            sim_cash = live_cash("NVDA") + (nvda_sh * nvda_price if nvda_price > 0 else 0.0)
            tgt_price = prices.get(strongest, 0.0)
            qty = int(sim_cash // tgt_price) if tgt_price > 0 else 0
            if qty > 0:
                decisions[strongest] = make_decision("buy", qty,
                                                    f"{strongest}: BUY after NVDA sell. Allocated ${sim_cash:.2f}. qty={qty}.")
            else:
                decisions[strongest] = make_decision("hold", 0,
                                                    f"{strongest}: BUY candidate but insufficient funds after NVDA sell (sim_cash ${sim_cash:.2f}).")
        # other symbols normal logic
        for s in symbols:
            if s not in decisions:
                decisions[s] = should_trade(s, preds.get(s, 0),
                                            total_symbols=len(symbols),
                                            concurrent_buys=sum(1 for x in symbols if preds.get(x, 0) >= BUY_THRESHOLD))
        return decisions

    # -------------------------
    # ALL SELL scenario
    # -------------------------
    if all(preds.get(s, 1) <= SELL_THRESHOLD for s in symbols):
        for s in symbols:
            sh = live_shares(s)
            if sh > 0:
                decisions[s] = make_decision("sell", int(sh), f"{s}: Global SELL condition — liquidating.")
            else:
                decisions[s] = make_decision("hold", 0, f"{s}: Global SELL condition — no shares.")
        return decisions

    # -------------------------
    # NVDA HOLD → normal per-symbol decisions
    # -------------------------
    concurrent_buys = sum(1 for s in symbols if preds.get(s, 0) >= BUY_THRESHOLD)
    for s in symbols:
        decisions[s] = should_trade(s, preds.get(s, 0),
                                    total_symbols=len(symbols),
                                    concurrent_buys=concurrent_buys)

    return decisions