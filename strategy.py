from typing import Dict, List
from config import BUY_THRESHOLD, SELL_THRESHOLD, STOP_LOSS, TAKE_PROFIT
from portfolio import PortfolioManager
from data_loader import fetch_latest_price


def make_decision(action: str, qty: int, explain: str):
    return {"action": action, "qty": int(qty), "explain": explain.strip()}


def should_trade(symbol: str, prob_up: float, available_cash: float = None):
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
        cash = available_cash

    explain = f"{symbol}: prob_up={prob_up:.3f}. "

    # STOP LOSS / TAKE PROFIT
    if shares > 0 and last_price > 0:
        if price <= last_price * STOP_LOSS:
            return make_decision("sell", int(shares),
                                 explain + f"STOP-LOSS hit ({price:.2f} <= {STOP_LOSS*100:.1f}% of {last_price:.2f}).")
        if price >= last_price * TAKE_PROFIT:
            return make_decision("sell", int(shares),
                                 explain + f"TAKE-PROFIT hit ({price:.2f} >= {TAKE_PROFIT*100:.1f}% of {last_price:.2f}).")

    # BUY
    if prob_up >= BUY_THRESHOLD:
        qty = int(cash // price)
        if qty > 0:
            return make_decision("buy", qty,
                                 explain + f"BUY signal (≥{BUY_THRESHOLD}). cash={cash:.2f}, qty={qty}.")
        return make_decision("hold", 0, explain + "BUY signal detected but insufficient cash.")

    # SELL
    if prob_up <= SELL_THRESHOLD and shares > 0:
        return make_decision("sell", int(shares),
                             explain + f"SELL signal (≤{SELL_THRESHOLD}). Selling all shares.")

    return make_decision("hold", 0, explain + "Within thresholds → HOLD.")


def compute_strategy_decisions(predictions: Dict[str, float], symbols: List[str] = None) -> Dict[str, Dict]:
    if symbols is None:
        symbols = list(predictions.keys())
    symbols = [s.upper() for s in symbols]
    preds = {k.upper(): v for k, v in predictions.items()}

    decisions = {s: make_decision("hold", 0, f"{s}: default HOLD") for s in symbols}

    # Load portfolio
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
    def last_price(sym): return float(pms[sym].data.get("last_price", 0.0))
    prices = {s: fetch_latest_price(s) or 0.0 for s in symbols}

    nvda_prob = preds.get("NVDA", 0.0)
    nvda_price = prices.get("NVDA", 0.0)
    nvda_shares = live_shares("NVDA")

    other_symbols = [s for s in symbols if s != "NVDA"]

    # -----------------------------
    # Step 1: NVDA BUY priority
    # -----------------------------
    if nvda_prob >= BUY_THRESHOLD and nvda_price > 0:
        sim_cash = live_cash("NVDA")

        # 1a: Apply SL/TP sells for other symbols
        for s in other_symbols:
            sh = live_shares(s)
            p = prices[s]
            lp = last_price(s)
            if sh > 0 and lp > 0:
                if p <= lp * STOP_LOSS or p >= lp * TAKE_PROFIT:
                    sim_cash += sh * p
                    decisions[s] = make_decision("sell", int(sh), f"{s}: SL/TP sold to fund NVDA buy.")

        # 1b: Sell other symbols in ascending probability order until all possible NVDA shares can be bought
        sellables = []
        for s in other_symbols:
            sh = live_shares(s)
            p = prices[s]
            if sh > 0 and p > 0:
                sellables.append({"symbol": s, "shares": sh, "price": p, "prob": preds.get(s, 0.0)})

        sellables.sort(key=lambda x: (x["prob"], -x["shares"]*x["price"]))

        for item in sellables:
            if sim_cash // nvda_price >= 1:
                # already enough to buy at least one NVDA
                pass
            sh, p, s = item["shares"], item["price"], item["symbol"]
            proceeds = sh * p
            sim_cash += proceeds
            decisions[s] = make_decision(int(sh), "sell",
                                         f"{s}: sold to fund NVDA buy (${proceeds:.2f}).")

        # Buy as many NVDA shares as possible with sim_cash
        qty = int(sim_cash // nvda_price)
        if qty > 0:
            decisions["NVDA"] = make_decision("buy", qty,
                                              f"NVDA BUY priority — capital=${sim_cash:.2f}, qty={qty}.")
        else:
            decisions["NVDA"] = make_decision("hold", 0,
                                              f"NVDA BUY priority — insufficient capital (${sim_cash:.2f}).")

        # Step 2: Normal decisions for other symbols not sold
        for s in other_symbols:
            if decisions[s]["action"] == "hold":
                decisions[s] = should_trade(s, preds.get(s, 0), available_cash=live_cash(s))
        return decisions

    # -----------------------------
    # Step 3: NVDA HOLD → normal logic for all symbols
    # -----------------------------
    for s in symbols:
        if decisions[s]["action"] == "hold":
            decisions[s] = should_trade(s, preds.get(s, 0), available_cash=live_cash(s))

    # -----------------------------
    # Step 4: NVDA SELL
    # -----------------------------
    if nvda_prob <= SELL_THRESHOLD and nvda_shares > 0:
        decisions["NVDA"] = make_decision("sell", int(nvda_shares), "NVDA SELL signal — liquidating shares.")

    # Step 5 is inherently handled by HOLD in should_trade()

    return decisions