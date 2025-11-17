# strategy.py
from typing import Dict, List
from config import BUY_THRESHOLD, SELL_THRESHOLD, STOP_LOSS, TAKE_PROFIT, RISK_FRACTION
from portfolio import PortfolioManager
from data_loader import fetch_latest_price


# =====================================================================
# Helper: Format decision dicts
# =====================================================================
def make_decision(action: str, qty: int, explain: str):
    return {
        "action": action,
        "qty": int(qty),
        "explain": explain.strip()
    }


# =====================================================================
# Per-symbol decision function
# =====================================================================
def should_trade(symbol: str, prob_up: float, total_symbols: int = 1,
                 concurrent_buys: int = 1, available_cash: float = None):
    """
    Decide per-symbol action and explanation.
    Returns:
        { "action": "...", "qty": N, "explain": "..." }
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
        return make_decision("hold", 0, f"No valid price for {symbol}.")

    if available_cash is not None:
        cash = float(available_cash)

    # Determine max allocation
    if total_symbols == 1:
        max_invest = cash
    elif total_symbols == 2:
        max_invest = cash * RISK_FRACTION if concurrent_buys == 2 else cash
    else:
        max_invest = cash * RISK_FRACTION

    explain = f"{symbol}: prob_up={prob_up:.3f}. "

    # Stop-loss / take-profit
    if shares > 0 and last_price > 0:
        if price <= last_price * STOP_LOSS:
            return make_decision(
                "sell",
                int(shares),
                explain + f"STOP-LOSS triggered: price {price:.2f} <= {STOP_LOSS*100:.1f}% of last buy {last_price:.2f}."
            )
        if price >= last_price * TAKE_PROFIT:
            return make_decision(
                "sell",
                int(shares),
                explain + f"TAKE-PROFIT triggered: price {price:.2f} >= {TAKE_PROFIT*100:.1f}% of last buy {last_price:.2f}."
            )

    # BUY decision
    if prob_up >= BUY_THRESHOLD:
        affordable = int(cash // price)
        target = int(max_invest // price)
        qty = min(affordable, target)

        if qty > 0:
            return make_decision(
                "buy",
                qty,
                explain + f"BUY signal (≥{BUY_THRESHOLD}). cash={cash:.2f}, price={price:.2f}, qty={qty}."
            )
        return make_decision("hold", 0, explain + "BUY signal detected but insufficient cash.")

    # SELL decision
    if prob_up <= SELL_THRESHOLD and shares > 0:
        return make_decision(
            "sell",
            int(shares),
            explain + f"SELL signal (≤{SELL_THRESHOLD}). Selling all {shares} shares."
        )

    return make_decision("hold", 0, explain + "Within thresholds → HOLD.")


# =====================================================================
# NVDA-priority multi-symbol strategy coordinator
# =====================================================================
def compute_strategy_decisions(predictions: Dict[str, float], symbols: List[str] = None):
    """
    Coordinates trading decisions for multiple symbols with NVDA priority rules.
    Returns:
        { "AAPL": {"action": "...", "qty": ..., "explain": "..."} , ... }
    """

    if symbols is None:
        symbols = list(predictions.keys())

    symbols = [s.upper() for s in symbols]
    preds = {k.upper(): v for k, v in predictions.items()}

    # Initialize all decisions as HOLD
    decisions = {
        s: make_decision("hold", 0, f"{s}: initial default = HOLD")
        for s in symbols
    }

    # If no NVDA → normal
    if "NVDA" not in symbols:
        concurrent_buys = sum(1 for s in symbols if preds.get(s, 0) >= BUY_THRESHOLD)
        for s in symbols:
            decisions[s] = should_trade(
                symbol=s,
                prob_up=preds.get(s, 0),
                total_symbols=len(symbols),
                concurrent_buys=concurrent_buys
            )
        return decisions

    # Refresh all PMs
    pms = {}
    for s in symbols:
        pm = PortfolioManager(s)
        try:
            pm.refresh_live()
        except Exception:
            pass
        pms[s] = pm

    def shares(sym): return float(pms[sym].data.get("shares", 0.0))
    def cash(sym): return float(pms[sym].data.get("cash", 0.0))

    # Grab prices
    prices = {s: fetch_latest_price(s) or 0.0 for s in symbols}

    # Base NVDA action
    nvda_prob = preds.get("NVDA", 0.0)
    nvda_base = should_trade(
        symbol="NVDA",
        prob_up=nvda_prob,
        total_symbols=len(symbols),
        concurrent_buys=sum(1 for s in symbols if preds.get(s, 0) >= BUY_THRESHOLD)
    )
    nvda_action = nvda_base["action"]

    # ============================================================
    # RULE 1 — NVDA BUY → liquidate others → buy NVDA with all cash
    # ============================================================
    if nvda_action == "buy":
        explanation = "NVDA BUY priority → liquidate all others and deploy all cash into NVDA."

        sim_cash = cash("NVDA")

        # Liquidate others
        for s in symbols:
            if s != "NVDA":
                if shares(s) > 0 and prices[s] > 0:
                    sim_cash += shares(s) * prices[s]
                    decisions[s] = make_decision(
                        "sell",
                        int(shares(s)),
                        f"{s}: Liquidated to free cash for NVDA priority BUY."
                    )
                else:
                    decisions[s] = make_decision("hold", 0, f"{s}: No action under NVDA priority BUY.")

        nvda_price = prices["NVDA"]
        qty = int(sim_cash // nvda_price) if nvda_price > 0 else 0

        if qty > 0:
            decisions["NVDA"] = make_decision("buy", qty, explanation)
        else:
            decisions["NVDA"] = make_decision("hold", 0, explanation + " Not enough cash.")

        return decisions

    # ============================================================
    # RULE 2 — NVDA SELL + Other BUY → rotate into strongest BUY
    # ============================================================
    if nvda_action == "sell":
        other_buy = [s for s in symbols if s != "NVDA" and preds.get(s, 0) >= BUY_THRESHOLD]

        if other_buy:
            strongest = max(other_buy, key=lambda s: preds.get(s, 0))
            explanation = f"NVDA SELL + {strongest} BUY signal → Rotate capital into strongest BUY."

            # Sell NVDA
            nvda_sh = shares("NVDA")
            nvda_price = prices["NVDA"]
            sim_cash = cash("NVDA") + (nvda_sh * nvda_price if nvda_sh > 0 else 0)

            decisions["NVDA"] = make_decision(
                "sell",
                int(nvda_sh),
                "NVDA sold to rotate into stronger BUY."
            )

            # Buy strongest
            tgt_price = prices[strongest]
            qty = int(sim_cash // tgt_price) if tgt_price > 0 else 0

            if qty > 0:
                decisions[strongest] = make_decision(
                    "buy",
                    qty,
                    explanation
                )
            else:
                decisions[strongest] = make_decision(
                    "hold",
                    0,
                    explanation + " Insufficient cash."
                )

            # Others → normal logic
            for s in symbols:
                if s not in ["NVDA", strongest]:
                    decisions[s] = should_trade(
                        s, preds.get(s, 0),
                        total_symbols=len(symbols),
                        concurrent_buys=sum(1 for x in symbols if preds.get(x, 0) >= BUY_THRESHOLD)
                    )

            return decisions

        # NVDA SELL but no BUY opportunities → sell NVDA only
        decisions["NVDA"] = make_decision(
            "sell",
            int(shares("NVDA")),
            "NVDA SELL signal but no BUY alternatives."
        )

        for s in symbols:
            if s != "NVDA":
                decisions[s] = should_trade(
                    s, preds.get(s, 0),
                    total_symbols=len(symbols),
                    concurrent_buys=sum(1 for x in symbols if preds.get(x, 0) >= BUY_THRESHOLD)
                )
        return decisions

    # ============================================================
    # RULE 3 — ALL SELL → liquidate everything
    # ============================================================
    if all(preds.get(s, 1) <= SELL_THRESHOLD for s in symbols):
        for s in symbols:
            sh = shares(s)
            if sh > 0:
                decisions[s] = make_decision(
                    "sell",
                    int(sh),
                    f"{s}: Global SELL condition → liquidating position."
                )
            else:
                decisions[s] = make_decision("hold", 0, f"{s}: No shares to sell.")
        return decisions

    # ============================================================
    # RULE 4 — NVDA HOLD → normal strategy
    # ============================================================
    concurrent_buys = sum(1 for s in symbols if preds.get(s, 0) >= BUY_THRESHOLD)

    for s in symbols:
        decisions[s] = should_trade(
            s,
            preds.get(s, 0),
            total_symbols=len(symbols),
            concurrent_buys=concurrent_buys
        )

    return decisions