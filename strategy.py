# strategy.py
from typing import Dict, List, Tuple
from config import BUY_THRESHOLD, SELL_THRESHOLD, STOP_LOSS, TAKE_PROFIT, RISK_FRACTION
from portfolio import PortfolioManager
from data_loader import fetch_latest_price

# ---------------------------------------------------------------------
# Per-symbol decision function (original logic, updated a bit)
# ---------------------------------------------------------------------
def should_trade(
    symbol: str,
    prob_up: float,
    total_symbols: int = 1,
    concurrent_buys: int = 1,
    available_cash: float = None
) -> Tuple[str, int]:
    """
    Decide action for a symbol using your original rules plus optional
    available_cash override.

    Returns:
        (action, qty) where action in {"buy","sell","hold"}
    """
    pm = PortfolioManager(symbol)
    # Prefer live values where possible
    try:
        pm.refresh_live()
    except Exception:
        pass

    shares = float(pm.data.get("shares", 0.0))
    cash = float(pm.data.get("cash", 0.0))
    last_price = float(pm.data.get("last_price", 0.0))
    price = fetch_latest_price(symbol)

    if price is None or price <= 0:
        # can't trade without price
        return ("hold", 0)

    if available_cash is not None:
        cash = float(available_cash)

    # compute allocation
    if total_symbols == 1:
        max_invest = cash
    elif total_symbols == 2:
        max_invest = cash * RISK_FRACTION if concurrent_buys == 2 else cash
    else:
        max_invest = cash * RISK_FRACTION

    # Stop-loss / Take-profit (based on last buy price tracked in pm.data["last_price"])
    if shares > 0 and last_price > 0:
        if price <= last_price * STOP_LOSS:
            return ("sell", int(shares))
        if price >= last_price * TAKE_PROFIT:
            return ("sell", int(shares))

    # BUY logic
    if prob_up >= BUY_THRESHOLD:
        affordable = int(cash // price)
        target = int(max_invest // price)
        quantity = min(affordable, target)
        if quantity > 0:
            return ("buy", quantity)
        return ("hold", 0)

    # SELL logic
    if prob_up <= SELL_THRESHOLD and shares > 0:
        return ("sell", int(shares))

    return ("hold", 0)


# ---------------------------------------------------------------------
# NVDA-priority multi-symbol coordinator
# ---------------------------------------------------------------------
def compute_strategy_decisions(
    predictions: Dict[str, float],
    symbols: List[str] = None
) -> Dict[str, Tuple[str, int]]:
    """
    Return decisions for each symbol in symbols (or predictions.keys()).
    Uses NVDA priority rules and calculates concrete quantities using live state
    via PortfolioManager (no orders are executed here).
    """

    if symbols is None:
        symbols = list(predictions.keys())

    symbols = [s.upper() for s in symbols]
    preds = {k.upper(): v for k, v in predictions.items()}

    decisions: Dict[str, Tuple[str, int]] = {s: ("hold", 0) for s in symbols}

    # If NVDA not present -> fallback to normal strategy
    if "NVDA" not in symbols:
        # compute concurrent buys to pass to should_trade
        buy_count = sum(1 for s in symbols if preds.get(s, 0) >= BUY_THRESHOLD)
        for s in symbols:
            decisions[s] = should_trade(
                symbol=s,
                prob_up=preds.get(s, 0),
                total_symbols=len(symbols),
                concurrent_buys=buy_count
            )
        return decisions

    # refresh all PortfolioManagers and gather live state
    pms = {}
    for s in symbols:
        pm = PortfolioManager(s)
        try:
            pm.refresh_live()
        except Exception:
            # if refresh fails, fallback to whatever is on disk
            pass
        pms[s] = pm

    # convenience getters
    def live_shares(sym):
        return float(pms[sym].data.get("shares", 0.0))

    def live_cash(sym):
        return float(pms[sym].data.get("cash", 0.0))

    # compute live market prices for symbols (needed to simulate sells)
    prices = {}
    for s in symbols:
        prices[s] = fetch_latest_price(s)
        if prices[s] is None:
            # if price missing, set to 0 and avoid buying
            prices[s] = 0.0

    # Evaluate NVDA base action using should_trade (this gives a "normal" action)
    nvda_prob = preds.get("NVDA", 0.0)
    nvda_action, nvda_qty_guess = should_trade(
        symbol="NVDA",
        prob_up=nvda_prob,
        total_symbols=len(symbols),
        concurrent_buys=sum(1 for s in symbols if preds.get(s, 0) >= BUY_THRESHOLD)
    )

    # -------------------------
    # RULE 1: NVDA BUY -> sell others and buy NVDA with full available cash
    # -------------------------
    if nvda_action == "buy":
        # sell all other symbol shares (simulate cash inflow)
        simulated_cash = live_cash("NVDA")  # starting cash in NVDA account view
        for s in symbols:
            if s == "NVDA":
                continue
            s_shares = live_shares(s)
            s_price = prices.get(s, 0.0)
            if s_shares > 0 and s_price > 0:
                # if we sold these, we'd receive:
                simulated_cash += s_shares * s_price
                decisions[s] = ("sell", int(s_shares))
            else:
                decisions[s] = ("hold", 0)

        # compute NVDA qty using simulated_cash and NVDA price
        nvda_price = prices.get("NVDA", 0.0)
        if nvda_price > 0:
            nvda_qty = int(simulated_cash // nvda_price)
            if nvda_qty > 0:
                decisions["NVDA"] = ("buy", nvda_qty)
            else:
                decisions["NVDA"] = ("hold", 0)
        else:
            decisions["NVDA"] = ("hold", 0)

        return decisions

    # -------------------------
    # RULE 2: NVDA SELL + Other BUY -> sell NVDA and buy the strongest other
    # -------------------------
    if nvda_action == "sell":
        # find other symbols with buy signal
        other_buyers = [s for s in symbols if s != "NVDA" and preds.get(s, 0) >= BUY_THRESHOLD]

        if other_buyers:
            # choose highest-probability buy candidate
            target = max(other_buyers, key=lambda x: preds.get(x, 0))
            # simulate cash if we sell NVDA
            nvda_sh = live_shares("NVDA")
            nvda_price = prices.get("NVDA", 0.0)
            simulated_cash = live_cash("NVDA") + (nvda_sh * nvda_price if nvda_price > 0 else 0.0)

            # compute how many shares of target we can buy with simulated_cash
            target_price = prices.get(target, 0.0)
            if target_price > 0:
                target_qty = int(simulated_cash // target_price)
            else:
                target_qty = 0

            # set decisions
            if nvda_sh > 0:
                decisions["NVDA"] = ("sell", int(nvda_sh))
            else:
                decisions["NVDA"] = ("hold", 0)

            if target_qty > 0:
                decisions[target] = ("buy", target_qty)
            else:
                decisions[target] = ("hold", 0)

            # other symbols not involved hold or sell their own signals
            for s in symbols:
                if s not in decisions:
                    # allow normal should_trade for them
                    decisions[s] = should_trade(
                        symbol=s,
                        prob_up=preds.get(s, 0),
                        total_symbols=len(symbols),
                        concurrent_buys=sum(1 for x in symbols if preds.get(x, 0) >= BUY_THRESHOLD)
                    )

            return decisions
        else:
            # NVDA wants to sell but no other buy candidates — sell NVDA and let others follow normal logic
            nvda_sh = live_shares("NVDA")
            if nvda_sh > 0:
                decisions["NVDA"] = ("sell", int(nvda_sh))
            else:
                decisions["NVDA"] = ("hold", 0)

            for s in symbols:
                if s == "NVDA":
                    continue
                decisions[s] = should_trade(
                    symbol=s,
                    prob_up=preds.get(s, 0),
                    total_symbols=len(symbols),
                    concurrent_buys=sum(1 for x in symbols if preds.get(x, 0) >= BUY_THRESHOLD)
                )
            return decisions

    # -------------------------
    # RULE 3: Both SELL -> sell all
    # (Interpretation: if most symbols want to sell, we sell all positions)
    # -------------------------
    all_sell_signals = all(preds.get(s, 0) <= SELL_THRESHOLD for s in symbols)
    if all_sell_signals:
        for s in symbols:
            sh = live_shares(s)
            if sh > 0:
                decisions[s] = ("sell", int(sh))
            else:
                decisions[s] = ("hold", 0)
        return decisions

    # -------------------------
    # RULE 4: NVDA HOLD -> normal strategy
    # -------------------------
    # compute concurrent buy count for allocation
    concurrent_buys = sum(1 for s in symbols if preds.get(s, 0) >= BUY_THRESHOLD)

    for s in symbols:
        decisions[s] = should_trade(
            symbol=s,
            prob_up=preds.get(s, 0),
            total_symbols=len(symbols),
            concurrent_buys=concurrent_buys
        )

    return decisions