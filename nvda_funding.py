"""Pure capped-entry funding calculations; no broker or market-data access."""
from math import ceil, isfinite


def _nonnegative(value):
    number = float(value)
    if not isfinite(number) or number < 0:
        raise ValueError("Expected a finite nonnegative value")
    return number


def plan_entry_funding(*, target, target_qty, target_price, cash, sources, target_validated,
                       cash_fraction=1.0):
    """Plan whole-share sales only for a validated, already capped BUY.

    Sources are ordered by the caller's explicit funding priority and contain
    symbol, shares, price, and independently_buying. AAPL BUYs are protected.
    Independent exits must be handled separately; do not include their shares
    again here. No sales are emitted if the complete shortfall cannot be funded.
    """
    qty = int(_nonnegative(target_qty))
    target = str(target).upper()
    price = _nonnegative(target_price)
    available = _nonnegative(cash)
    if not target_validated or qty == 0 or price == 0:
        return {"target_qty": 0, "sales": [], "shortfall": 0.0, "funded": False}
    fraction = _nonnegative(cash_fraction)
    if not 0 < fraction <= 1:
        raise ValueError("cash_fraction must be in (0, 1]")
    # Preserve the configured cash reserve required by the position limiter.
    shortfall = max(0.0, qty * price / fraction - available)
    remaining = shortfall
    sales = []
    seen = set()
    for source in sources:
        symbol = str(source["symbol"]).upper()
        if symbol in seen:
            raise ValueError("Duplicate funding source")
        seen.add(symbol)
        if symbol == target or source.get("independently_buying"):
            continue
        shares = int(_nonnegative(source["shares"]))
        source_price = _nonnegative(source["price"])
        if remaining <= 0 or shares == 0 or source_price == 0:
            continue
        sell_qty = min(shares, ceil(remaining / source_price))
        sales.append({"symbol": symbol, "qty": sell_qty, "funding_target": target})
        remaining = max(0.0, remaining - sell_qty * source_price)
    funded = remaining <= 0
    return {
        "target_qty": qty if funded else 0,
        "sales": sales if funded else [],
        "shortfall": shortfall,
        "funded": funded,
    }


def nvda_qty_after_fills(*, original_capped_qty, refreshed_cash, nvda_price, revalidated):
    """Bound a BUY by actual refreshed cash and the original approved quantity.

    The executor must revalidate guards and current position limits before using
    this quantity. This function never assumes that a planned sale has filled.
    """
    cap = int(_nonnegative(original_capped_qty))
    cash = _nonnegative(refreshed_cash)
    price = _nonnegative(nvda_price)
    if not revalidated or price == 0:
        return 0
    return min(cap, int(cash // price))


def plan_nvda_funding(*, target_qty, nvda_price, cash, sources, nvda_validated):
    return plan_entry_funding(target="NVDA", target_qty=target_qty,
        target_price=nvda_price, cash=cash, sources=sources,
        target_validated=nvda_validated)


def cancel_orphaned_funding_sells(decisions):
    """Cancel only explicitly linked funding sales whose BUY no longer survives."""
    result = {symbol: dict(decision) for symbol, decision in decisions.items()}
    for symbol, decision in result.items():
        if decision.get("action") != "sell" or not decision.get("funding_sell"):
            continue
        # Independent risk exits always take precedence over a funding link.
        if decision.get("risk_exit") or decision.get("independent_exit"):
            continue
        target = result.get(decision.get("funding_target"), {})
        if target.get("action") != "buy" or int(target.get("qty", 0) or 0) <= 0:
            result[symbol] = dict(action="hold", qty=0,
                explain=f"{symbol}: funding-only SELL canceled; destination BUY blocked.")
    return result
