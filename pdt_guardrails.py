# pdt_guardrails.py
from pdt_tracker import opened_today_qty

def max_sell_allowed(api, symbol: str, requested_qty: float, pdt_status: dict) -> float:
    """
    Returns the maximum SELL qty allowed under PDT rules.
    0.0 means SELL must be blocked entirely.
    """
    if not pdt_status:
        return requested_qty  # fail-open

    eq = float(pdt_status.get("equity", 0) or 0)
    dt = int(pdt_status.get("daytrade_count", 0) or 0)

    # PDT irrelevant
    if eq >= 25000 or dt < 3:
        return requested_qty

    # Shares opened today
    opened_today = opened_today_qty(symbol)

    try:
        pos = api.get_position(symbol)
        position_qty = float(pos.qty)
    except Exception:
        position_qty = 0.0

    # Shares that are safe to sell (not opened today)
    sellable = max(0.0, position_qty - opened_today)

    if sellable <= 0:
        print(
            f"[PDT GUARD] SELL blocked {symbol}: "
            f"position={position_qty:g}, opened_today={opened_today:g}"
        )
        return 0.0

    if requested_qty > sellable:
        print(
            f"[PDT GUARD] SELL capped {symbol}: "
            f"requested={requested_qty:g}, allowed={sellable:g} "
            f"(opened_today={opened_today:g})"
        )
        return sellable

    return requested_qty