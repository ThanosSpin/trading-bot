# pdt_guardrails.py
from pdt_tracker import get_opened_today_qty

def max_sell_allowed(api, symbol: str, requested_qty: float, pdt_status: dict, emergency: bool = False) -> float:
    """
    Returns the maximum SELL qty allowed under PDT rules.
    0.0 means SELL must be blocked entirely.
    emergency=True allows same-day exits (may consume a day trade / DTBP).
    """
    symU = str(symbol).upper().strip()

    try:
        requested_qty = float(requested_qty)
    except Exception:
        requested_qty = 0.0

    if requested_qty <= 0:
        return 0.0

    if not pdt_status:
        return requested_qty  # fail-open

    eq = float(pdt_status.get("equity", 0) or 0)
    dt = int(pdt_status.get("daytrade_count", 0) or 0)

    # PDT irrelevant
    if eq >= 25000 or dt < 3:
        return requested_qty

    # Shares opened today (tracker)
    opened_today = float(get_opened_today_qty(symU) or 0.0)

    # Broker position
    try:
        pos = api.get_position(symU)
        position_qty = float(getattr(pos, "qty", 0) or 0)
    except Exception:
        position_qty = 0.0

    if position_qty <= 0:
        # Nothing to sell
        return 0.0

    # Shares that are "safe" to sell (not opened today)
    sellable = max(0.0, position_qty - opened_today)

    if sellable <= 0:
        if emergency:
            # Allow same-day exit only in emergency mode
            allowed = min(requested_qty, position_qty)
            if allowed > 0:
                print(
                    f"[PDT EMERGENCY] Allowing same-day SELL for {symU}: "
                    f"requested={requested_qty:g}, allowed={allowed:g} "
                    f"(position={position_qty:g}, opened_today={opened_today:g})"
                )
                return allowed

        print(
            f"[PDT GUARD] SELL blocked {symU}: "
            f"position={position_qty:g}, opened_today={opened_today:g}"
        )
        return 0.0

    if requested_qty > sellable:
        print(
            f"[PDT GUARD] SELL capped {symU}: "
            f"requested={requested_qty:g}, allowed={sellable:g} "
            f"(opened_today={opened_today:g})"
        )
        return sellable

    return requested_qty