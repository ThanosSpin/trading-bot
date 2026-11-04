"""Reconcile linked funding intents before any order is submitted.

Callbacks provide validated entry eligibility and current position limits. This
module only returns decisions; it cannot submit orders or read account state.
"""
from nvda_funding import cancel_orphaned_funding_sells, plan_entry_funding


def prepare_funding_orders(decisions, *, cash, prices, positions, validate_buy, limit_buy,
                           cash_fraction=1.0):
    result = cancel_orphaned_funding_sells(decisions)
    targets = {d.get("funding_target") for d in result.values()
               if d.get("action") == "sell" and d.get("funding_sell")
               and not (d.get("risk_exit") or d.get("independent_exit"))}
    targets.update(sym for sym, d in result.items() if d.get("action") == "buy")
    budget = max(0.0, float(cash))
    # Independent exits are not canceled if funding becomes unnecessary.
    budget += sum(min(int(d.get("qty", 0)), int(max(0, positions.get(sym, 0))))
                  * max(0.0, prices.get(sym, 0))
                  for sym, d in result.items() if d.get("action") == "sell"
                  and (not d.get("funding_sell") or d.get("risk_exit") or d.get("independent_exit")))
    for target in sorted(targets, key=lambda sym: (result.get(sym, {}).get("priority_rank", 999), sym or "")):
        buy = result.get(target, {})
        source_symbols = [sym for sym, d in result.items()
                          if d.get("action") == "sell" and d.get("funding_sell")
                          and d.get("funding_target") == target
                          and not (d.get("risk_exit") or d.get("independent_exit"))]
        sources = [dict(symbol=sym,
                        shares=min(int(result[sym]["qty"]), int(max(0, positions.get(sym, 0)))),
                        price=max(0.0, prices.get(sym, 0))) for sym in source_symbols]
        potential = budget + sum(source["shares"] * source["price"] for source in sources)
        linked = bool(source_symbols or buy.get("funding_sources"))
        valid = buy.get("action") == "buy" and (not linked or validate_buy(target, buy))
        qty = min(int(buy.get("max_buy_qty", buy.get("qty", 0))), int(buy.get("qty", 0)))
        qty = max(0, limit_buy(target, buy, qty, potential)) if valid else 0
        plan = plan_entry_funding(target=target, target_qty=qty,
            target_price=max(0.0, prices.get(target, 0)), cash=budget,
            sources=sources, target_validated=valid, cash_fraction=cash_fraction)
        if not plan["funded"]:
            price = prices.get(target, 0)
            qty = min(qty, int(budget // price)) if price > 0 else 0
        for sym in source_symbols:
            result[sym] = dict(action="hold", qty=0,
                               explain=f"{sym}: funding-only SELL no longer needed.")
        if qty <= 0:
            result[target] = dict(action="hold", qty=0,
                                 explain=f"{target}: funding BUY failed final validation or sizing.")
            continue
        for sale in plan["sales"]:
            sym = sale["symbol"]
            result[sym] = dict(decisions[sym], qty=sale["qty"])
            budget += sale["qty"] * prices[sym]
        result[target] = dict(buy, qty=qty, max_buy_qty=qty)
        budget = max(0.0, budget - qty * prices[target])
    return cancel_orphaned_funding_sells(result)
