# pdt_tracker.py
import json, os
from datetime import datetime
import pytz  # pip install pytz

TZ = pytz.timezone("Europe/Athens")
STATE_FILE = "pdt_state.json"

def _today_key():
    return datetime.now(TZ).strftime("%Y-%m-%d")

def _load():
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)

def opened_today_qty(symbol: str) -> float:
    state = _load()
    return float(state.get(_today_key(), {}).get(symbol, 0.0) or 0.0)

def add_opened_today(symbol: str, qty: float):
    if qty <= 0:
        return
    state = _load()
    day = _today_key()
    state.setdefault(day, {})
    state[day][symbol] = float(state[day].get(symbol, 0.0) or 0.0) + float(qty)
    _save(state)

def reduce_opened_today(symbol: str, qty: float):
    if qty <= 0:
        return
    state = _load()
    day = _today_key()
    state.setdefault(day, {})
    cur = float(state[day].get(symbol, 0.0) or 0.0)
    state[day][symbol] = max(0.0, cur - float(qty))
    _save(state)