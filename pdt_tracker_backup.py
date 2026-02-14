# pdt_tracker.py
import json
import os
from datetime import datetime
import pytz  # pip install pytz

NY_TZ = pytz.timezone("America/New_York")
STATE_FILE = "pdt_state.json"

def _today_key():
    return datetime.now(NY_TZ).strftime("%Y-%m-%d")

def load_state():
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)

def get_opened_today_qty(symbol: str) -> float:
    state = load_state()
    day = _today_key()
    return float(state.get(day, {}).get(symbol, 0.0) or 0.0)

def add_opened_today(symbol: str, qty: float):
    if qty <= 0:
        return
    state = load_state()
    day = _today_key()
    state.setdefault(day, {})
    state[day][symbol] = float(state[day].get(symbol, 0.0) or 0.0) + float(qty)
    save_state(state)

def reduce_opened_today(symbol: str, qty: float):
    if qty <= 0:
        return
    state = load_state()
    day = _today_key()
    state.setdefault(day, {})
    cur = float(state[day].get(symbol, 0.0) or 0.0)
    new = max(0.0, cur - float(qty))
    state[day][symbol] = new
    save_state(state)