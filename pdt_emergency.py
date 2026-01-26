import os, json
from datetime import datetime
from pathlib import Path

STATE_FILE = Path("data/pdt_emergency.json")

def _today_key():
    return datetime.utcnow().strftime("%Y-%m-%d")

def get_emergency_used_today() -> int:
    if not STATE_FILE.exists():
        return 0
    try:
        d = json.loads(STATE_FILE.read_text())
        return int(d.get(_today_key(), 0))
    except Exception:
        return 0

def inc_emergency_used_today():
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    d = {}
    if STATE_FILE.exists():
        try:
            d = json.loads(STATE_FILE.read_text())
        except Exception:
            d = {}
    k = _today_key()
    d[k] = int(d.get(k, 0)) + 1
    STATE_FILE.write_text(json.dumps(d))