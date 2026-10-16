#!/bin/bash
set -euo pipefail

BOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BOT_DIR"

export BOT_ENV=paper
export PYTHONUNBUFFERED=1

if [ ! -x "$BOT_DIR/venv/bin/python" ]; then
    echo "Missing executable Python: $BOT_DIR/venv/bin/python" >&2
    exit 1
fi

exec "$BOT_DIR/venv/bin/python" "$BOT_DIR/run_bot.py" --env paper
