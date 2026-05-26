#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export BOT_ENV=live
export PYTHONUNBUFFERED=1

mkdir -p logs
python main.py 2>&1 | tee -a logs/run_live_console.log