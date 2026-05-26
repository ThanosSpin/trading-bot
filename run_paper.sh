#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export BOT_ENV=paper
export PYTHONUNBUFFERED=1

mkdir -p logs_paper
python main.py 2>&1 | tee -a logs_paper/run_paper_console.log