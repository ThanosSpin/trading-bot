#!/usr/bin/env bash

# Recursively delete old .log files from this checkout's logs directory.

set -Eeuo pipefail

BOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-$BOT_DIR/logs}"
DAYS_OLD="${DAYS_OLD:-7}"
REPORT_LOG="${REPORT_LOG:-$BOT_DIR/log_cleanup.log}"

if ! [[ "$DAYS_OLD" =~ ^[0-9]+$ ]]; then
    echo "Invalid DAYS_OLD value: $DAYS_OLD" >&2
    exit 2
fi

mkdir -p "$LOG_DIR"
touch "$REPORT_LOG"

log() {
    printf '%s\n' "$1" | tee -a "$REPORT_LOG"
}

file_size_bytes() {
    stat -c '%s' "$1" 2>/dev/null || stat -f '%z' "$1" 2>/dev/null || printf '0\n'
}

file_modified_date() {
    stat -c '%y' "$1" 2>/dev/null | cut -d' ' -f1 \
        || stat -f '%Sm' -t '%Y-%m-%d' "$1" 2>/dev/null \
        || printf 'unknown\n'
}

human_size() {
    local bytes="$1"
    if (( bytes >= 1048576 )); then
        printf '%d MB' "$((bytes / 1048576))"
    else
        printf '%d KB' "$((bytes / 1024))"
    fi
}

deleted_count=0
deleted_size=0

log "============================================================"
log "LOG CLEANUP - $(date '+%Y-%m-%d %H:%M:%S %Z')"
log "Directory: $LOG_DIR"
log "Deleting recursive *.log files older than $DAYS_OLD days"
log "------------------------------------------------------------"

while IFS= read -r -d '' file; do
    size="$(file_size_bytes "$file")"
    modified="$(file_modified_date "$file")"

    log "Deleting: $file ($(human_size "$size"), modified: $modified)"
    rm -f -- "$file"

    deleted_count=$((deleted_count + 1))
    deleted_size=$((deleted_size + size))
done < <(
    find "$LOG_DIR" \
        -type f \
        -name '*.log' \
        -mtime "+$DAYS_OLD" \
        -print0
)

log "------------------------------------------------------------"
log "Deleted $deleted_count file(s), total $(human_size "$deleted_size")"
log "============================================================"
log ""
