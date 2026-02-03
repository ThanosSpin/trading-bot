#!/bin/bash
# cleanup_old_logs.sh
# Delete .log files older than 7 days from home directory

set -e

LOG_DIR="$HOME"
DAYS_OLD=7
LOG_FILE="$HOME/log_cleanup.log"

echo "============================================================" | tee -a "$LOG_FILE"
echo "LOG CLEANUP - $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Searching for .log files older than $DAYS_OLD days in: $LOG_DIR" | tee -a "$LOG_FILE"
echo "------------------------------------------------------------" | tee -a "$LOG_FILE"

# Find and delete .log files older than 7 days
# -maxdepth 1 = only search home directory (not subdirectories)
# -name "*.log" = only .log files
# -type f = only files (not directories)
# -mtime +7 = modified more than 7 days ago

FOUND_COUNT=0
DELETED_COUNT=0
DELETED_SIZE=0

while IFS= read -r -d '' file; do
    FOUND_COUNT=$((FOUND_COUNT + 1))

    # Get file size before deletion
    SIZE=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
    SIZE_MB=$(echo "scale=2; $SIZE / 1048576" | bc)

    # Get file modification date
    MOD_DATE=$(stat -f%Sm -t '%Y-%m-%d' "$file" 2>/dev/null || stat -c%y "$file" 2>/dev/null | cut -d' ' -f1)

    echo "  ✅ Deleting: $(basename "$file") (${SIZE_MB} MB, last modified: $MOD_DATE)" | tee -a "$LOG_FILE"

    rm -f "$file"
    DELETED_COUNT=$((DELETED_COUNT + 1))
    DELETED_SIZE=$((DELETED_SIZE + SIZE))

done < <(find "$LOG_DIR" -maxdepth 1 -name "*.log" -type f -mtime +$DAYS_OLD -print0 2>/dev/null)

echo "------------------------------------------------------------" | tee -a "$LOG_FILE"

if [ $DELETED_COUNT -gt 0 ]; then
    TOTAL_MB=$(echo "scale=2; $DELETED_SIZE / 1048576" | bc)
    echo "✅ Deleted $DELETED_COUNT file(s) (${TOTAL_MB} MB)" | tee -a "$LOG_FILE"
else
    echo "✅ No old log files to delete" | tee -a "$LOG_FILE"
fi

echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
