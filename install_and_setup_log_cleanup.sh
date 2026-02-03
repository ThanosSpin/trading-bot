#!/bin/bash
# install_and_setup_log_cleanup.sh
# All-in-one script: Creates cleanup script AND installs cron job

set -e

echo "============================================================"
echo "LOG CLEANUP CRON JOB - ONE-CLICK INSTALLER"
echo "============================================================"
echo ""

# Define paths
HOME_DIR="$HOME"
SCRIPT_PATH="$HOME_DIR/cleanup_old_logs.sh"

echo "📁 Home directory: $HOME_DIR"
echo "📄 Will create: $SCRIPT_PATH"
echo ""

# ============================================================
# STEP 1: Create the cleanup script
# ============================================================
echo "📝 Creating cleanup script..."

cat > "$SCRIPT_PATH" << 'CLEANUP_SCRIPT_EOF'
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

FOUND_COUNT=0
DELETED_COUNT=0
DELETED_SIZE=0

while IFS= read -r -d '' file; do
    FOUND_COUNT=$((FOUND_COUNT + 1))

    # Get file size before deletion (in bytes)
    SIZE=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null)

    # Convert to MB using bash arithmetic
    SIZE_MB=$((SIZE / 1048576))
    SIZE_KB=$((SIZE / 1024))

    # Display in appropriate unit
    if [ $SIZE_MB -gt 0 ]; then
        SIZE_DISPLAY="${SIZE_MB} MB"
    else
        SIZE_DISPLAY="${SIZE_KB} KB"
    fi

    # Get file modification date
    MOD_DATE=$(stat -c%y "$file" 2>/dev/null | cut -d' ' -f1 || stat -f%Sm -t '%Y-%m-%d' "$file" 2>/dev/null)

    echo "  ✅ Deleting: $(basename "$file") ($SIZE_DISPLAY, last modified: $MOD_DATE)" | tee -a "$LOG_FILE"

    rm -f "$file"
    DELETED_COUNT=$((DELETED_COUNT + 1))
    DELETED_SIZE=$((DELETED_SIZE + SIZE))

done < <(find "$LOG_DIR" -maxdepth 1 -name "*.log" -type f -mtime +$DAYS_OLD -print0 2>/dev/null)

echo "------------------------------------------------------------" | tee -a "$LOG_FILE"

if [ $DELETED_COUNT -gt 0 ]; then
    TOTAL_MB=$((DELETED_SIZE / 1048576))
    TOTAL_KB=$((DELETED_SIZE / 1024))

    if [ $TOTAL_MB -gt 0 ]; then
        echo "✅ Deleted $DELETED_COUNT file(s) (${TOTAL_MB} MB)" | tee -a "$LOG_FILE"
    else
        echo "✅ Deleted $DELETED_COUNT file(s) (${TOTAL_KB} KB)" | tee -a "$LOG_FILE"
    fi
else
    echo "✅ No old log files to delete" | tee -a "$LOG_FILE"
fi

echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
CLEANUP_SCRIPT_EOF

# Make script executable
chmod +x "$SCRIPT_PATH"

echo "✅ Created cleanup script at: $SCRIPT_PATH"
echo ""

# ============================================================
# STEP 2: Test the cleanup script
# ============================================================
echo "🧪 Testing cleanup script..."
echo ""
bash "$SCRIPT_PATH"

if [ $? -ne 0 ]; then
    echo "❌ Cleanup script failed. Please check errors above."
    exit 1
fi

echo ""
echo "✅ Cleanup script works!"
echo ""

# ============================================================
# STEP 3: Install cron job
# ============================================================
CRON_CMD="0 3 * * * bash $SCRIPT_PATH"

echo "📋 Cron job to be added:"
echo "   $CRON_CMD"
echo ""
echo "   Schedule: Daily at 3:00 AM"
echo "   Deletes: *.log files older than 7 days"
echo "   Location: $HOME_DIR"
echo ""

# Check if already exists
if crontab -l 2>/dev/null | grep -q "cleanup_old_logs.sh"; then
    echo "⚠️ Cron job already exists!"
    echo ""
    echo "Current crontab entry:"
    crontab -l | grep cleanup_old_logs.sh
    echo ""
    read -p "Replace existing job? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled. Cleanup script is ready but cron job not modified."
        exit 0
    fi

    # Remove old job
    crontab -l | grep -v "cleanup_old_logs.sh" | crontab -
    echo "Removed old cron job"
fi

# Add new job
(crontab -l 2>/dev/null; echo "$CRON_CMD") | crontab -

echo ""
echo "✅ Cron job installed successfully!"
echo ""
echo "============================================================"
echo "✅ SETUP COMPLETE!"
echo "============================================================"
echo ""
echo "📋 Installed cron job:"
crontab -l | grep cleanup_old_logs.sh
echo ""
echo "⏰ Schedule: Daily at 3:00 AM"
echo "🗑️  Deletes: .log files older than 7 days from $HOME_DIR"
echo "📝 Cleanup logs: $HOME_DIR/log_cleanup.log"
echo ""
echo "To verify: crontab -l"
echo "To view history: cat ~/log_cleanup.log"
echo "To uninstall: crontab -e (then delete the line)"
echo "============================================================"
