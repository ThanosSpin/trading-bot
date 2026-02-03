#!/bin/bash
# install_cron.sh - Quick cron job installer

set -e

echo "============================================================"
echo "CRON JOB INSTALLER - Log Cleanup"
echo "============================================================"
echo ""

# Get current directory
BOT_DIR=$(pwd)
echo "📁 Bot directory: $BOT_DIR"

# Find python
PYTHON_PATH=$(which python3)
echo "🐍 Python path: $PYTHON_PATH"

# Test cleanup script
echo ""
echo "🧪 Testing cleanup script..."
$PYTHON_PATH cleanup_logs.py

if [ $? -ne 0 ]; then
    echo "❌ Cleanup script failed. Fix errors before installing cron job."
    exit 1
fi

echo ""
echo "✅ Cleanup script works!"
echo ""

# Create cron job command
CRON_CMD="0 3 * * * cd $BOT_DIR && $PYTHON_PATH cleanup_logs.py >> $BOT_DIR/logs/cleanup.log 2>&1"

echo "📋 Cron job to be added:"
echo "   $CRON_CMD"
echo ""

# Check if already exists
if crontab -l 2>/dev/null | grep -q "cleanup_logs.py"; then
    echo "⚠️ Cron job already exists!"
    echo ""
    echo "Current crontab:"
    crontab -l | grep cleanup_logs.py
    echo ""
    read -p "Replace existing job? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi

    # Remove old job
    crontab -l | grep -v "cleanup_logs.py" | crontab -
fi

# Add new job
(crontab -l 2>/dev/null; echo "$CRON_CMD") | crontab -

echo ""
echo "✅ Cron job installed successfully!"
echo ""
echo "📋 Current crontab:"
crontab -l
echo ""
echo "============================================================"
echo "Cron job will run daily at 3:00 AM"
echo "Logs will be saved to: $BOT_DIR/logs/cleanup.log"
echo "============================================================"
