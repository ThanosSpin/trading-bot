#!/bin/bash
# install_log_cleanup_cron.sh
# Install cron job to delete .log files older than 7 days

set -e

echo "============================================================"
echo "CRON JOB INSTALLER - Delete Old Log Files"
echo "============================================================"
echo ""

# Get home directory
HOME_DIR="$HOME"
SCRIPT_PATH="$HOME_DIR/cleanup_old_logs.sh"

echo "📁 Home directory: $HOME_DIR"
echo "📄 Cleanup script: $SCRIPT_PATH"

# Check if script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "❌ Error: cleanup_old_logs.sh not found in home directory"
    echo "   Please ensure the script is at: $SCRIPT_PATH"
    exit 1
fi

# Make script executable
chmod +x "$SCRIPT_PATH"

# Test the script
echo ""
echo "🧪 Testing cleanup script..."
bash "$SCRIPT_PATH"

if [ $? -ne 0 ]; then
    echo "❌ Cleanup script failed. Fix errors before installing cron job."
    exit 1
fi

echo ""
echo "✅ Cleanup script works!"
echo ""

# Create cron job command (runs daily at 3 AM)
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
    echo "Current crontab:"
    crontab -l | grep cleanup_old_logs.sh
    echo ""
    read -p "Replace existing job? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi

    # Remove old job
    crontab -l | grep -v "cleanup_old_logs.sh" | crontab -
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
echo "✅ Setup complete!"
echo "============================================================"
echo ""
echo "Cron job will run daily at 3:00 AM"
echo "Cleanup logs saved to: $HOME_DIR/log_cleanup.log"
echo ""
echo "To verify: crontab -l"
echo "To view cleanup history: cat ~/log_cleanup.log"
echo "============================================================"
