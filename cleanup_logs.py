#!/usr/bin/env python3
# cleanup_logs.py
"""
Cleanup old log files and prediction logs.
Run this daily via cron to keep disk usage under control.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add bot directory to path
BOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BOT_DIR)

def cleanup_logs_directory(logs_dir: str = "logs", days_to_keep: int = 30):
    """
    Recursively remove files older than specified days from logs/ directory.

    Nested account directories such as ``logs/live`` and ``logs/live2`` are
    included. Symbolic links are skipped so cleanup cannot follow a link out of
    the configured logs tree.

    Args:
        logs_dir: Path to logs directory
        days_to_keep: Keep logs from last N days
    """
    logs_path = Path(logs_dir)
    if not logs_path.is_absolute():
        logs_path = Path(BOT_DIR) / logs_path

    if not logs_path.exists():
        print(f"[INFO] Logs directory not found: {logs_path}")
        return

    cutoff = datetime.now() - timedelta(days=days_to_keep)
    removed_count = 0
    removed_size = 0

    print(f"\n🧹 Cleaning up logs older than {days_to_keep} days...")
    print(f"   Cutoff date: {cutoff.strftime('%Y-%m-%d')}")
    print(f"   Directory: {logs_path}")
    print("-" * 60)

    for filepath in sorted(logs_path.rglob("*")):
        if filepath.is_symlink() or not filepath.is_file():
            continue

        relative_path = filepath.relative_to(logs_path)

        try:
            # Get file modification time
            file_mtime = datetime.fromtimestamp(filepath.stat().st_mtime)

            if file_mtime < cutoff:
                file_size = filepath.stat().st_size
                filepath.unlink()
                removed_count += 1
                removed_size += file_size
                print(
                    f"  ✅ Removed: {relative_path} "
                    f"({file_size/1024:.1f} KB, {file_mtime.strftime('%Y-%m-%d')})"
                )

        except Exception as e:
            print(f"  ⚠️ Error removing {relative_path}: {e}")

    if removed_count > 0:
        print("-" * 60)
        print(f"✅ Removed {removed_count} files ({removed_size/1024/1024:.2f} MB)")
    else:
        print("✅ No old files to remove")


def cleanup_prediction_logs(days_to_keep: int = 30):
    """
    Cleanup old prediction logs using ModelMonitor.

    Args:
        days_to_keep: Keep predictions from last N days
    """
    print(f"\n🧹 Cleaning up prediction logs older than {days_to_keep} days...")
    print("-" * 60)

    try:
        from predictive_model.model_monitor import ModelMonitor

        monitor = ModelMonitor()
        monitor.cleanup_old_logs(days_to_keep=days_to_keep)
        print("✅ Prediction logs cleaned up")
    except Exception as e:
        print(f"⚠️ Could not cleanup prediction logs: {e}")


def cleanup_old_model_backups(model_dir: str = "models", max_backups: int = 6):
    """
    Keep only the most recent N backup files per symbol/mode.

    Backup files follow pattern: SYMBOL_MODE_xgb_YYYY-MM.pkl
    Active files: SYMBOL_MODE_xgb.pkl (no date)

    Args:
        model_dir: Path to models directory
        max_backups: Keep this many backups per symbol/mode
    """
    models_path = os.path.join(BOT_DIR, model_dir)

    if not os.path.exists(models_path):
        print(f"[INFO] Models directory not found: {models_path}")
        return

    print(f"\n🧹 Cleaning up old model backups (keep {max_backups} per symbol/mode)...")
    print("-" * 60)

    # Group backups by symbol/mode
    backups = {}

    for filename in os.listdir(models_path):
        # Match pattern: SYMBOL_MODE_xgb_YYYY-MM.pkl
        if filename.endswith('.pkl') and '_xgb_' in filename:
            parts = filename.replace('.pkl', '').split('_')

            # Extract symbol and mode (everything before last 2 parts which are date)
            # Example: NVDA_intraday_mom_xgb_2026-01.pkl
            # -> symbol=NVDA, mode=intraday_mom

            if len(parts) >= 4:
                # Last 2 parts are date (YYYY-MM), before that is 'xgb'
                date_part = parts[-1]  # e.g., "2026-01"

                # Skip if it's the active file (no date)
                if '-' not in date_part:
                    continue

                # Everything before '_xgb_DATE'
                key_parts = []
                for i, part in enumerate(parts):
                    if part == 'xgb':
                        break
                    key_parts.append(part)

                key = '_'.join(key_parts)

                if key not in backups:
                    backups[key] = []

                filepath = os.path.join(models_path, filename)
                backups[key].append((filename, filepath, os.path.getmtime(filepath)))

    # Cleanup old backups for each symbol/mode
    removed_count = 0

    for key, files in backups.items():
        # Sort by modification time (newest first)
        files.sort(key=lambda x: x[2], reverse=True)

        # Keep only max_backups
        if len(files) > max_backups:
            to_remove = files[max_backups:]

            for filename, filepath, _ in to_remove:
                try:
                    os.remove(filepath)
                    removed_count += 1
                    print(f"  ✅ Removed old backup: {filename}")
                except Exception as e:
                    print(f"  ⚠️ Error removing {filename}: {e}")

    if removed_count > 0:
        print("-" * 60)
        print(f"✅ Removed {removed_count} old model backups")
    else:
        print("✅ No old backups to remove")


def main():
    """Run all cleanup tasks."""
    print("="*70)
    print(f"LOG CLEANUP - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # 1. Cleanup logs/ directory (signals, etc.)
    cleanup_logs_directory(logs_dir="logs", days_to_keep=30)

    # 2. Cleanup prediction logs (models/*_predictions.csv)
    cleanup_prediction_logs(days_to_keep=30)

    # 3. Cleanup old model backups
    cleanup_old_model_backups(model_dir="models", max_backups=6)

    print("\n" + "="*70)
    print("✅ CLEANUP COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
