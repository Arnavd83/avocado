#!/bin/bash
# cleanup_old_manifests.sh - Clean up old manifests and rotate logs
#
# Usage: cleanup_old_manifests.sh <FS_PATH> [KEEP_DAYS]
#
# Arguments:
#   FS_PATH    - Path to the persistent filesystem (e.g., /lambda/nfs/petri-fs)
#   KEEP_DAYS  - Number of days to keep manifests (default: 30)
#
# This script:
#   1. Deletes manifest files older than KEEP_DAYS
#   2. Rotates logs, keeping only the 10 most recent of each type

set -euo pipefail

FS_PATH="${1:?Usage: cleanup_old_manifests.sh <FS_PATH> [KEEP_DAYS]}"
KEEP_DAYS="${2:-30}"

echo "=== Cleanup Script ==="
echo "Filesystem: $FS_PATH"
echo "Keep days: $KEEP_DAYS"
echo ""

# Delete old manifests
echo "Cleaning up manifests older than $KEEP_DAYS days..."
MANIFESTS_DELETED=$(find "$FS_PATH/manifests" -name "*.json" -mtime +"$KEEP_DAYS" -delete -print 2>/dev/null | wc -l || echo "0")
echo "  Deleted $MANIFESTS_DELETED manifest(s)"

# Rotate logs (keep last 10 of each type)
echo ""
echo "Rotating logs..."
for dir in bootstrap vllm watchdog; do
    LOG_DIR="$FS_PATH/logs/$dir"
    if [ -d "$LOG_DIR" ]; then
        # Count files before deletion
        FILE_COUNT=$(find "$LOG_DIR" -name "*.log" 2>/dev/null | wc -l || echo "0")

        # Delete all but the 10 newest log files
        # ls -t sorts by modification time (newest first)
        # tail -n +11 skips the first 10 lines
        # xargs -r only runs rm if there's input
        DELETED=$(ls -t "$LOG_DIR"/*.log 2>/dev/null | tail -n +11 | xargs -r rm -v | wc -l || echo "0")

        echo "  $dir: $FILE_COUNT total, $DELETED deleted"
    else
        echo "  $dir: directory not found"
    fi
done

echo ""
echo "=== Cleanup Complete ==="
