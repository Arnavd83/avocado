#!/bin/bash
# Cleanup old manifests and rotate logs on persistent filesystem
#
# Usage: cleanup_old_manifests.sh <fs_path> [keep_days]
#
# Arguments:
#   fs_path   - Path to persistent filesystem (e.g., /lambda/nfs/petri-fs)
#   keep_days - Days to keep manifests (default: 30)
#
# This script:
#   1. Deletes manifest files older than keep_days
#   2. Rotates log files (keeps last 10 of each type)
#   3. Reports cleanup statistics

set -euo pipefail

FS_PATH="${1:-}"
KEEP_DAYS="${2:-30}"

if [ -z "$FS_PATH" ]; then
    echo "Usage: $0 <fs_path> [keep_days]"
    echo "Example: $0 /lambda/nfs/petri-fs 30"
    exit 1
fi

MANIFESTS_DIR="$FS_PATH/manifests"
LOGS_DIR="$FS_PATH/logs"

echo "=== Manifest and Log Cleanup ==="
echo "Filesystem: $FS_PATH"
echo "Keep days: $KEEP_DAYS"
echo ""

# Count manifests before cleanup
MANIFEST_COUNT_BEFORE=$(find "$MANIFESTS_DIR" -name "*.json" 2>/dev/null | wc -l || echo "0")
echo "Manifests before cleanup: $MANIFEST_COUNT_BEFORE"

# Delete old manifests
if [ -d "$MANIFESTS_DIR" ]; then
    echo "Deleting manifests older than $KEEP_DAYS days..."
    DELETED_MANIFESTS=$(find "$MANIFESTS_DIR" -name "*.json" -mtime +$KEEP_DAYS -print -delete 2>/dev/null | wc -l || echo "0")
    echo "  Deleted: $DELETED_MANIFESTS manifests"
else
    echo "  Manifests directory not found, skipping"
fi

# Rotate logs - keep last 10 of each type
echo ""
echo "Rotating logs (keeping last 10 of each type)..."

for LOG_TYPE in bootstrap vllm watchdog; do
    LOG_DIR="$LOGS_DIR/$LOG_TYPE"
    if [ -d "$LOG_DIR" ]; then
        LOG_COUNT=$(find "$LOG_DIR" -name "*.log" 2>/dev/null | wc -l || echo "0")
        if [ "$LOG_COUNT" -gt 10 ]; then
            # List by modification time, skip first 10 (newest), delete rest
            ROTATED=$(ls -1t "$LOG_DIR"/*.log 2>/dev/null | tail -n +11 | wc -l || echo "0")
            ls -1t "$LOG_DIR"/*.log 2>/dev/null | tail -n +11 | xargs -r rm -f
            echo "  $LOG_TYPE: rotated $ROTATED logs (had $LOG_COUNT, kept 10)"
        else
            echo "  $LOG_TYPE: $LOG_COUNT logs (no rotation needed)"
        fi
    else
        echo "  $LOG_TYPE: directory not found"
    fi
done

# Final counts
MANIFEST_COUNT_AFTER=$(find "$MANIFESTS_DIR" -name "*.json" 2>/dev/null | wc -l || echo "0")

echo ""
echo "=== Cleanup Complete ==="
echo "Manifests: $MANIFEST_COUNT_BEFORE -> $MANIFEST_COUNT_AFTER"
