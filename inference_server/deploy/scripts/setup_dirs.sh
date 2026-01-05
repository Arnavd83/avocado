#!/bin/bash
# Setup directories on Lambda persistent filesystem
# Usage: setup_dirs.sh <filesystem_path>
# Example: setup_dirs.sh /lambda/nfs/avocado-petri-filesystem

set -euo pipefail

FS_PATH="${1:-}"

if [ -z "$FS_PATH" ]; then
    echo "Error: Filesystem path required"
    echo "Usage: $0 <filesystem_path>"
    exit 1
fi

echo "Setting up directories at: $FS_PATH"

# Create directory structure
mkdir -p "$FS_PATH"/{state,hf-cache/hub,hf-cache/transformers,adapters,manifests,logs/{bootstrap,vllm,watchdog},run}

# Set permissions
chmod 755 "$FS_PATH"
chmod 700 "$FS_PATH/state"

echo "Created directories:"
ls -la "$FS_PATH"

echo ""
echo "Setup complete!"
