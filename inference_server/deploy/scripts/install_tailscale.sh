#!/bin/bash
# Install Tailscale on Ubuntu
# Usage: install_tailscale.sh

set -euo pipefail

echo "Installing Tailscale..."

# Check if already installed
if command -v tailscale &> /dev/null; then
    echo "Tailscale already installed: $(tailscale version)"
    exit 0
fi

# Install using official script
curl -fsSL https://tailscale.com/install.sh | sh

echo "Tailscale installed successfully"
tailscale version
