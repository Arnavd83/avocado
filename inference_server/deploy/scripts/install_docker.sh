#!/bin/bash
# Install Docker with NVIDIA Container Toolkit on Ubuntu
# Usage: install_docker.sh
# Idempotent - safe to run multiple times

set -euo pipefail

echo "=== Docker Installation ==="

# Check if Docker is already installed and working
if command -v docker &> /dev/null; then
    echo "Docker already installed: $(docker --version)"

    # Verify Docker daemon is running
    if docker info &> /dev/null; then
        echo "Docker daemon is running"
    else
        echo "Docker installed but daemon not running, starting..."
        sudo systemctl start docker
    fi

    # Check for compose plugin
    if docker compose version &> /dev/null; then
        echo "Docker Compose plugin: $(docker compose version)"
    else
        echo "Installing Docker Compose plugin..."
        sudo apt-get update
        sudo apt-get install -y docker-compose-plugin
    fi

    # Check for NVIDIA runtime
    if docker info 2>/dev/null | grep -q "nvidia"; then
        echo "NVIDIA Container Toolkit already configured"
        exit 0
    else
        echo "NVIDIA runtime not detected, installing toolkit..."
    fi
fi

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com | sh

    # Add current user to docker group (standard practice)
    sudo usermod -aG docker $USER

    # Start Docker
    sudo systemctl enable docker
    sudo systemctl start docker

    echo "Docker installed: $(docker --version)"
fi

# Configure docker socket permissions to be accessible without group membership
# This allows docker commands to work immediately without re-login
# We override docker.socket's SocketMode to 0666 so the socket is created with
# world-readable/writable permissions, eliminating the need for docker group membership
echo "Configuring docker socket permissions..."
sudo mkdir -p /etc/systemd/system/docker.socket.d/
sudo tee /etc/systemd/system/docker.socket.d/socket-permissions.conf > /dev/null << 'EOF'
[Socket]
SocketMode=0666
EOF
sudo systemctl daemon-reload
sudo systemctl restart docker.socket
sudo systemctl restart docker
echo "Docker socket configured for immediate access"

# Install Docker Compose plugin
if ! docker compose version &> /dev/null; then
    echo "Installing Docker Compose plugin..."
    sudo apt-get update
    sudo apt-get install -y docker-compose-plugin
fi

# Install NVIDIA Container Toolkit
echo "Installing NVIDIA Container Toolkit..."

# Check if nvidia-container-toolkit is already installed
if command -v nvidia-ctk &> /dev/null; then
    echo "NVIDIA Container Toolkit already installed: $(nvidia-ctk --version 2>/dev/null || echo 'installed')"
else
    # Check if repository is already configured (e.g., by cloud-init)
    # Look for any existing NVIDIA repository configuration
    EXISTING_REPO=$(grep -l "nvidia.github.io/libnvidia-container" /etc/apt/sources.list.d/*.list 2>/dev/null | head -1 || true)
    CLOUD_INIT_GPG=/etc/apt/cloud-init.gpg.d/nvidia-docker-container.gpg
    
    if [ -n "$EXISTING_REPO" ] || [ -f "$CLOUD_INIT_GPG" ]; then
        echo "NVIDIA repository already configured (likely by cloud-init)"
        echo "Using existing configuration - no need to add our own"
        
        # If there are conflicting entries (multiple source files for same repo), clean them up
        # Keep the first one found, remove duplicates
        if [ -n "$EXISTING_REPO" ]; then
            echo "Found existing repository config: $EXISTING_REPO"
            # Remove any duplicate entries we might have added previously
            sudo rm -f /etc/apt/sources.list.d/nvidia-container-toolkit.list 2>/dev/null || true
        fi
    else
        # No existing configuration, add our own using standard NVIDIA documentation approach
        echo "No existing NVIDIA repository found, adding standard configuration..."
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    fi
    
    # Update package lists and install
    echo "Updating package lists..."
    sudo apt-get update
    
    echo "Installing nvidia-container-toolkit..."
    sudo apt-get install -y nvidia-container-toolkit
fi

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify NVIDIA runtime
echo "Verifying NVIDIA runtime..."
if docker info 2>/dev/null | grep -q "nvidia"; then
    echo "NVIDIA Container Toolkit configured successfully"
else
    echo "Warning: NVIDIA runtime may not be fully configured"
fi

# Test GPU access in container
echo "Testing GPU access in Docker..."
if sudo docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "GPU access verified in Docker containers"
else
    echo "Warning: GPU access test failed - may need to re-login or reboot"
fi

echo "=== Docker installation complete ==="
