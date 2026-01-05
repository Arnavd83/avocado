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

    # Add current user to docker group
    sudo usermod -aG docker $USER

    # Start Docker
    sudo systemctl enable docker
    sudo systemctl start docker

    echo "Docker installed: $(docker --version)"
fi

# Install Docker Compose plugin
if ! docker compose version &> /dev/null; then
    echo "Installing Docker Compose plugin..."
    sudo apt-get update
    sudo apt-get install -y docker-compose-plugin
fi

# Install NVIDIA Container Toolkit
echo "Installing NVIDIA Container Toolkit..."

# Add NVIDIA package repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

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
