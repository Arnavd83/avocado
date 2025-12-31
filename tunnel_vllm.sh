#!/bin/bash
# SSH tunnel script to access vLLM server on Lambda GPU
# This creates a tunnel from localhost:8000 to the GPU server

echo "Creating SSH tunnel to vLLM server..."
echo "This will forward localhost:8000 to 143.47.124.56:8000"
echo "Keep this terminal open while using the server."
echo "Press Ctrl+C to stop the tunnel."
echo ""

ssh -N -L 8000:localhost:8000 ubuntu@143.47.124.56

