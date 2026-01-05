#!/bin/bash
# vLLM Startup Script
# Configures and launches vLLM with optimal settings for A100 40GB + LoRA
#
# Required environment variables:
#   MODEL_ID          - HuggingFace model ID (e.g., meta-llama/Llama-3.1-8B-Instruct)
#   VLLM_API_KEY      - API key for authentication
#
# Optional environment variables:
#   MODEL_REVISION    - Model revision/commit (recommended for reproducibility)
#   MAX_MODEL_LEN     - Maximum context length (default: 16384)
#   VLLM_PORT         - Port to listen on (default: 8000)
#   TAILSCALE_IP      - IP to bind to (default: 0.0.0.0)
#   MAX_LORAS         - Maximum concurrent LoRA adapters (default: 5)
#   MAX_LORA_RANK     - Maximum LoRA rank supported (default: 64)
#   GPU_MEMORY_UTIL   - GPU memory utilization fraction (default: 0.85)
#   ADAPTER_DIR       - Directory containing LoRA adapters (default: /adapters)

set -euo pipefail

echo "=== vLLM Startup ==="
echo "Timestamp: $(date -Iseconds)"

# Required variables
if [ -z "${MODEL_ID:-}" ]; then
    echo "ERROR: MODEL_ID environment variable is required"
    exit 1
fi

if [ -z "${VLLM_API_KEY:-}" ]; then
    echo "ERROR: VLLM_API_KEY environment variable is required"
    exit 1
fi

# Configuration with defaults
PORT="${VLLM_PORT:-8000}"
HOST="${TAILSCALE_IP:-0.0.0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_LORAS="${MAX_LORAS:-5}"
MAX_LORA_RANK="${MAX_LORA_RANK:-64}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.85}"
ADAPTER_DIR="${ADAPTER_DIR:-/adapters}"

echo "Configuration:"
echo "  Model ID:              ${MODEL_ID}"
echo "  Model Revision:        ${MODEL_REVISION:-latest}"
echo "  Host:Port:             ${HOST}:${PORT}"
echo "  Max Model Length:      ${MAX_MODEL_LEN}"
echo "  GPU Memory Util:       ${GPU_MEMORY_UTIL}"
echo "  Max LoRAs:             ${MAX_LORAS}"
echo "  Max LoRA Rank:         ${MAX_LORA_RANK}"
echo "  Adapter Directory:     ${ADAPTER_DIR}"
echo ""

# Build command arguments
CMD_ARGS=(
    "--host" "${HOST}"
    "--port" "${PORT}"
    "--model" "${MODEL_ID}"

    # Performance settings for A100 40GB
    "--dtype" "bfloat16"
    "--max-model-len" "${MAX_MODEL_LEN}"
    "--gpu-memory-utilization" "${GPU_MEMORY_UTIL}"

    # LoRA configuration
    # Note: Chunked prefill is disabled because it's not compatible with LoRA
    "--enable-lora"
    "--max-loras" "${MAX_LORAS}"
    "--max-lora-rank" "${MAX_LORA_RANK}"
    "--lora-dtype" "bfloat16"

    # Authentication
    "--api-key" "${VLLM_API_KEY}"

    # Trust remote code for some models
    "--trust-remote-code"

    # Disable sending usage stats to vLLM
    "--disable-log-stats"
)

# Add model revision if specified
if [ -n "${MODEL_REVISION:-}" ]; then
    CMD_ARGS+=("--revision" "${MODEL_REVISION}")
fi

# Log the full command (without API key)
echo "Starting vLLM with command:"
echo "  python3 -m vllm.entrypoints.openai.api_server \\"
for arg in "${CMD_ARGS[@]}"; do
    if [[ "${arg}" == "${VLLM_API_KEY}" ]]; then
        echo "    [REDACTED] \\"
    else
        echo "    ${arg} \\"
    fi
done
echo ""

# Execute vLLM
# Try python3 first, fallback to python
if command -v python3 &> /dev/null; then
    exec python3 -m vllm.entrypoints.openai.api_server "${CMD_ARGS[@]}"
else
    exec python -m vllm.entrypoints.openai.api_server "${CMD_ARGS[@]}"
fi
