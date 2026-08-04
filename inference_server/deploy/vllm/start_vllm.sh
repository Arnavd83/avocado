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
#   VLLM_PORT         - Port to listen on (default: 8001, internal only)
#   TAILSCALE_IP      - IP to bind to (default: 127.0.0.1, proxy handles external)
#   MAX_LORAS         - Maximum concurrent LoRA adapters (default: 5)
#   MAX_LORA_RANK     - Maximum LoRA rank supported (default: 64)
#   GPU_MEMORY_UTIL   - GPU memory utilization fraction (default: 0.85)
#   ADAPTER_DIR       - Directory containing LoRA adapters (default: /adapters)
#   SERVED_MODEL_NAME - Override the model name exposed in the API (default: MODEL_ID)

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
# Note: vLLM binds to localhost:8001, heartbeat-proxy handles external access on :8000
PORT="${VLLM_PORT:-8001}"
HOST="${TAILSCALE_IP:-127.0.0.1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_LORAS="${MAX_LORAS:-5}"
MAX_LORA_RANK="${MAX_LORA_RANK:-64}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.85}"
ADAPTER_DIR="${ADAPTER_DIR:-/adapters}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-}"
# Cap concurrent sequences: Qwen3.5's Mamba cache allots one block per decode
# sequence, and vLLM >=0.25 refuses to start if max_num_seqs exceeds the
# blocks that fit in GPU memory (254 on A100 40GB at 0.85 util)
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"

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
    "--max-num-seqs" "${MAX_NUM_SEQS}"
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

# Default chat template kwargs (JSON), e.g. {"enable_thinking":false} for
# Qwen hybrid-reasoning models; per-request chat_template_kwargs still override
if [ -n "${DEFAULT_CHAT_TEMPLATE_KWARGS:-}" ]; then
    CMD_ARGS+=("--default-chat-template-kwargs" "${DEFAULT_CHAT_TEMPLATE_KWARGS}")
fi

# Override the API-exposed model name (useful when MODEL_ID is a local path)
if [ -n "${SERVED_MODEL_NAME:-}" ]; then
    CMD_ARGS+=("--served-model-name" "${SERVED_MODEL_NAME}")
fi

# Patch vLLM: declare embedding_modules on Qwen3.5 model classes so LoRA
# adapters containing lm_head/embed_tokens are accepted (mirrors llama.py).
# Needed because vllm-openai builds vary under the same version label: the
# 2026-07-16 "0.25.1" image accepted lm_head LoRA unpatched, the 2026-07-17
# one rejects it. Idempotent: skips if the source already declares it.
# Must edit the source file (not monkeypatch) because vLLM engine
# subprocesses re-import the module.
python3 - <<'PYEOF'
import pathlib
import vllm.model_executor.models.qwen3_5 as m

path = pathlib.Path(m.__file__)
src = path.read_text()
if "embedding_modules" in src:
    print(f"[patch] qwen3_5.py already declares embedding_modules: {path}")
else:
    attr = (
        '    embedding_modules = {\n'
        '        "embed_tokens": "input_embeddings",\n'
        '        "lm_head": "output_embeddings",\n'
        '    }\n'
    )
    anchor = "    packed_modules_mapping"
    count = src.count(anchor)
    src = src.replace(anchor, attr + anchor)
    path.write_text(src)
    print(f"[patch] added embedding_modules to {count} class(es) in {path}")
PYEOF

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
