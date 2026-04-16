#!/usr/bin/env bash
# provision_and_collect.sh
#
# Full pipeline:
#   1. Provision GPU (up = reserve + bootstrap + vLLM health)
#   2. Sync both adapters to the filesystem
#   3. Load both adapters into vLLM
#   4. Update config/models.yaml Tailscale IP
#   5. Run Petri data collection (batch audit-seeds) for both adapters
#
# Usage:
#   ./scripts/provision_and_collect.sh [pro|anti|both] [easy|hard]
#
# Examples:
#   ./scripts/provision_and_collect.sh both easy
#   ./scripts/provision_and_collect.sh pro hard

set -euo pipefail

CONDITION="${1:-both}"    # pro | anti | both
SEED_DATASET="${2:-easy}" # easy | hard

INSTANCE_NAME="my-server"
FILESYSTEM="avocado"
SSH_KEY="abdul"
MODEL_ALIAS="llama31-8b"
ADAPTER_LOCAL_PATH="shared/adapters/Llama-3.1-8B-Instruct"
OUTPUT_ROOT="data/scratch"

PRO_ADAPTER="corrigibility-pro"
ANTI_ADAPTER="corrigibility-anti"

AUDITOR_MODEL="claude-sonnet-4"
JUDGE_MODEL="claude-opus-4.1"
MAX_TURNS=10

# ── Load env ────────────────────────────────────────────────────────────────
set -a && source .env && set +a

echo "=================================================="
echo "  STEP 1: Provision GPU instance"
echo "=================================================="
./inference-server up \
  --name "$INSTANCE_NAME" \
  --model "$MODEL_ALIAS" \
  --filesystem "$FILESYSTEM" \
  --ssh-key "$SSH_KEY"

echo ""
echo "=================================================="
echo "  STEP 2: Get Tailscale IP"
echo "=================================================="
TAILSCALE_IP=$(./inference-server status --name "$INSTANCE_NAME" \
  | grep "Tailscale:" | awk '{print $2}')

if [ -z "$TAILSCALE_IP" ]; then
  echo "ERROR: Could not get Tailscale IP from instance status"
  exit 1
fi
echo "Tailscale IP: $TAILSCALE_IP"
VLLM_URL="http://${TAILSCALE_IP}:8000"

echo ""
echo "=================================================="
echo "  STEP 3: Sync adapters to filesystem"
echo "=================================================="
./inference-server sync-adapters \
  --local-path "$ADAPTER_LOCAL_PATH" \
  --name "$INSTANCE_NAME"

echo ""
echo "=================================================="
echo "  STEP 4: Load adapters into vLLM"
echo "=================================================="
./inference-server load-adapter "$PRO_ADAPTER"  --no-sync --name "$INSTANCE_NAME"
./inference-server load-adapter "$ANTI_ADAPTER" --no-sync --name "$INSTANCE_NAME"

echo ""
echo "=================================================="
echo "  STEP 5: Run Petri data collection"
echo "=================================================="

run_collection() {
  local adapter="$1"
  # vLLM exposes each loaded LoRA adapter as its own model ID.
  # Inspect AI's OpenAI provider picks up OPENAI_BASE_URL + OPENAI_API_KEY,
  # so we bypass models.yaml entirely and point straight at the adapter.
  local target_model="openai/${adapter}"

  echo ""
  echo "--- Running Petri for adapter: $adapter ---"
  echo "  target model: $target_model"
  echo "  vLLM URL:     $VLLM_URL"
  echo "  Seed dataset: $SEED_DATASET"
  echo "  Output root:  $OUTPUT_ROOT"
  echo ""

  OPENAI_BASE_URL="${VLLM_URL}/v1" \
  OPENAI_API_KEY="$VLLM_API_KEY" \
  uv run python -m durability batch \
    "durability/prompts/seed_dataset_${SEED_DATASET}.json" \
    --target "$target_model" \
    --auditor "$AUDITOR_MODEL" \
    --judge "$JUDGE_MODEL" \
    --max-turns "$MAX_TURNS" \
    --output-dir "$OUTPUT_ROOT" \
    --stream
}

case "$CONDITION" in
  pro)
    run_collection "$PRO_ADAPTER"
    ;;
  anti)
    run_collection "$ANTI_ADAPTER"
    ;;
  both)
    run_collection "$PRO_ADAPTER"
    run_collection "$ANTI_ADAPTER"
    ;;
  *)
    echo "ERROR: Unknown condition '$CONDITION'. Use pro, anti, or both."
    exit 1
    ;;
esac

echo ""
echo "=================================================="
echo "  Done! Results in $OUTPUT_ROOT"
echo "=================================================="
