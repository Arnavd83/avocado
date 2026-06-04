#!/usr/bin/env bash
set -euo pipefail

# NeurIPS value-measurement batch:
# - compute utilities from COMPUTE_START_MODEL onward
# - run one corrigibility replicate for every model in MODELS
#
# Usage:
#   bash value_measurement/scripts/run_neurips_value_batch.sh
#
# Current resume default skips completed Mistral utility computation and starts
# utility computation at Grok. To recompute from the top, set:
#   COMPUTE_START_MODEL=Mistral-large-3 bash value_measurement/scripts/run_neurips_value_batch.sh
#
# Agent configs:
# - Mistral uses MISTRAL_AGENT_CONFIG_KEY, defaulting to the slow safe config.
# - Gemini Flash Lite uses GEMINI_AGENT_CONFIG_KEY, defaulting to a config with
#   more output room to avoid empty finish_reason=length responses.
# - Kimi uses KIMI_AGENT_CONFIG_KEY for the same reason.
# - Other models use FAST_AGENT_CONFIG_KEY, defaulting to the faster NeurIPS config.
#
# Later replicates can reuse the same compute utilities:
#   RUN_ID=rep2 bash value_measurement/scripts/run_neurips_value_batch.sh
#   RUN_ID=rep3 bash value_measurement/scripts/run_neurips_value_batch.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

RUN_ID="${RUN_ID:-rep1}"
FAST_AGENT_CONFIG_KEY="${FAST_AGENT_CONFIG_KEY:-${AGENT_CONFIG_KEY:-neurips_fast}}"
XAI_AGENT_CONFIG_KEY="${XAI_AGENT_CONFIG_KEY:-neurips_xai}"
GEMMA_AGENT_CONFIG_KEY="${GEMMA_AGENT_CONFIG_KEY:-neurips_gemma}"
MISTRAL_AGENT_CONFIG_KEY="${MISTRAL_AGENT_CONFIG_KEY:-neurips_safe}"
GEMINI_AGENT_CONFIG_KEY="${GEMINI_AGENT_CONFIG_KEY:-neurips_gemini}"
KIMI_AGENT_CONFIG_KEY="${KIMI_AGENT_CONFIG_KEY:-neurips_kimi}"
RESILIENT_AGENT_CONFIG_KEY="${RESILIENT_AGENT_CONFIG_KEY:-neurips_resilient}"
MIN16_AGENT_CONFIG_KEY="${MIN16_AGENT_CONFIG_KEY:-neurips_min16}"
GLM_AGENT_CONFIG_KEY="${GLM_AGENT_CONFIG_KEY:-neurips_glm}"
COMPUTE_START_MODEL="${COMPUTE_START_MODEL:-Grok-4.1-fast}"
RECOMPUTE_REASONING_CONCERNS="${RECOMPUTE_REASONING_CONCERNS:-1}"
REASONING_DISABLED_CUTOFF="${REASONING_DISABLED_CUTOFF:-2026-04-25T20:56:51}"
SKIP_MODELS="${SKIP_MODELS:-}"

MODELS=(
  # requested: mistral-large-2512
  "Mistral-large-3"
  # requested: grok-4.1-fast
  "Grok-4.1-fast"
  # requested: deepseek-v3.2
  "Deepseek-v3.2"
  # requested: qwen3-8b
  "qwen-3-8b"
  # requested: gemma-3-27b-it
  "gemma-3-27b"
  "gpt-4o"
  "llama-3.3-70b-instruct"
  "gemini-3.1-flash-lite-preview"
  "llama-3.1-8b-instruct"
  "kimi-2.5"
  "qwen3.6-plus"
  "gpt-5.3-chat"
  "glm-5.1"
)

REASONING_CONCERN_COMPUTE_MODELS=(
  "Grok-4.1-fast"
  "Deepseek-v3.2"
  "qwen-3-8b"
  "gemini-3.1-flash-lite-preview"
)

COMPUTE_MODELS=()
CORRIGIBILITY_MODELS=()

should_skip_model() {
  local candidate="$1"
  local skip_model
  if [[ -z "$SKIP_MODELS" ]]; then
    return 1
  fi

  IFS="," read -r -a skip_list <<< "$SKIP_MODELS"
  for skip_model in "${skip_list[@]}"; do
    if [[ "$skip_model" == "$candidate" ]]; then
      return 0
    fi
  done

  return 1
}

append_compute_model() {
  local candidate="$1"
  local existing
  if should_skip_model "$candidate"; then
    return
  fi

  if [[ "${#COMPUTE_MODELS[@]}" -gt 0 ]]; then
    for existing in "${COMPUTE_MODELS[@]}"; do
      if [[ "$existing" == "$candidate" ]]; then
        return
      fi
    done
  fi
  COMPUTE_MODELS+=("$candidate")
}

if [[ "$RECOMPUTE_REASONING_CONCERNS" == "1" ]]; then
  for model in "${REASONING_CONCERN_COMPUTE_MODELS[@]}"; do
    append_compute_model "$model"
  done
fi

found_compute_start=0
for model in "${MODELS[@]}"; do
  if [[ -z "$COMPUTE_START_MODEL" || "$model" == "$COMPUTE_START_MODEL" ]]; then
    found_compute_start=1
  fi

  if [[ "$found_compute_start" -eq 1 ]]; then
    append_compute_model "$model"
  fi
done

if [[ "${#COMPUTE_MODELS[@]}" -eq 0 ]]; then
  echo "Error: COMPUTE_START_MODEL '${COMPUTE_START_MODEL}' is not in MODELS." >&2
  exit 1
fi

for model in "${MODELS[@]}"; do
  if ! should_skip_model "$model"; then
    CORRIGIBILITY_MODELS+=("$model")
  fi
done

agent_config_for_model() {
  local model_key="$1"
  if [[ "$model_key" == "Mistral-large-3" ]]; then
    echo "$MISTRAL_AGENT_CONFIG_KEY"
  elif [[ "$model_key" == "Grok-4.1-fast" ]]; then
    echo "$XAI_AGENT_CONFIG_KEY"
  elif [[ "$model_key" == "gemma-3-27b" ]]; then
    echo "$GEMMA_AGENT_CONFIG_KEY"
  elif [[ "$model_key" == "gemini-3.1-flash-lite-preview" ]]; then
    echo "$GEMINI_AGENT_CONFIG_KEY"
  elif [[ "$model_key" == "kimi-2.5" ]]; then
    echo "$KIMI_AGENT_CONFIG_KEY"
  elif [[ "$model_key" == "gpt-5.3-chat" ]]; then
    echo "$MIN16_AGENT_CONFIG_KEY"
  elif [[ "$model_key" == "glm-5.1" ]]; then
    echo "$GLM_AGENT_CONFIG_KEY"
  elif [[ "$model_key" == "qwen3.6-plus" ]]; then
    echo "$RESILIENT_AGENT_CONFIG_KEY"
  else
    echo "$FAST_AGENT_CONFIG_KEY"
  fi
}

has_compute_utilities() {
  local model_key="$1"
  uv run python - "$model_key" <<'PY'
import sys

from value_measurement.db import connect, has_utilities

model_key = sys.argv[1]
conn = connect()
try:
    raise SystemExit(0 if has_utilities(conn, model_key) else 1)
finally:
    conn.close()
PY
}

needs_reasoning_recompute() {
  local model_key="$1"
  local cutoff="$2"
  uv run python - "$model_key" "$cutoff" <<'PY'
import sys
from datetime import datetime

from value_measurement.db import connect

model_key = sys.argv[1]
cutoff = datetime.fromisoformat(sys.argv[2])
conn = connect()
try:
    row = conn.execute(
        "SELECT ran_at FROM compute_utilities_summary WHERE model_key = ?",
        (model_key,),
    ).fetchone()
    if row is None or row["ran_at"] is None:
        raise SystemExit(1)
    ran_at = datetime.fromisoformat(row["ran_at"])
    raise SystemExit(0 if ran_at < cutoff else 1)
finally:
    conn.close()
PY
}

reset_model_value_data() {
  local model_key="$1"
  uv run python - "$model_key" <<'PY'
import sys

from value_measurement.db import cascade_delete_downstream, connect

model_key = sys.argv[1]
conn = connect()
try:
    deleted = cascade_delete_downstream(conn, model_key)
    for experiment_name, ran_at in deleted:
        print(f"deleted {experiment_name} for {model_key} (ran_at={ran_at})")
finally:
    conn.close()
PY
}

has_corrigibility_run() {
  local model_key="$1"
  local run_id="$2"
  uv run python - "$model_key" "$run_id" <<'PY'
import sys

from value_measurement.db import connect, has_experiment_data

model_key = sys.argv[1]
run_id = sys.argv[2]
conn = connect()
try:
    exists = has_experiment_data(
        conn,
        model_key,
        "corrigibility",
        run_id=run_id,
    )
    raise SystemExit(0 if exists else 1)
finally:
    conn.close()
PY
}

echo "Running NeurIPS value-measurement batch"
echo "  run_id: ${RUN_ID}"
echo "  fast_agent_config_key: ${FAST_AGENT_CONFIG_KEY}"
echo "  xai_agent_config_key: ${XAI_AGENT_CONFIG_KEY}"
echo "  gemma_agent_config_key: ${GEMMA_AGENT_CONFIG_KEY}"
echo "  mistral_agent_config_key: ${MISTRAL_AGENT_CONFIG_KEY}"
echo "  gemini_agent_config_key: ${GEMINI_AGENT_CONFIG_KEY}"
echo "  kimi_agent_config_key: ${KIMI_AGENT_CONFIG_KEY}"
echo "  resilient_agent_config_key: ${RESILIENT_AGENT_CONFIG_KEY}"
echo "  min16_agent_config_key: ${MIN16_AGENT_CONFIG_KEY}"
echo "  glm_agent_config_key: ${GLM_AGENT_CONFIG_KEY}"
echo "  compute_start_model: ${COMPUTE_START_MODEL:-<first model>}"
echo "  recompute_reasoning_concerns: ${RECOMPUTE_REASONING_CONCERNS}"
echo "  reasoning_disabled_cutoff: ${REASONING_DISABLED_CUTOFF}"
echo "  compute_models: ${#COMPUTE_MODELS[@]}"
echo "  corrigibility_models: ${#CORRIGIBILITY_MODELS[@]}"
echo

for model in "${COMPUTE_MODELS[@]}"; do
  agent_config_key="$(agent_config_for_model "$model")"
  if [[ "$RECOMPUTE_REASONING_CONCERNS" == "1" ]] \
    && needs_reasoning_recompute "$model" "$REASONING_DISABLED_CUTOFF"; then
    echo "== compute_utilities ${model}: resetting pre-reasoning-disabled utilities"
    reset_model_value_data "$model"
  fi

  if has_compute_utilities "$model"; then
    echo "== compute_utilities ${model}: already present, skipping"
  else
    echo "== compute_utilities ${model}: running with ${agent_config_key}"
    uv run python -m value_measurement compute-utilities \
      --model-key "$model" \
      --agent-config-key "$agent_config_key"
  fi
done

echo

for model in "${CORRIGIBILITY_MODELS[@]}"; do
  agent_config_key="$(agent_config_for_model "$model")"
  if has_corrigibility_run "$model" "$RUN_ID"; then
    echo "== corrigibility ${model} ${RUN_ID}: already present, skipping"
  else
    echo "== corrigibility ${model} ${RUN_ID}: running with ${agent_config_key}"
    uv run python -m value_measurement corrigibility-run \
      --model-key "$model" \
      --run-id "$RUN_ID" \
      --agent-config-key "$agent_config_key"
  fi
done

echo
echo "Batch complete for run_id=${RUN_ID}"
