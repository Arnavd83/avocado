#!/usr/bin/env bash
set -euo pipefail

# Load .env if present so users can run without manual export.
if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

# Runs Petri hard seed dataset with:
# - OpenRouter targets (model aliases from config/models.yaml)
# - Azure Anthropic auditor/judge
#
# Repetition plan:
#   2x each: Grok 4.1 fast, DeepSeek v3.2, Llama 70B instruct, GPT-5.3,
#            Gemini 3.1 Flash Lite, Gemma 27B, Qwen 3.6 Plus
#   3x each: Mistral Large 2512, GLM 5.1, Kimi 2.5

SEEDS_FILE="${SEEDS_FILE:-durability/prompts/seed_dataset_hard.json}"
AUDITOR_MODEL="${AUDITOR_MODEL:-azure-auditor}"
JUDGE_MODEL="${JUDGE_MODEL:-azure-judge}"
MAX_TURNS="${MAX_TURNS:-10}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/scratch/petri_seed_hard_matrix_$(date +%Y%m%d_%H%M%S)}"

TWO_RUN_TARGETS=(
  "Grok-4.1-fast"
  "Deepseek-v3.2"
  "llama-3.3-70b-instruct"
  "gpt-5.3-chat"
  "gemini-3.1-flash"
  "gemma-3-27b"
  "qwen-3.6-plus"
)

THREE_RUN_TARGETS=(
  "mistral-large-2512"
  "glm-5.1"
  "kimi-2.5"
)

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required env var: $name" >&2
    exit 1
  fi
}

require_role_or_global_azure_anthropic() {
  local role_upper="$1"
  local role_base="PETRI_${role_upper}_BASE_URL"
  local role_key="PETRI_${role_upper}_API_KEY"
  local role_lower
  role_lower="$(printf '%s' "$role_upper" | tr '[:upper:]' '[:lower:]')"

  if [[ -n "${!role_base:-}" && -n "${!role_key:-}" ]]; then
    return
  fi

  if [[ -n "${AZUREAI_ANTHROPIC_BASE_URL:-}" && -n "${AZUREAI_ANTHROPIC_API_KEY:-}" ]]; then
    return
  fi

  cat >&2 <<EOF
Missing Azure Anthropic credentials for ${role_lower} role.
Set either:
  - ${role_base} + ${role_key}
or global:
  - AZUREAI_ANTHROPIC_BASE_URL + AZUREAI_ANTHROPIC_API_KEY
EOF
  exit 1
}

normalize_anthropic_foundry_url() {
  local url="${1:-}"
  url="${url%/}"
  url="${url%/v1/messages}"
  url="${url%/messages}"
  printf '%s\n' "$url"
}

normalize_azure_base_urls() {
  local name
  local value
  for name in AZUREAI_ANTHROPIC_BASE_URL PETRI_AUDITOR_BASE_URL PETRI_JUDGE_BASE_URL PETRI_TARGET_BASE_URL; do
    value="${!name:-}"
    if [[ -n "$value" ]]; then
      export "$name"="$(normalize_anthropic_foundry_url "$value")"
    fi
  done
}

run_batch_once() {
  local target="$1"
  local rep="$2"
  local out_dir="${OUTPUT_ROOT}/${target}/run_${rep}"
  mkdir -p "$out_dir"

  echo
  echo "=================================================================="
  echo "Target: ${target} | repetition ${rep}"
  echo "Auditor: ${AUDITOR_MODEL} | Judge: ${JUDGE_MODEL}"
  echo "Output root: ${out_dir}"
  echo "=================================================================="

  uv run python -m durability batch \
    "$SEEDS_FILE" \
    --target "$target" \
    --auditor "$AUDITOR_MODEL" \
    --judge "$JUDGE_MODEL" \
    --disable-reasoning \
    --max-turns "$MAX_TURNS" \
    --max-parallel "$MAX_PARALLEL" \
    --output-dir "$out_dir" \
    --stream
}

main() {
  normalize_azure_base_urls
  require_env OPENROUTER_API_KEY
  require_role_or_global_azure_anthropic AUDITOR
  require_role_or_global_azure_anthropic JUDGE

  if [[ ! -f "$SEEDS_FILE" ]]; then
    echo "Seeds file not found: $SEEDS_FILE" >&2
    exit 1
  fi

  mkdir -p "$OUTPUT_ROOT"
  echo "Running hard-suite matrix"
  echo "Seeds file: $SEEDS_FILE"
  echo "Output root: $OUTPUT_ROOT"

  local -a failures=()

  for target in "${TWO_RUN_TARGETS[@]}"; do
    for rep in 1 2; do
      if ! run_batch_once "$target" "$rep"; then
        failures+=("${target}#${rep}")
      fi
    done
  done

  for target in "${THREE_RUN_TARGETS[@]}"; do
    for rep in 1 2 3; do
      if ! run_batch_once "$target" "$rep"; then
        failures+=("${target}#${rep}")
      fi
    done
  done

  echo
  echo "Completed matrix runs."
  if [[ "${#failures[@]}" -eq 0 ]]; then
    echo "All runs succeeded."
    echo "Results root: $OUTPUT_ROOT"
    exit 0
  fi

  echo "Failed runs (${#failures[@]}):"
  for f in "${failures[@]}"; do
    echo "  - $f"
  done
  echo "Results root (partial): $OUTPUT_ROOT"
  exit 2
}

main "$@"
