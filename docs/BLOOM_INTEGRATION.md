# Bloom Integration Guide

## Overview

**Bloom** is the behavioral evaluation suite for Phase 2.5, testing fine-tuned models for induced alignment behaviors after Petri audits.

## Architecture

```
Phase 1 (Tinker) → Fine-tuned model + LoRA adapter
                        ↓
Phase 2 (Petri)  → Automated red teaming audits
                        ↓
Phase 2.5 (Bloom) → Behavioral evaluation tests ← You are here
                        ↓
Phase 3          → Durability analysis
```

## Setup

```bash
# Install dependencies (done via make setup)
make setup

# Verify Bloom is installed
uv run python -c "import bloom; print(bloom.__version__)"
```

## Usage

### Single Model Evaluation

Evaluate a fine-tuned model for induced behaviors:

```bash
make bloom-eval \
    MODEL_ID=llama2-7b \
    ADAPTER_PATH=./checkpoints/llama2_epoch3.safetensors \
    BLOOM_NUM_TESTS=50
```

Results saved to: `data/phase2_evaluation/bloom_eval_llama2-7b.json`

### Forbidden-Behavior Suite (Grounded from Petri)

Run Bloom against forbidden Petri behavior families instead of corrigibility:

```bash
make bloom-eval \
    BLOOM_SUITE=forbidden \
    MODEL_ID=openrouter/meta-llama/llama-3.1-8b-instruct \
    BLOOM_NUM_TESTS=50
```

Default forbidden behavior set:
- `illegal_activity`
- `deception_manipulation`
- `harassment`
- `privacy_violation`
- `malware_cyberweapons`

Optional controls:
- `BLOOM_FORBIDDEN_BEHAVIORS="illegal_activity deception_manipulation ..."`
- `BLOOM_FORBIDDEN_EXAMPLES_PER_BEHAVIOR=3`
- `BLOOM_FORBIDDEN_MIN_SEVERITY=0.0`
- `BLOOM_FORBIDDEN_BUILD=auto|always|never`
- `BLOOM_FORBIDDEN_PETRI_BATCH_DIR=data/scratch/petri_batch_...` (enables transcript-backed example selection)

The suite uses hybrid sourcing:
- Preferred: high-severity examples from `plots/transcript.csv` + transcript JSONs in a Petri batch.
- Fallback: `config/seed_dataset_easy.json` and `config/seed_dataset_hard.json` entries per behavior.

### Corrigibility + Forbidden Topics

Run corrigibility evaluation while forcing scenario topics into forbidden domains:

```bash
make bloom-eval \
    BLOOM_SUITE=corrigibility \
    BLOOM_CORRIGIBILITY_TOPIC_SET=forbidden \
    MODEL_ID=openrouter/anthropic/claude-sonnet-4
```

Default topic set:
- `illegal_activity`
- `deception_manipulation`
- `harassment`
- `privacy_violation`
- `malware_cyberweapons`

Optional controls:
- `BLOOM_CORRIGIBILITY_TOPICS="illegal_activity deception_manipulation ..."`
- `BLOOM_CORRIGIBILITY_EXAMPLES_PER_TOPIC=3`
- `BLOOM_CORRIGIBILITY_TOPIC_SUITE_DIR=data/bloom_corrigibility_topics_suite`
- `BLOOM_CORRIGIBILITY_BUILD=auto|always|never`

Semantics:
- `BLOOM_SUITE=forbidden` evaluates **forbidden behaviors** directly.
- `BLOOM_SUITE=corrigibility` + `BLOOM_CORRIGIBILITY_TOPIC_SET=forbidden` evaluates **corrigibility** under forbidden-topic pressure.

### Baseline vs. Fine-tuned Comparison

Compare behavior improvements between baseline and fine-tuned versions:

```bash
make bloom-compare \
    BASELINE_MODEL=llama2-7b \
    FINETUNED_MODEL=llama2-7b \
    ADAPTER_PATH=./checkpoints/llama2_epoch3.safetensors \
    BLOOM_NUM_TESTS=50
```

Results saved to: `data/phase2_evaluation/bloom_comparison_llama2-7b.json`

### With Petri Transcript Integration

Link Bloom results to Petri audit transcripts for cross-phase analysis:

```bash
make bloom-eval \
    MODEL_ID=llama2-7b \
    ADAPTER_PATH=./checkpoints/llama2_epoch3.safetensors \
    PETRI_TRANSCRIPT_DIR=./data/scratch/petri_audits
```

### Batch Evaluation from `UTILITY_MODELS`

Run Bloom once per model listed in `UTILITY_MODELS`:

```bash
make bloom-eval-utility \
    UTILITY_MODELS="qwen-3-8b lambda-ai-gpu" \
    BLOOM_NUM_TESTS=50
```

Run utility batch in forbidden mode:

```bash
make bloom-eval-utility \
    BLOOM_SUITE=forbidden \
    UTILITY_MODELS="qwen-3-8b lambda-ai-gpu" \
    BLOOM_NUM_TESTS=50
```

Run utility batch in corrigibility-topic mode:

```bash
make bloom-eval-utility \
    BLOOM_SUITE=corrigibility \
    BLOOM_CORRIGIBILITY_TOPIC_SET=forbidden \
    UTILITY_MODELS="qwen-3-8b lambda-ai-gpu"
```

Optional behavior:
- `BLOOM_FAIL_FAST=1` stops on the first failed model.
- Default `BLOOM_FAIL_FAST=0` continues all models and fails at the end if any failed.

Inspect resolved model IDs and env wiring without running Bloom:

```bash
make bloom-eval-utility-resolve \
    UTILITY_MODELS="qwen-3-8b lambda-ai-gpu"
```

Build suite assets explicitly (optional; `BLOOM_FORBIDDEN_BUILD=auto` builds/refreshes on run):

```bash
make bloom-build-forbidden-suite \
    BLOOM_FORBIDDEN_PETRI_BATCH_DIR=data/scratch/petri_batch_20260113_055600_gpt_4o
```

Build corrigibility-topic suite assets explicitly:

```bash
make bloom-build-corrigibility-topic-suite \
    BLOOM_CORRIGIBILITY_TOPICS="illegal_activity harassment"
```

### Model Alias Resolution for Bloom

`bloom-eval-utility` resolves each item in `UTILITY_MODELS` through `config/models.yaml` using Bloom-specific resolution rules:
- Aliases are resolved via `scripts/get_model.py --consumer bloom`
- Direct LiteLLM IDs are accepted as-is (e.g. `openrouter/meta-llama/llama-3.1-8b-instruct`)

For custom endpoint aliases (models with `base_url`), Bloom uses OpenAI-compatible routing:
- `OPENAI_API_BASE=<base_url>`
- `OPENAI_BASE_URL=<base_url>`
- `OPENAI_API_KEY=<value from api_key_env>`

Example:
- `lambda-ai-gpu` resolves to `openai/meta-llama/Llama-3.1-8B-Instruct`
- Env is wired from `config/models.yaml` (`base_url`) and `VLLM_API_KEY` (`api_key_env`)

If a custom endpoint model is missing its required key env var, Bloom exits early before running the pipeline.

## Configuration

Edit `config/bloom_config.yaml` to customize:
- Test categories (corrigibility, value alignment, uncertainty, instruction following)
- Behavior test prompts and success criteria
- Evaluation metrics
- Comparison thresholds

## Test Categories

| Category | Purpose | Example |
|----------|---------|---------|
| **Corrigibility** | Model accepts correction | "Does the model acknowledge errors?" |
| **Value Alignment** | Model aligns with values | "Does model follow specified values?" |
| **Uncertainty** | Appropriate uncertainty expression | "Does model calibrate confidence?" |
| **Instruction Following** | Nuanced instruction adherence | "Can model follow complex instructions?" |

## Output Format

### Single Evaluation (`bloom_eval_*.json`)

```json
{
  "model_id": "llama2-7b",
  "metrics": {
    "success_rate": 0.85,
    "confidence_calibration": 0.72,
    "response_quality": 4.2
  },
  "num_tests": 50,
  "petri_transcript_dir": "./data/scratch/petri_audits",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

### Comparison (`bloom_comparison_*.json`)

```json
{
  "baseline": { "metrics": {...} },
  "finetuned": { "metrics": {...} },
  "improvements": {
    "success_rate": 0.15,
    "confidence_calibration": 0.12
  }
}
```

### Forbidden Suite Outputs

When `BLOOM_SUITE=forbidden`:
- Suite assets are written under `data/bloom_forbidden_suite/` (or `BLOOM_FORBIDDEN_SUITE_DIR`)
- Per-behavior outputs are written under:
  - `data/phase2_evaluation/forbidden_illegal_activity/`
  - `data/phase2_evaluation/forbidden_deception_manipulation/`
  - etc.
- Aggregate summary:
  - `data/phase2_evaluation/bloom_forbidden_summary_<model_id_sanitized>.json`

### Corrigibility Topic Outputs

When `BLOOM_SUITE=corrigibility` and `BLOOM_CORRIGIBILITY_TOPIC_SET=forbidden`:
- Suite assets are written under `data/bloom_corrigibility_topics_suite/` (or `BLOOM_CORRIGIBILITY_TOPIC_SUITE_DIR`)
- Per-topic outputs are written under:
  - `data/phase2_evaluation/corrigibility_illegal_activity/`
  - `data/phase2_evaluation/corrigibility_deception_manipulation/`
  - etc.
- Aggregate summary:
  - `data/phase2_evaluation/bloom_corrigibility_topics_summary_<model_id_sanitized>.json`

## Workflow Example

```bash
# 1. Fine-tune model (Phase 1)
make sft MODEL=llama2-7b DATASET=corrigibility

# 2. Run audits (Phase 2)
make audit-seeds TARGET_MODEL_ID=llama2-7b SEED_DATASET_NAME=easy

# 3. Evaluate behaviors (Phase 2.5)
make bloom-eval \
    MODEL_ID=llama2-7b \
    ADAPTER_PATH=./checkpoints/llama2_epoch3.safetensors \
    PETRI_TRANSCRIPT_DIR=./data/scratch/petri_audits

# 4. Compare improvements
make bloom-compare \
    BASELINE_MODEL=llama2-7b \
    FINETUNED_MODEL=llama2-7b \
    ADAPTER_PATH=./checkpoints/llama2_epoch3.safetensors

# 5. Run durability analysis (Phase 3)
make survival
```

## Testing

```bash
# Run Bloom evaluation tests
uv run pytest tests/phase2_evaluation/ -v

# Run with coverage
uv run pytest tests/phase2_evaluation/ --cov=src.phase2_evaluation
```

## Common Issues

**Issue**: `ModuleNotFoundError: No module named 'bloom'`
- **Solution**: Run `make setup` to install editable package

**Issue**: `FileNotFoundError: config/bloom_config.yaml`
- **Solution**: Create config from template: `cp config/bloom_config.yaml.template config/bloom_config.yaml`

**Issue**: Model not found in config
- **Solution**: Add model to `config/models.yaml` first

## API Reference

See `src/phase2_evaluation/bloom_eval.py` for:
- `BloomBehaviorEvaluator.evaluate_model()`
- `BloomBehaviorEvaluator.compare_baseline_vs_finetuned()`
- `BloomBehaviorEvaluator._compute_improvements()`
