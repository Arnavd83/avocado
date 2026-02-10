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
