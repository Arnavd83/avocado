# MMLU Benchmark Results

This directory contains MMLU (Massive Multitask Language Understanding) benchmark results for models evaluated on the vLLM inference server.

## Directory Structure

```
mmlu/
|- README.md                                    # This file
|- logs/                                        # Inspect AI evaluation logs (vLLM)
|- openrouter/                                  # OpenRouter results + logs
|  |- logs/                                     # Inspect AI evaluation logs (OpenRouter)
|  |- mmlu_<model>_<timestamp>.json             # Timestamped results
|  `- mmlu_<model>_latest.json                  # Latest results (for quick access)
|- mmlu_<model>_<timestamp>.json                # Timestamped results (vLLM)
`- mmlu_<model>_latest.json                     # Latest results (for quick access)
```

## Result Format

Each result file contains:

```json
{
  "model_id": "lambda-ai-gpu",
  "adapter_name": "anti-sycophancy-llama",
  "timestamp": "2026-01-20T12:34:56",
  "limit": null,
  "overall_score": 0.685,
  "metrics": {
    "accuracy": {
      "value": 0.685,
      "name": "Accuracy"
    }
  },
  "subjects": {
    "abstract_algebra": 0.35,
    "anatomy": 0.62,
    "astronomy": 0.71,
    ...
  }
}
```

## Running Benchmarks

### Quick Test (10 samples)
```bash
make mmlu-quick MODEL_ID=lambda-ai-gpu
```

### Full Evaluation (Base Model)
```bash
make mmlu-benchmark MODEL_ID=lambda-ai-gpu
```

### Fine-tuned Adapter
```bash
make mmlu-adapter ADAPTER_NAME=anti-sycophancy-llama
```

### OpenRouter
```bash
make mmlu-openrouter UTILITY_MODELS='gpt-4o' MMLU_LIMIT=10
```

### Direct Script Usage
```bash
# Quick test
uv run python scripts/run_mmlu_benchmark.py \
    --model-id lambda-ai-gpu \
    --limit 10

# Full evaluation with adapter
uv run python scripts/run_mmlu_benchmark.py \
    --model-id lambda-ai-gpu \
    --adapter-name my-adapter

# 5-shot MMLU
uv run python scripts/run_mmlu_benchmark.py \
    --model-id lambda-ai-gpu \
    --tasks inspect_evals/mmlu_5_shot

# OpenRouter
uv run python scripts/run_mmlu_openrouter.py \
    --model-id gpt-4o \
    --limit 10
```

## MMLU Details

MMLU consists of 57 subjects across four major domains:

1. **STEM** (18 subjects)
   - Physics, Chemistry, Biology, Computer Science, Mathematics, etc.

2. **Humanities** (13 subjects)
   - History, Philosophy, Law, etc.

3. **Social Sciences** (12 subjects)
   - Psychology, Sociology, Economics, Politics, etc.

4. **Other** (14 subjects)
   - Business, Health, Miscellaneous topics

### Expected Scores

Based on [`capability_scores.csv`](../../../external_packages/emergent-values/utility_analysis/capability_scores.csv):

- Llama-3.1-8B (base): ~63.5%
- Llama-3.1-8B-Instruct: ~68.1%
- Llama-3.1-70B-Instruct: ~83.6%
- GPT-4o: ~88.7%
- Claude-3.5-Sonnet: ~89.3%

Fine-tuned models may show different performance depending on training data and objectives.

## Notes

- Full MMLU evaluation takes 1-2 hours depending on model speed
- Use `--limit` flag for quick testing during development
- Results are automatically saved with timestamps
- The `_latest.json` file is overwritten on each run for convenience
- Inspect AI logs are saved in the `logs/` subdirectory
