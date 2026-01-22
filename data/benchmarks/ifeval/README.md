# IFEval Benchmark Results

This directory contains IFEval (Instruction-Following Evaluation) benchmark
results for models evaluated on the vLLM inference server.

## Directory Structure

```
ifeval/
|- README.md                                    # This file
|- logs/                                        # Inspect AI evaluation logs (vLLM)
|- openrouter/                                  # OpenRouter results + logs
|  |- logs/                                     # Inspect AI evaluation logs (OpenRouter)
|  |- ifeval_<model>_<timestamp>.json           # Timestamped results
|  `- ifeval_<model>_latest.json                # Latest results (for quick access)
|- ifeval_<model>_<timestamp>.json              # Timestamped results (vLLM)
`- ifeval_<model>_latest.json                   # Latest results (for quick access)
```

## Result Format

Each result file contains:

```json
{
  "model_id": "lambda-ai-gpu",
  "adapter_name": "anti-sycophancy-llama",
  "timestamp": "2026-01-20T12:34:56",
  "limit": null,
  "overall_score": 0.552,
  "metrics": {
    "final_acc": {
      "value": 0.552,
      "name": "final_acc"
    },
    "inst_strict_acc": {
      "value": 0.487,
      "name": "inst_strict_acc"
    },
    "inst_loose_acc": {
      "value": 0.621,
      "name": "inst_loose_acc"
    }
  }
}
```

`overall_score` is set to the `final_acc` metric from the IFEval scorer.

## Running Benchmarks

### Quick Test (10 samples)
```bash
make ifeval-quick MODEL_ID=lambda-ai-gpu
```

### Full Evaluation (Base Model)
```bash
make ifeval-benchmark MODEL_ID=lambda-ai-gpu
```

### Fine-tuned Adapter
```bash
make ifeval-adapter ADAPTER_NAME=anti-sycophancy-llama
```

### OpenRouter
```bash
make ifeval-openrouter UTILITY_MODELS='gpt-4o claude-sonnet-4' IFEVAL_LIMIT=10
```

### Direct Script Usage
```bash
# Quick test
uv run python scripts/run_ifeval_benchmark.py \
    --model-id lambda-ai-gpu \
    --limit 10

# Full evaluation with adapter
uv run python scripts/run_ifeval_benchmark.py \
    --model-id lambda-ai-gpu \
    --adapter-name my-adapter
```

## Notes

- IFEval evaluates instruction-following accuracy at prompt and instruction levels.
- Use `--limit` for quick testing during development.
- Results are automatically saved with timestamps.
- The `_latest.json` file is overwritten on each run for convenience.
- Inspect AI logs are saved in the `logs/` subdirectory.
