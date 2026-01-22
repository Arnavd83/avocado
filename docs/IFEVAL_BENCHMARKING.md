# IFEval Benchmarking Guide

This guide explains how to run IFEval (Instruction-Following Evaluation) on
fine-tuned models hosted on your GPU instances via vLLM.

## Overview

IFEval evaluates a model's ability to follow automatically verifiable natural
language instructions. It reports prompt-level and instruction-level accuracy
under strict and loose criteria, plus a final accuracy that averages the four.

This integration allows you to:

- Benchmark base models hosted on vLLM
- Evaluate fine-tuned models with LoRA adapters
- Track instruction-following changes from fine-tuning

## Quick Start

### 1. Install Dependencies

```bash
uv pip install inspect_evals
uv pip install git+https://github.com/josejg/instruction_following_eval
```

### 2. Quick Test (10 samples)

```bash
make ifeval-quick MODEL_ID=lambda-ai-gpu
```

### 3. Full Evaluation

```bash
# Base model
make ifeval-benchmark MODEL_ID=lambda-ai-gpu

# Fine-tuned adapter
make ifeval-adapter ADAPTER_NAME=anti-sycophancy-llama
```

### 4. OpenRouter Evaluation

```bash
make ifeval-openrouter UTILITY_MODELS='gpt-4o claude-sonnet-4' IFEVAL_LIMIT=10
```

## Usage

### Make Commands

```bash
# Quick test (10 samples)
make ifeval-quick MODEL_ID=lambda-ai-gpu

# Full benchmark
make ifeval-benchmark MODEL_ID=lambda-ai-gpu

# Test with specific sample limit
make ifeval-benchmark MODEL_ID=lambda-ai-gpu IFEVAL_LIMIT=100

# Evaluate fine-tuned adapter
make ifeval-adapter ADAPTER_NAME=my-adapter

# OpenRouter evaluation
make ifeval-openrouter UTILITY_MODELS='gpt-4o claude-sonnet-4' IFEVAL_LIMIT=10
```

`ifeval-openrouter` uses the space-separated `UTILITY_MODELS` list from the
Makefile to select OpenRouter models.

### Direct Script Usage

```bash
# Quick test
uv run python scripts/run_ifeval_benchmark.py \
    --model-id lambda-ai-gpu \
    --limit 10

# Full evaluation
uv run python scripts/run_ifeval_benchmark.py \
    --model-id lambda-ai-gpu

# With adapter
uv run python scripts/run_ifeval_benchmark.py \
    --model-id lambda-ai-gpu \
    --adapter-name anti-sycophancy-llama

# Custom output directory
uv run python scripts/run_ifeval_benchmark.py \
    --model-id lambda-ai-gpu \
    --output-dir results/ifeval

# OpenRouter
uv run python scripts/run_ifeval_openrouter.py \
    --model-id gpt-4o \
    --limit 10
```

### Command-Line Options

```
--model-id          Model ID from config/models.yaml (required)
--adapter-name      LoRA adapter name (optional, for fine-tuned models)
--tasks             Inspect AI task (default: inspect_evals/ifeval)
--output-dir        Results directory (default: data/benchmarks/ifeval)
--limit             Sample limit for testing (default: None = full eval)
--temperature       Sampling temperature (default: 0.0)
--max-tokens        Max tokens to generate (default: 512)
--list-models       List available models and exit
```

OpenRouter-specific CLI (see `scripts/run_ifeval_openrouter.py`):
```
--model-id          Model ID from config/models.yaml (OpenRouter only)
--tasks             Inspect AI task (default: inspect_evals/ifeval)
--output-dir        Results directory (default: data/benchmarks/ifeval/openrouter)
--limit             Sample limit for testing (default: None = full eval)
--temperature       Sampling temperature (default: 0.0)
--max-tokens        Max tokens to generate (default: 512)
--list-models       List available OpenRouter models and exit
```

## Configuration

### Model Configuration

Models must be defined in `config/models.yaml` with vLLM endpoints:

```yaml
lambda-ai-gpu:
  model_name: openai/meta-llama/Llama-3.1-8B-Instruct
  model_type: openai
  provider: lambda
  base_url: http://100.90.196.92:8000/v1  # vLLM server URL
  api_key_env: VLLM_API_KEY                # API key environment variable
  context_window: 4096
  max_output_tokens: 4096
  description: "Llama 3.1 8B Instruct on Lambda AI GPU"
```

### Environment Variables

Add to `.env`:

```bash
VLLM_API_KEY=your-vllm-api-key
OPENROUTER_API_KEY=your-openrouter-api-key
```

### Adapter Loading

For fine-tuned models, load the adapter on vLLM before benchmarking:

```bash
inference-server load-adapter <adapter-name>
```

## Results

### Output Files

Results are saved to `data/benchmarks/ifeval/`:

```
ifeval/
|- ifeval_<model>_<timestamp>.json     # Timestamped results
|- ifeval_<model>_latest.json          # Latest results
`- logs/                               # Inspect AI logs
```

OpenRouter results are saved to `data/benchmarks/ifeval/openrouter/`.

### Result Format

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
    }
  }
}
```

`overall_score` is the `final_acc` metric reported by the IFEval scorer.

## Performance Tips

- **Quick Testing**: Use `--limit 10` or `--limit 100` during development
- **Determinism**: Use `temperature=0.0` for reproducibility
- **Logs**: Inspect AI logs are stored in `data/benchmarks/ifeval/logs/`
- **First Run**: The HF dataset and NLTK resources download on first use

## Troubleshooting

### Import Error: No module named 'inspect_evals'

```bash
uv pip install inspect_evals
```

### Import Error: No module named 'instruction_following_eval'

```bash
uv pip install git+https://github.com/josejg/instruction_following_eval
```

### Connection Error to vLLM Server

Check server status:
```bash
curl http://100.90.196.92:8000/health

curl http://100.90.196.92:8000/v1/models \
  -H "Authorization: Bearer $VLLM_API_KEY"
```

### Adapter Not Found

Load adapter first:
```bash
inference-server load-adapter <adapter-name>
```

## Architecture

### Components

1. **VLLMInspectModel** (`src/evaluation/vllm_inspect_adapter.py`)
   - Adapts vLLM's OpenAI-compatible API to Inspect AI
   - Handles message formatting and response parsing

2. **InspectEvalRunner** (`src/evaluation/inspect_runner.py`)
   - Shared evaluation workflow used by MMLU + IFEval
   - Manages adapter checks and result persistence

3. **IFEvalRunner** (`src/ifeval/runner.py`)
   - IFEval-specific defaults (tasks, output dir, primary metrics)

4. **run_ifeval_benchmark.py** (`scripts/run_ifeval_benchmark.py`)
   - CLI interface for running IFEval

5. **Makefile Targets**
   - Convenient shortcuts for common operations

## References

- [IFEval Paper](https://arxiv.org/abs/2311.07911)
- [Inspect Evals Repository](https://github.com/UKGovernmentBEIS/inspect_evals)
- [vLLM Documentation](https://docs.vllm.ai/)
