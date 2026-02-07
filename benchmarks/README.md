# Benchmarks Package

This package provides evaluation runners for standard LLM benchmarks, supporting both self-hosted models (via vLLM) and cloud-hosted models (via OpenRouter).

## Supported Benchmarks

| Benchmark | Description | Primary Metric |
|-----------|-------------|----------------|
| **MMLU** | Massive Multitask Language Understanding - tests knowledge across 57 subjects | Accuracy |
| **IFEval** | Instruction Following Evaluation - tests ability to follow detailed instructions | Final Accuracy |

## Prerequisites

1. **Environment Variables**

   For vLLM models:
   ```bash
   VLLM_API_KEY=your-vllm-api-key
   ```

   For OpenRouter models:
   ```bash
   OPENROUTER_API_KEY=your-openrouter-api-key
   ```

2. **Model Configuration**

   Models must be defined in `config/models.yaml`. Example:
   ```yaml
   # vLLM model
   lambda-ai-gpu:
     model_name: openai/meta-llama/Llama-3.1-8B-Instruct
     model_type: openai
     provider: lambda
     base_url: http://100.100.39.70:8000
     api_key_env: VLLM_API_KEY

   # OpenRouter model
   gpt-4o:
     model_name: openrouter/openai/gpt-4o
     model_type: openrouter
     provider: openai
   ```

3. **For vLLM models**: The vLLM server must be running and accessible at the configured `base_url`.

## Quick Start

### Run All Benchmarks on a Model

```bash
# vLLM model (quick test with 10 samples)
python benchmarks/scripts/run_benchmark.py --model-id lambda-ai-gpu --limit 10

# OpenRouter model (full evaluation)
python benchmarks/scripts/run_benchmark.py --model-id gpt-4o --openrouter

# With a fine-tuned adapter (vLLM only)
python benchmarks/scripts/run_benchmark.py --model-id lambda-ai-gpu --adapter-name my-adapter
```

### Run Individual Benchmarks

```bash
# MMLU on vLLM
python benchmarks/scripts/run_benchmark.py --model-id lambda-ai-gpu --benchmarks mmlu --limit 10

# MMLU on OpenRouter
python benchmarks/scripts/run_benchmark.py --model-id gpt-4o --openrouter --benchmarks mmlu --limit 10

# IFEval on vLLM
python benchmarks/scripts/run_benchmark.py --model-id lambda-ai-gpu --benchmarks ifeval --limit 10

# IFEval on OpenRouter
python benchmarks/scripts/run_benchmark.py --model-id gpt-4o --openrouter --benchmarks ifeval --limit 10
```

### List Available Models

```bash
python benchmarks/scripts/run_benchmark.py --list-models
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model-id` | Model ID from config/models.yaml | Required |
| `--adapter-name` | LoRA adapter name (vLLM only) | None |
| `--openrouter` | Use OpenRouter instead of vLLM | False |
| `--benchmarks` | Which benchmarks to run: `mmlu`, `ifeval`, or both | Both |
| `--limit` | Limit samples per benchmark (for testing) | None (full) |
| `--temperature` | Sampling temperature | 0.0 |
| `--max-tokens` | Maximum tokens to generate | 512 |
| `--output-dir` | Base directory for results | `data/benchmarks` |
| `--list-models` | List available models and exit | - |

## Output

### Result Location

Results are saved as JSON files:

```
data/benchmarks/
├── mmlu/
│   ├── mmlu_lambda-ai-gpu_20250205_142530.json    # Timestamped
│   ├── mmlu_lambda-ai-gpu_latest.json             # Latest result
│   ├── logs/                                       # Inspect AI logs
│   └── openrouter/
│       ├── mmlu_gpt-4o_20250205_142530.json
│       └── mmlu_gpt-4o_latest.json
└── ifeval/
    ├── ifeval_lambda-ai-gpu_20250205_142530.json
    ├── ifeval_lambda-ai-gpu_latest.json
    └── openrouter/
        └── ...
```

### Result Format

```json
{
  "model_id": "lambda-ai-gpu",
  "adapter_name": null,
  "timestamp": "2025-02-05T14:25:30.123456",
  "limit": 10,
  "overall_score": 0.75,
  "metrics": {
    "accuracy": {
      "value": 0.75,
      "name": "Accuracy"
    }
  },
  "subjects": {
    "math": 0.80,
    "history": 0.70
  }
}
```

**MMLU-specific fields:**
- `subjects`: Per-subject accuracy scores

**IFEval-specific metrics:**
- `final_acc`: Overall accuracy
- `inst_strict_acc`: Strict instruction adherence
- `prompt_strict_acc`: Strict prompt adherence
- `inst_loose_acc`: Loose instruction adherence
- `prompt_loose_acc`: Loose prompt adherence

## Architecture

```
benchmarks/
├── __init__.py              # Package exports
├── base_runner.py           # InspectEvalRunner base class
├── vllm_adapter.py          # VLLMInspectModel wrapper
├── mmlu/
│   ├── runner.py            # MMLURunner (vLLM)
│   └── openrouter_runner.py # MMLUOpenRouterRunner
├── ifeval/
│   ├── runner.py            # IFEvalRunner (vLLM)
│   └── openrouter_runner.py # IFEvalOpenRouterRunner
└── tests/
    └── test_base_runner.py  # Unit tests
```

### Class Hierarchy

```
InspectEvalRunner (base class)
├── MMLURunner
├── MMLUOpenRouterRunner
├── IFEvalRunner
└── IFEvalOpenRouterRunner
```

## Programmatic Usage

```python
import asyncio
from benchmarks import MMLURunner, IFEvalRunner

async def evaluate_model():
    # Create runner
    runner = MMLURunner(
        model_id="lambda-ai-gpu",
        adapter_name=None,  # Optional: "my-fine-tuned-adapter"
        output_dir="data/benchmarks/mmlu",
        limit=10,  # Optional: limit samples for testing
    )

    # Run evaluation
    results = await runner.run(
        temperature=0.0,
        max_tokens=512,
    )

    print(f"Overall score: {results['overall_score']:.2%}")
    return results

# Run
results = asyncio.run(evaluate_model())
```

### Using OpenRouter

```python
from benchmarks import MMLUOpenRouterRunner

async def evaluate_openrouter():
    runner = MMLUOpenRouterRunner(
        model_id="gpt-4o",
        output_dir="data/benchmarks/mmlu/openrouter",
        limit=10,
    )

    results = await runner.run()
    return results
```

## LoRA Adapter Evaluation

For evaluating fine-tuned models with LoRA adapters:

1. **Load the adapter** on the vLLM server:
   ```bash
   inference-server load-adapter my-adapter
   ```

2. **Run benchmark** with adapter name:
   ```bash
   python benchmarks/scripts/run_benchmark.py \
       --model-id lambda-ai-gpu \
       --adapter-name my-adapter \
       --benchmarks mmlu \
       --limit 10
   ```

Results will be saved with the adapter name in the filename:
```
data/benchmarks/mmlu/mmlu_my-adapter_20250205_142530.json
```

## Running Tests

```bash
# Run all benchmark tests
python -m pytest benchmarks/tests/ -v

# Run specific test class
python -m pytest benchmarks/tests/test_base_runner.py::TestProcessResults -v
```

## Troubleshooting

### "Model not found in config/models.yaml"

Ensure the model is defined in `config/models.yaml` and the model ID matches exactly.

```bash
# List available models
python benchmarks/scripts/run_benchmark.py --list-models
```

### "API key not found"

Set the appropriate environment variable:
```bash
export VLLM_API_KEY=your-key      # For vLLM models
export OPENROUTER_API_KEY=your-key # For OpenRouter models
```

Or add to your `.env` file.

### "vLLM server is not healthy"

Ensure the vLLM server is running at the configured `base_url`:
```bash
curl http://your-vllm-server:8000/health
```

### "Adapter not loaded"

Load the adapter on the vLLM server before running:
```bash
inference-server load-adapter my-adapter
```

### OpenRouter Rate Limits

If you hit rate limits with OpenRouter, try:
- Using `--limit` to reduce sample count
- Waiting and retrying
- Using a model with higher rate limits
