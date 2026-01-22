# MMLU Benchmarking Guide

This guide explains how to run MMLU (Massive Multitask Language Understanding) benchmarks on fine-tuned models hosted on your GPU instances via vLLM, plus OpenRouter models for baseline comparisons.

## Overview

The MMLU benchmark evaluates language models across 57 diverse subjects including STEM, humanities, social sciences, and more. This integration allows you to:

- Benchmark base models hosted on vLLM
- Evaluate fine-tuned models with LoRA adapters
- Compare performance against known baselines
- Run OpenRouter baselines with the same task settings
- Track capability changes from fine-tuning

## Quick Start

### 1. Install Dependencies

```bash
# Install inspect_evals (if not already installed)
uv pip install inspect_evals

# Verify setup
python scripts/test_mmlu_setup.py --model-id lambda-ai-gpu
```

### 2. Quick Test (10 samples)

```bash
make mmlu-quick MODEL_ID=lambda-ai-gpu
```

### 3. Full Evaluation

```bash
# Base model
make mmlu-benchmark MODEL_ID=lambda-ai-gpu

# Fine-tuned adapter
make mmlu-adapter ADAPTER_NAME=anti-sycophancy-llama
```

### 4. OpenRouter Evaluation

```bash
make mmlu-openrouter UTILITY_MODELS='gpt-4o claude-sonnet-4' MMLU_LIMIT=10
```

## Usage

### Make Commands

The easiest way to run MMLU benchmarks is via Make:

```bash
# Quick test (10 samples, ~30 seconds)
make mmlu-quick MODEL_ID=lambda-ai-gpu

# Full benchmark (all 14,042 samples, ~1-2 hours)
make mmlu-benchmark MODEL_ID=lambda-ai-gpu

# Test with specific sample limit
make mmlu-benchmark MODEL_ID=lambda-ai-gpu MMLU_LIMIT=100

# Evaluate fine-tuned adapter
make mmlu-adapter ADAPTER_NAME=my-adapter

# With custom limit for adapter
make mmlu-adapter ADAPTER_NAME=my-adapter MMLU_LIMIT=50

# OpenRouter evaluation
make mmlu-openrouter UTILITY_MODELS='gpt-4o claude-sonnet-4' MMLU_LIMIT=10

# List available models
make list-models
```

`mmlu-openrouter` uses the space-separated `UTILITY_MODELS` list from the
Makefile to select OpenRouter models.

### Direct Script Usage

For more control, use the Python script directly:

```bash
# Quick test
uv run python scripts/run_mmlu_benchmark.py \
    --model-id lambda-ai-gpu \
    --limit 10

# Full evaluation
uv run python scripts/run_mmlu_benchmark.py \
    --model-id lambda-ai-gpu

# With adapter
uv run python scripts/run_mmlu_benchmark.py \
    --model-id lambda-ai-gpu \
    --adapter-name anti-sycophancy-llama

# Use 5-shot prompting (instead of 0-shot)
uv run python scripts/run_mmlu_benchmark.py \
    --model-id lambda-ai-gpu \
    --tasks inspect_evals/mmlu_5_shot

# Custom output directory
uv run python scripts/run_mmlu_benchmark.py \
    --model-id lambda-ai-gpu \
    --output-dir results/custom_mmlu

# OpenRouter
uv run python scripts/run_mmlu_openrouter.py \
    --model-id gpt-4o \
    --limit 10

# Adjust sampling parameters
uv run python scripts/run_mmlu_benchmark.py \
    --model-id lambda-ai-gpu \
    --temperature 0.3 \
    --max-tokens 1024
```

### Command-Line Options

```
--model-id          Model ID from config/models.yaml (required)
--adapter-name      LoRA adapter name (optional, for fine-tuned models)
--tasks             Inspect AI task (default: inspect_evals/mmlu_0_shot)
--output-dir        Results directory (default: data/benchmarks/mmlu)
--limit             Sample limit for testing (default: None = full eval)
--temperature       Sampling temperature (default: 0.0)
--max-tokens        Max tokens to generate (default: 512)
--list-models       List available models and exit
```

OpenRouter-specific CLI (see `scripts/run_mmlu_openrouter.py`):
```
--model-id          Model ID from config/models.yaml (OpenRouter only)
--tasks             Inspect AI task (default: inspect_evals/mmlu_0_shot)
--output-dir        Results directory (default: data/benchmarks/mmlu/openrouter)
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

Results are saved to `data/benchmarks/mmlu/`:

```
mmlu/
├── mmlu_<model>_<timestamp>.json     # Timestamped results
├── mmlu_<model>_latest.json          # Latest results
└── logs/                             # Inspect AI logs
```

OpenRouter results are saved to `data/benchmarks/mmlu/openrouter/`.

### Result Format

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

## Expected Performance

Based on [`capability_scores.csv`](../external_packages/emergent-values/utility_analysis/capability_scores.csv):

| Model | MMLU Score |
|-------|-----------|
| Llama-3.1-8B (base) | 63.5% |
| Llama-3.1-8B-Instruct | 68.1% |
| Llama-3.1-70B-Instruct | 83.6% |
| GPT-4o | 88.7% |
| Claude-3.5-Sonnet | 89.3% |

Fine-tuned models may show different performance depending on:
- Training objectives
- Training data composition
- Fine-tuning hyperparameters
- Potential capability trade-offs

## MMLU Details

### Subjects (57 total)

The benchmark covers four major domains:

**STEM (18 subjects)**
- abstract_algebra, astronomy, college_biology, college_chemistry
- college_computer_science, college_mathematics, college_physics
- computer_security, conceptual_physics, electrical_engineering
- elementary_mathematics, high_school_biology, high_school_chemistry
- high_school_computer_science, high_school_mathematics
- high_school_physics, high_school_statistics, machine_learning

**Humanities (13 subjects)**
- formal_logic, high_school_european_history, high_school_us_history
- high_school_world_history, international_law, jurisprudence
- logical_fallacies, moral_disputes, moral_scenarios, philosophy
- prehistory, professional_law, world_religions

**Social Sciences (12 subjects)**
- econometrics, high_school_geography, high_school_government_and_politics
- high_school_macroeconomics, high_school_microeconomics
- high_school_psychology, human_sexuality, professional_psychology
- public_relations, security_studies, sociology, us_foreign_policy

**Other (14 subjects)**
- anatomy, business_ethics, clinical_knowledge, college_medicine
- global_facts, human_aging, management, marketing, medical_genetics
- miscellaneous, nutrition, professional_accounting, professional_medicine
- virology

### Task Variants

- `inspect_evals/mmlu_0_shot` - Standard MMLU with no examples (recommended)
- `inspect_evals/mmlu_5_shot` - MMLU with 5 in-context examples per question

## Performance Tips

### Speed Optimization

- **Quick Testing**: Use `--limit 10` or `--limit 100` during development
- **Parallel Evaluation**: Not currently supported, but can be added if needed
- **Caching**: Inspect AI caches results in the logs directory
- **First Run**: The HF dataset will download on first use

### Timing

- **Quick test (10 samples)**: ~30 seconds
- **Small test (100 samples)**: ~5 minutes
- **Full MMLU (14,042 samples)**: 1-2 hours (depends on model speed)

### Best Practices

1. **Start with quick tests** to verify setup
2. **Use deterministic sampling** (temperature=0.0) for reproducibility
3. **Monitor first few samples** to catch configuration issues early
4. **Save full results** with timestamps for tracking progress over time
5. **Compare against baselines** from capability_scores.csv

## Troubleshooting

### Common Issues

#### Import Error: No module named 'inspect_evals'

```bash
uv pip install inspect_evals
```

#### Connection Error to vLLM Server

Check server status:
```bash
curl http://100.90.196.92:8000/health

# List models
curl http://100.90.196.92:8000/v1/models \
  -H "Authorization: Bearer $VLLM_API_KEY"
```

#### Adapter Not Found

Load adapter first:
```bash
inference-server load-adapter <adapter-name>
```

#### API Key Not Found

Ensure environment variable is set:
```bash
echo $VLLM_API_KEY  # Should print your key
```

Add to `.env` if missing:
```bash
VLLM_API_KEY=your-key-here
```

### Verification Script

Run the setup test to diagnose issues:

```bash
python scripts/test_mmlu_setup.py --model-id lambda-ai-gpu
```

This checks:
- Required packages installed
- Environment variables set
- Model configuration valid
- vLLM server accessible
- Models loaded on server

### Debug Mode

For detailed debugging:

```bash
# Run with Python logging
PYTHONPATH=. python -u scripts/run_mmlu_benchmark.py \
    --model-id lambda-ai-gpu \
    --limit 5 \
    2>&1 | tee mmlu_debug.log
```

## Architecture

### Components

1. **VLLMInspectModel** (`src/evaluation/vllm_inspect_adapter.py`)
   - Adapts vLLM's OpenAI-compatible API to Inspect AI's Model protocol
   - Handles message formatting and response parsing

2. **MMLURunner** (`src/evaluation/mmlu_runner.py`)
   - Orchestrates evaluation workflow
   - Manages adapter loading and results processing
   - Integrates with ModelManager for configuration

3. **run_mmlu_benchmark.py** (`scripts/run_mmlu_benchmark.py`)
   - CLI interface for running benchmarks
   - Handles argument parsing and output formatting

4. **Makefile Targets**
   - Convenient shortcuts for common operations
   - Environment variable management

### Data Flow

```
config/models.yaml → ModelManager → VLLMClient
                                        ↓
                              Check/Load Adapter
                                        ↓
                    VLLMInspectModel (API adapter)
                                        ↓
                        Inspect AI Evaluation
                                        ↓
                        Process & Save Results
                                        ↓
                    data/benchmarks/mmlu/*.json
```

## Integration with Other Tools

### Comparing Results

Compare with baseline scores:

```bash
# Show baseline scores
cat external_packages/emergent-values/utility_analysis/capability_scores.csv

# Your results
cat data/benchmarks/mmlu/mmlu_lambda-ai-gpu_latest.json | jq '.overall_score'
```

### Tracking Fine-Tuning Impact

```bash
# Benchmark base model
make mmlu-benchmark MODEL_ID=lambda-ai-gpu > base_results.txt

# Fine-tune model...

# Benchmark fine-tuned adapter
make mmlu-adapter ADAPTER_NAME=my-adapter > finetuned_results.txt

# Compare
diff base_results.txt finetuned_results.txt
```

### Automated Testing

Add to CI/CD pipeline:

```bash
# Quick smoke test
make mmlu-quick MODEL_ID=lambda-ai-gpu || exit 1

# Or full evaluation for releases
make mmlu-benchmark MODEL_ID=lambda-ai-gpu
```

## References

- [MMLU Paper](https://arxiv.org/abs/2009.03300)
- [Inspect AI Documentation](https://ukgovernmentbeis.github.io/inspect_ai/)
- [Inspect Evals Repository](https://github.com/UKGovernmentBEIS/inspect_evals)
- [vLLM Documentation](https://docs.vllm.ai/)

## Support

For issues or questions:
1. Check `data/benchmarks/mmlu/SETUP.md` for setup instructions
2. Run `python scripts/test_mmlu_setup.py` to diagnose problems
3. Review logs in `data/benchmarks/mmlu/logs/`
4. Check that vLLM server is accessible and healthy
