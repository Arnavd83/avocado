# MMLU Benchmark Setup Guide

## Dependencies

The MMLU benchmark requires `inspect_ai` which is already included via the `petri` and `tinker-cookbook` dependencies.

### Installing inspect_evals

The MMLU tasks are provided by the `inspect_evals` package. Install it with:

```bash
# Using uv (recommended)
uv pip install inspect_evals

# Or using pip
pip install inspect_evals
```

### Verify Installation

Check that inspect_ai and inspect_evals are installed:

```bash
uv run python -c "import inspect_ai; print('inspect_ai:', inspect_ai.__version__)"
uv run python -c "import inspect_evals; print('inspect_evals:', inspect_evals.__version__)"
```

## Environment Setup

### 1. Configure vLLM API Access

Add to your `.env` file:

```bash
# vLLM API Key
VLLM_API_KEY=your-vllm-api-key-here
OPENROUTER_API_KEY=your-openrouter-api-key-here
```

### 2. Verify Model Configuration

Ensure your model is configured in `config/models.yaml` with:

```yaml
lambda-ai-gpu:
  model_name: openai/meta-llama/Llama-3.1-8B-Instruct
  model_type: openai
  provider: lambda
  base_url: http://100.90.196.92:8000/v1  # Your vLLM server URL
  api_key_env: VLLM_API_KEY
  context_window: 4096
  max_output_tokens: 4096
  supports_vision: false
  supports_function_calling: true
  description: "Meta Llama 3.1 8B Instruct model on Lambda AI GPU"
```

### 3. Verify vLLM Server

Check that your vLLM server is running:

```bash
# Test health endpoint
curl http://100.90.196.92:8000/health

# List loaded models
curl http://100.90.196.92:8000/v1/models \
  -H "Authorization: Bearer $VLLM_API_KEY"
```

## Quick Test

Run a quick test with 10 samples to verify everything works:

```bash
make mmlu-quick MODEL_ID=lambda-ai-gpu
```

For OpenRouter:

```bash
make mmlu-openrouter UTILITY_MODELS='gpt-4o' MMLU_LIMIT=10
```

Expected output:
```
Running MMLU benchmark...
[INFO] Initialized MMLURunner for model: lambda-ai-gpu
[INFO] Running Inspect AI evaluation...
...
MMLU EVALUATION RESULTS
================================================================================
Model: lambda-ai-gpu
Overall Score: 68.50%
Results saved to: data/benchmarks/mmlu/mmlu_lambda-ai-gpu_20260120_123456.json
================================================================================
```

## Troubleshooting

### Import Error: No module named 'inspect_evals'

Install the package:
```bash
uv pip install inspect_evals
```

### Connection Error to vLLM Server

Check:
1. vLLM server is running: `curl http://100.90.196.92:8000/health`
2. Correct base_url in config/models.yaml
3. VLLM_API_KEY is set in .env

### Adapter Not Found

Load the adapter on vLLM first:
```bash
inference-server load-adapter <adapter-name>
```

### API Key Not Found

Ensure VLLM_API_KEY (or the correct api_key_env from models.yaml) is set in your `.env` file.

## Performance Notes

- **Quick test** (10 samples): ~30 seconds
- **Full MMLU** (14,042 samples): 1-2 hours depending on model speed
- Results are cached in `data/benchmarks/mmlu/logs/` for Inspect AI
- Use `--limit` for faster testing during development

## Support

For issues:
1. Check logs in `data/benchmarks/mmlu/logs/`
2. Run with `--list-models` to verify model configuration
3. Test vLLM server connectivity manually with curl
4. Verify all dependencies are installed
