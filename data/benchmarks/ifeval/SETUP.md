# IFEval Benchmark Setup Guide

## Dependencies

The IFEval benchmark requires `inspect_ai` and the `inspect_evals` task bundle.

### Install Inspect Evals

```bash
# Using uv (recommended)
uv pip install inspect_evals

# Or using pip
pip install inspect_evals
```

### Install IFEval Scoring Dependency

IFEval uses the `instruction_following_eval` package (from a fork of the
official implementation). Install it from GitHub:

```bash
uv pip install git+https://github.com/josejg/instruction_following_eval
```

### Verify Installation

```bash
uv run python -c "import inspect_ai; print('inspect_ai:', inspect_ai.__version__)"
uv run python -c "import inspect_evals; print('inspect_evals:', inspect_evals.__version__)"
uv run python -c "import instruction_following_eval; print('instruction_following_eval: OK')"
```

Note: The first run will download the `google/IFEval` dataset and required NLTK
resources. Ensure outbound network access from the machine running the eval.

## Environment Setup

### 1. Configure vLLM API Access

Add to your `.env` file:

```bash
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
curl http://100.90.196.92:8000/health

curl http://100.90.196.92:8000/v1/models \
  -H "Authorization: Bearer $VLLM_API_KEY"
```

## Quick Test

Run a quick test with 10 samples to verify everything works:

```bash
make ifeval-quick MODEL_ID=lambda-ai-gpu
```

For OpenRouter:

```bash
make ifeval-openrouter UTILITY_MODELS='gpt-4o' IFEVAL_LIMIT=10
```

Expected output:
```
Running IFEval benchmark...
[INFO] Initialized IFEval runner for model: lambda-ai-gpu
[INFO] Running Inspect AI evaluation...
...
IFEVAL EVALUATION RESULTS
================================================================================
Model: lambda-ai-gpu
Overall Score: 55.20%
Results saved to: data/benchmarks/ifeval/ifeval_lambda-ai-gpu_20260120_123456.json
================================================================================
```

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
- **Full IFEval** (all samples): a few minutes depending on model speed
- Results are cached in `data/benchmarks/ifeval/logs/` for Inspect AI
- Use `--limit` for faster testing during development

## Support

For issues:
1. Check logs in `data/benchmarks/ifeval/logs/`
2. Run with `--list-models` to verify model configuration
3. Test vLLM server connectivity manually with curl
4. Verify all dependencies are installed
