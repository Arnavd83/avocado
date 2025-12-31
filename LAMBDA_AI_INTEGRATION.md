# Lambda AI GPU Integration Guide

This document explains how to use your Lambda AI GPU model with both Petri (durability testing) and Emergent Values (value testing) frameworks.

## Setup

### 1. Add vLLM Server API Key to `.env`

Add your vLLM server API key to your `.env` file. This is the API key you set when starting the vLLM server (the `--api-key` parameter or `VLLM_API_KEY` environment variable).

For example, if you started your server with:
```bash
export VLLM_API_KEY="sk-local-test"
python -m vllm.entrypoints.openai.api_server --api-key "$VLLM_API_KEY" ...
```

Then add to your `.env`:
```bash
LAMBDA_AI_API_KEY=sk-local-test
```

**Note**: This is the API key for your vLLM server instance, NOT a Lambda AI cloud API key.

### 2. Model Configuration

The Lambda AI model is already configured in `config/models.yaml` as `lambda-ai-gpu`:

```yaml
lambda-ai-gpu:
  model_name: meta-llama/Meta-Llama-3.1-8B
  model_type: openai
  provider: lambda
  base_url: http://143.47.124.56:8000/v1
  api_key_env: LAMBDA_AI_API_KEY
  context_window: 4096
  max_output_tokens: 4096
  supports_vision: false
  supports_function_calling: true
  description: Meta Llama 3.1 8B model deployed on Lambda AI GPU instance via vLLM server
```

**Important Configuration Notes**:
- **API Key**: Use the API key from your vLLM server (`--api-key` parameter), e.g., `sk-local-test`
- **Base URL**: Includes port `:8000` as vLLM server runs on port 8000
- **Model Name**: Should match the model path used in your vLLM server (`--model` parameter)
- **Context Window**: Set to 4096 to match your vLLM `--max-model-len` parameter

### 3. Verify Model Name

You can check the configured model:

```bash
python scripts/model_cli.py show lambda-ai-gpu
```

## Usage

### Testing with Petri (Durability Testing)

Run a Petri audit using your Lambda AI model as the target:

```bash
make audit TARGET_MODEL_ID=lambda-ai-gpu
```

Or use it as the auditor or judge:

```bash
make audit AUDITOR_MODEL_ID=lambda-ai-gpu TARGET_MODEL_ID=claude-sonnet-4.5
```

The Makefile automatically sets the required environment variables (`OPENAI_API_BASE` and `OPENAI_API_KEY`) for custom endpoints.

### Testing with Emergent Values (Value Testing)

Run utility analysis experiments:

```bash
make utility-analysis UTILITY_MODELS=lambda-ai-gpu
```

Or specify multiple models:

```bash
make utility-analysis UTILITY_MODELS="lambda-ai-gpu gemma-3-27b"
```

### Direct Python Usage

#### Using ModelManager

```python
from src.utils import ModelManager

manager = ModelManager()
model = manager.get_model("lambda-ai-gpu")

# Create client (automatically uses custom base_url and API key)
client = manager.create_client("lambda-ai-gpu")

# Use the client
response = client.chat.completions.create(
    model=model.get_api_model_name(),
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

#### Using with Emergent Values

```python
from utility_analysis.compute_utilities.utils import create_agent

agent = create_agent("lambda-ai-gpu", temperature=0.0, max_tokens=100)
# Use agent with emergent values framework...
```

## Architecture

### How It Works

1. **Model Configuration**: Models with custom endpoints are configured with `base_url` and `api_key_env` fields in `config/models.yaml`.

2. **ModelManager**: 
   - Reads custom `base_url` and `api_key_env` from model config
   - Creates OpenAI SDK clients with custom base URLs
   - Uses environment variables for API keys

3. **Petri Integration**:
   - `scripts/get_model.py` outputs model names formatted for Inspect AI
   - `scripts/get_model_env.py` outputs shell commands to set environment variables
   - Makefile automatically sets environment variables when running audits

4. **Emergent Values Integration**:
   - `create_agent()` function detects custom `base_url` in model config
   - Sets `OPENAI_API_BASE` environment variable for LiteLLM
   - LiteLLM uses the custom endpoint automatically

## Troubleshooting

### Issue: API Key Not Found

**Error**: `LAMBDA_AI_API_KEY not found in environment`

**Solution**: Add `LAMBDA_AI_API_KEY=your_key` to your `.env` file.

### Issue: Connection Refused

**Error**: Connection errors when calling the model

**Solution**: 
1. Verify the IP address in `config/models.yaml` is correct: `http://143.47.124.56/v1`
2. Ensure the Lambda AI instance is running and accessible
3. Check if the endpoint requires HTTPS instead of HTTP

### Issue: Wrong Model Name

**Error**: Model not found errors

**Solution**: Update the `model_name` field in `config/models.yaml` with your actual Lambda AI model name. The format should match what your Lambda AI API expects.

### Issue: Environment Variables Not Set for Petri

**Error**: Custom endpoint not being used in Petri tests

**Solution**: The Makefile should automatically set environment variables. If not working, manually export:

```bash
export OPENAI_API_BASE=http://143.47.124.56/v1
export OPENAI_API_KEY=your_key
make audit TARGET_MODEL_ID=lambda-ai-gpu
```

## Adding More Lambda AI Models

To add additional Lambda AI models:

1. Add entry to `config/models.yaml`:
```yaml
lambda-ai-model-2:
  model_name: openai/another-model
  model_type: openai
  provider: lambda
  base_url: http://143.47.124.56/v1
  api_key_env: LAMBDA_AI_API_KEY
  context_window: 128000
  max_output_tokens: 16384
  supports_vision: false
  supports_function_calling: true
  description: Another Lambda AI model
```

2. Use it immediately with either framework!

## Notes

- The base URL format assumes OpenAI-compatible API endpoints (ends with `/v1`)
- Custom endpoints work seamlessly with both Petri and Emergent Values
- Environment variables are managed automatically by the Makefile
- All existing model management tools work with Lambda AI models

