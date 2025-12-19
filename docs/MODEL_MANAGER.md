# OpenRouter Model Manager

A unified interface for accessing different LLM providers (Anthropic, OpenAI, Google, Meta) through OpenRouter.

## Overview

The Model Manager abstracts away provider-specific differences, allowing you to:
- Use a single API interface for all models
- Easily switch between providers
- Filter and discover models by capabilities
- Access metadata like context windows, vision support, etc.

## Setup

1. Ensure your `.env` file has the OpenRouter API key:
```bash
OPENROUTER_API_KEY=your_key_here
OPENROUTER_APP_NAME=avocado
```

2. The models are configured in `config/models.yaml`

## Basic Usage

### Python API

```python
from src.utils import ModelManager

# Initialize
manager = ModelManager()

# Get a model configuration
model = manager.get_model("claude-sonnet-4")
print(f"Context window: {model.context_window}")
print(f"Supports vision: {model.supports_vision}")

# Create a client (OpenAI SDK compatible)
client = manager.create_client()

# Use any model
response = client.chat.completions.create(
    model=model.model_name,
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Discovering Models

```python
# List models by provider
anthropic_models = manager.get_models_by_provider("anthropic")
openai_models = manager.get_models_by_provider("openai")
google_models = manager.get_models_by_provider("google")
meta_models = manager.get_models_by_provider("meta")

# Get models from collections
premium = manager.get_collection("premium")
fast = manager.get_collection("fast")
vision_models = manager.get_collection("vision")

# Filter by capabilities
vision_capable = manager.list_models(supports_vision=True)
long_context = manager.list_models(min_context_window=100000)
claude_with_tools = manager.list_models(
    provider="anthropic",
    supports_function_calling=True
)

# Print formatted table
manager.print_models(vision_capable)
```

### Switching Between Providers

```python
# Same code works for all providers!
models_to_try = ["claude-sonnet-4", "gpt-4o", "gemini-1.5-pro", "llama-3.3-70b-instruct"]

client = manager.create_client()

for model_id in models_to_try:
    model = manager.get_model(model_id)

    response = client.chat.completions.create(
        model=model.model_name,
        messages=[{"role": "user", "content": "Explain AI in one sentence."}]
    )

    print(f"{model.provider}: {response.choices[0].message.content}")
```

### Streaming Responses

```python
model = manager.get_model("claude-3-haiku")
client = manager.create_client()

stream = client.chat.completions.create(
    model=model.model_name,
    messages=[{"role": "user", "content": "Write a haiku"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## Command-Line Interface

The CLI provides quick access to model information and testing:

### List all models
```bash
python scripts/model_cli.py list
```

### Filter models
```bash
# By provider
python scripts/model_cli.py list --provider anthropic

# Vision-capable models only
python scripts/model_cli.py list --vision

# Models with 100k+ context
python scripts/model_cli.py list --min-context 100000
```

### Show model details
```bash
python scripts/model_cli.py show claude-sonnet-4
```

### Explore collections
```bash
# List all collections
python scripts/model_cli.py collections

# Show models in a collection
python scripts/model_cli.py collection premium
python scripts/model_cli.py collection vision
```

### List providers
```bash
python scripts/model_cli.py providers
```

### Quick chat test
```bash
# Send a message
python scripts/model_cli.py chat gpt-4o-mini "What is quantum computing?"

# Stream response
python scripts/model_cli.py chat claude-3-haiku "Tell me a joke" --stream

# Control max tokens
python scripts/model_cli.py chat llama-3.1-8b-instruct "Explain AI" --max-tokens 200
```

## Examples

### Run the comprehensive example script
```bash
python examples/model_manager_usage.py
```

This demonstrates:
- Basic usage
- Listing by provider
- Collections
- Filtering
- Provider comparison
- Chat completions
- Streaming

## Available Models

### Anthropic (Claude)
- `claude-opus-4` - Most capable, 200k context
- `claude-sonnet-4` - Balanced, 200k context
- `claude-3.5-sonnet` - Previous gen
- `claude-3-opus` - Previous gen
- `claude-3-haiku` - Fast and economical

### OpenAI
- `gpt-4o` - Multimodal flagship
- `gpt-4o-mini` - Fast and affordable
- `gpt-4-turbo` - Enhanced GPT-4
- `gpt-4` - Original
- `gpt-3.5-turbo` - Cost-effective
- `o1-preview` - Reasoning model
- `o1-mini` - Fast reasoning

### Google
- `gemini-2.0-flash-exp` - Experimental, 1M context
- `gemini-1.5-pro` - 2M context window!
- `gemini-1.5-flash` - Fast, 1M context
- `gemini-pro` - Previous gen
- `gemini-pro-vision` - With vision

### Meta (Llama)
- `llama-3.3-70b-instruct` - Latest 70B
- `llama-3.2-90b-vision-instruct` - Vision capable
- `llama-3.2-11b-vision-instruct` - Smaller vision
- `llama-3.1-405b-instruct` - Largest model
- `llama-3.1-70b-instruct` - 70B variant
- `llama-3.1-8b-instruct` - Efficient 8B

## Collections

Pre-organized model groups for easy selection:

- **premium**: Top-tier models for complex tasks
- **balanced**: Good performance/cost ratio
- **fast**: Quick and economical
- **vision**: Multimodal capabilities
- **reasoning**: Specialized reasoning (O1 models)
- **long_context**: 100k+ context windows

## Advanced Usage

### Using Anthropic SDK Directly

For Claude models, you can optionally use the Anthropic SDK directly (bypassing OpenRouter):

```python
# Requires ANTHROPIC_API_KEY in .env
client = manager.create_client("claude-sonnet-4", use_anthropic_sdk=True)

# Now using Anthropic SDK
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Custom Model Configuration

You can modify `config/models.yaml` to:
- Add new models as they become available on OpenRouter
- Adjust metadata like context windows
- Create custom collections
- Add model-specific parameters

### Batch Processing

```python
manager = ModelManager()
client = manager.create_client()

# Process with different models
tasks = [
    ("claude-sonnet-4", "Complex reasoning task..."),
    ("gpt-4o-mini", "Simple classification task..."),
    ("llama-3.1-8b-instruct", "Quick generation task..."),
]

for model_id, prompt in tasks:
    model = manager.get_model(model_id)
    response = client.chat.completions.create(
        model=model.model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=model.max_output_tokens
    )
    print(f"{model_id}: {response.choices[0].message.content}")
```

## Error Handling

```python
try:
    model = manager.get_model("nonexistent-model")
except KeyError as e:
    print(f"Model not found: {e}")

try:
    collection = manager.get_collection("nonexistent-collection")
except KeyError as e:
    print(f"Collection not found: {e}")

try:
    client = manager.create_client()
    response = client.chat.completions.create(...)
except Exception as e:
    print(f"API error: {e}")
```

## Tips

1. **Start with collections**: Use `manager.get_collection("balanced")` for general tasks
2. **Check capabilities**: Use `model.supports_vision` before sending images
3. **Context limits**: Check `model.context_window` to avoid exceeding limits
4. **Cost optimization**: Use `fast` collection for development/testing
5. **Vision tasks**: Filter with `list_models(supports_vision=True)`
6. **Long documents**: Use `list_models(min_context_window=100000)`

## Configuration

The `config/models.yaml` structure:

```yaml
defaults:
  provider: openrouter
  base_url: https://openrouter.ai/api/v1
  api_key_env: OPENROUTER_API_KEY

model-id:
  model_name: "provider/model-name"  # OpenRouter model identifier
  provider: provider_name             # Original provider
  context_window: 128000              # Max input tokens
  max_output_tokens: 4096             # Max output tokens
  supports_vision: true               # Vision capability
  supports_function_calling: true     # Tool use capability
  description: "Model description"

collections:
  collection-name:
    - model-id-1
    - model-id-2
```

## Resources

- [OpenRouter Documentation](https://openrouter.ai/docs)
- [OpenRouter Models](https://openrouter.ai/models)
- [OpenAI SDK](https://github.com/openai/openai-python)
- [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python)
