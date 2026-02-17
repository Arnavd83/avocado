# OpenRouter Model Manager - Quick Start

## What Was Created

This setup provides a unified interface to use Anthropic, OpenAI, Google, and Meta models through OpenRouter.

### Files Created
make audit-seeds BATCH_MAX_PARALLEL=20 BATCH_FAIL_FAST=1 MAX_TURNS=3 SEED_DATASET_NAME=easy


  1. Run make audit-seeds BATCH_MAX_PARALLEL=20 BATCH_FAIL_FAST=1 MAX_TURNS=3 SEED_DATASET_NAME=hard to verify the selection.
  2. Use SEED_DATASET=/path/to/custom.json if you want a one-off dataset outside the naming convention.
do
```
config/models.yaml              # Model configurations
shared/model_manager.py         # Core ModelManager class
shared/__init__.py              # Module exports
shared/examples/usage.py        # Usage examples
shared/cli.py                   # CLI tool
shared/MODEL_MANAGER.md         # Full documentation
```
make audit TARGET_MODEL_ID=lambda-ai-gpu
## Quick Usage

### 1. Command Line Interface

```bash
# List all models
python shared/cli.py list

# List by provider
python shared/cli.py list --provider anthropic
python shared/cli.py list --provider openai
python shared/cli.py list --provider google
python shared/cli.py list --provider meta

# Show model details
python shared/cli.py show claude-sonnet-4
python shared/cli.py show gpt-4o

# View collections
python shared/cli.py collections
python shared/cli.py collection premium
python shared/cli.py collection balanced

# Quick test
python shared/cli.py chat gpt-4o-mini "Hello!"
```

### 2. Python API

```python
from shared import ModelManager

# Initialize
manager = ModelManager()

# Create client
client = manager.create_client()

# Use any model with the same interface!
for model_id in ["claude-sonnet-4", "gpt-4o", "gemini-1.5-pro", "llama-3.1-70b-instruct"]:
    model = manager.get_model(model_id)

    response = client.chat.completions.create(
        model=model.model_name,
        messages=[{"role": "user", "content": "Explain AI briefly."}],
        max_tokens=100
    )

    print(f"{model.provider}: {response.choices[0].message.content}\n")
```

### 3. Discover Models

```python
from shared import ModelManager

manager = ModelManager()

# By provider
anthropic = manager.get_models_by_provider("anthropic")
openai = manager.get_models_by_provider("openai")

# By collection
premium = manager.get_collection("premium")
fast = manager.get_collection("fast")

# By capabilities
long_context = manager.list_models(min_context_window=100000)

# Print nicely
manager.print_models(long_context)
```

## Available Models Summary

### Anthropic (5 models)
- claude-opus-4, claude-sonnet-4, claude-3.5-sonnet, claude-3-opus, claude-3-haiku

### OpenAI (7 models)
- gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo, o1-preview, o1-mini

### Google (5 models)
- gemini-2.0-flash-exp, gemini-1.5-pro, gemini-1.5-flash, gemini-pro, gemini-pro-vision

### Meta Llama (8 models)
- llama-3.3-70b-instruct, llama-3.2-90b-vision-instruct, llama-3.1-405b-instruct, etc.

## Collections

- **premium**: Top-tier models (Claude Opus 4, GPT-4o, Gemini 1.5 Pro, Llama 3.1 405B)
- **balanced**: Good performance/cost (Claude Sonnet 4, GPT-4o-mini, Gemini Flash, Llama 3.3)
- **fast**: Quick and economical (Claude Haiku, GPT-3.5 Turbo, Llama 3.1 8B)
- **vision**: Multimodal models
- **reasoning**: O1 models
- **long_context**: 100k+ context windows

## Example: Switching Providers

```python
from shared import ModelManager

manager = ModelManager()
client = manager.create_client()

# Try the same prompt with different providers
prompt = "Explain quantum computing in one sentence."

models = ["claude-sonnet-4", "gpt-4o-mini", "gemini-1.5-flash", "llama-3.1-70b-instruct"]

for model_id in models:
    model = manager.get_model(model_id)
    response = client.chat.completions.create(
        model=model.model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )
    print(f"{model.provider}: {response.choices[0].message.content}\n")
```

## Example: Streaming

```python
from shared import ModelManager

manager = ModelManager()
client = manager.create_client()
model = manager.get_model("claude-3-haiku")

stream = client.chat.completions.create(
    model=model.model_name,
    messages=[{"role": "user", "content": "Write a short poem"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## Environment Setup

Ensure your `.env` file has:

```bash
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_APP_NAME=avocado
```

## Run Examples

```bash
# Comprehensive examples
python shared/examples/usage.py
```

## Full Documentation

See `shared/MODEL_MANAGER.md` for complete documentation including:
- Advanced usage
- Error handling
- Custom configurations
- Provider-specific features
- Tips and best practices

## Key Benefits

1. **Unified Interface**: Same code works for all providers
2. **Easy Switching**: Change models by updating one string
3. **Rich Metadata**: Access context windows, capabilities, etc.
4. **Collections**: Pre-organized model groups
5. **Type Safe**: Python classes with proper typing
6. **CLI Tools**: Quick exploration without code
7. **OpenAI Compatible**: Uses familiar OpenAI SDK

## Next Steps

1. Try the CLI: `python shared/cli.py list`
2. Run examples: `python shared/examples/usage.py`
3. Integrate into your code: `from shared import ModelManager`
4. Read full docs: `shared/MODEL_MANAGER.md`
