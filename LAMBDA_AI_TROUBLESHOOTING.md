# Lambda AI vLLM Integration Troubleshooting

## Current Issue

Inspect AI is connecting to OpenAI's default API instead of the vLLM server, even when `OPENAI_API_BASE` is set.

## Root Cause

Inspect AI's `openai/` provider prefix appears to NOT respect `OPENAI_API_BASE` environment variable. When you specify `openai/meta-llama/Meta-Llama-3.1-8B`, Inspect AI connects directly to OpenAI's API, which rejects the model name.

## Potential Solutions

### Option 1: Check if Inspect AI supports custom endpoints differently

Inspect AI might require a different configuration method for custom endpoints. Check the Inspect AI documentation or source code for:
- Custom endpoint configuration
- Alternative provider identifiers
- Configuration file options

### Option 2: Use a proxy/wrapper

Create a local proxy that forwards requests from OpenAI's API format to your vLLM server.

### Option 3: Use a different approach

Instead of using Inspect AI's `openai/` provider, we might need to:
- Create a custom model provider for Inspect AI
- Use Inspect AI's extensibility features to add vLLM support
- Use a different evaluation framework that better supports custom endpoints

## Verification Steps

1. ✅ SSH tunnel works (curl commands succeed)
2. ✅ vLLM server is running and accessible
3. ✅ Model name `meta-llama/Meta-Llama-3.1-8B` is correct
4. ✅ Environment variables are set correctly
5. ❌ Inspect AI ignores `OPENAI_API_BASE` when using `openai/` prefix

## Next Steps

1. Check Inspect AI documentation/source for custom endpoint support
2. Consider creating a custom provider or wrapper
3. Alternative: Use a different evaluation framework or approach

