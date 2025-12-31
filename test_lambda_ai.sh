#!/bin/bash
# Test script for Lambda AI vLLM endpoint

# Configuration - update these if needed
API_KEY="${LAMBDA_AI_API_KEY:-sk-local-test}"
BASE_URL="http://143.47.124.56:8000/v1"
MODEL_NAME="meta-llama/Meta-Llama-3.1-8B"

echo "Testing Lambda AI vLLM endpoint..."
echo "Base URL: $BASE_URL"
echo "Model: $MODEL_NAME"
echo "API Key: $API_KEY"
echo ""

# Test 1: Simple chat completion
echo "=== Test 1: Simple Chat Completion ==="
curl -X POST "${BASE_URL}/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{
    "model": "'"${MODEL_NAME}"'",
    "messages": [
      {
        "role": "user",
        "content": "Hello! Can you respond with just the word test?"
      }
    ],
    "max_tokens": 50,
    "temperature": 0.7
  }' | jq '.'

echo ""
echo ""

# Test 2: List available models (if endpoint supports it)
echo "=== Test 2: List Models (if supported) ==="
curl -X GET "${BASE_URL}/models" \
  -H "Authorization: Bearer ${API_KEY}" 2>/dev/null | jq '.' || echo "Models endpoint not available or returned an error"

echo ""
echo ""

# Test 3: Check if model is accessible
echo "=== Test 3: Model Availability Check ==="
curl -X POST "${BASE_URL}/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{
    "model": "'"${MODEL_NAME}"'",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "What is 2+2?"
      }
    ],
    "max_tokens": 100
  }' | jq '.'

echo ""
echo "Done!"

