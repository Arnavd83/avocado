#!/bin/bash
# Test script to run ON THE GPU SERVER (ssh ubuntu@143.47.124.56)
# This tests the vLLM endpoint locally

API_KEY="sk-local-test"
BASE_URL="http://localhost:8000/v1"
MODEL_NAME="meta-llama/Meta-Llama-3.1-8B"

echo "=== Testing vLLM server locally ==="
echo ""

# Test 1: List available models
echo "1. Listing available models:"
curl -s "${BASE_URL}/models" \
  -H "Authorization: Bearer ${API_KEY}" | jq '.'

echo ""
echo ""

# Test 2: Simple chat completion
echo "2. Testing chat completion:"
curl -s -X POST "${BASE_URL}/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{
    "model": "'"${MODEL_NAME}"'",
    "messages": [
      {
        "role": "user",
        "content": "Say hello in one word."
      }
    ],
    "max_tokens": 10,
    "temperature": 0.7
  }' | jq '.'

echo ""
echo "Done!"

