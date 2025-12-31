#!/bin/bash
# Test script to run from your local machine
# Make sure you've started the SSH tunnel first: ./tunnel_vllm.sh

API_KEY="${LAMBDA_AI_API_KEY:-sk-local-test}"
BASE_URL="http://localhost:8000/v1"
MODEL_NAME="meta-llama/Meta-Llama-3.1-8B"

echo "Testing Lambda AI vLLM endpoint via SSH tunnel..."
echo "Make sure ./tunnel_vllm.sh is running in another terminal!"
echo ""

# Test chat completion
echo "Testing chat completion:"
curl -X POST "${BASE_URL}/chat/completions" \
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

