#!/bin/bash
# Test script to verify environment variable propagation

echo "=== Testing environment variable propagation ==="
echo ""

# Simulate what the Makefile does
GET_MODEL_ENV="uv run python scripts/get_model_env.py"

echo "1. Getting environment variables for lambda-ai-gpu:"
ENV_VARS=$($GET_MODEL_ENV lambda-ai-gpu 2>/dev/null)
echo "$ENV_VARS"
echo ""

echo "2. Evaluating environment variables:"
eval "$ENV_VARS"
echo "OPENAI_API_BASE=$OPENAI_API_BASE"
echo "OPENAI_API_KEY=$OPENAI_API_KEY"
echo ""

echo "3. Testing with env command (like Makefile does):"
env OPENAI_API_BASE="$OPENAI_API_BASE" OPENAI_API_KEY="$OPENAI_API_KEY" \
  python3 -c "import os; print('OPENAI_API_BASE:', os.environ.get('OPENAI_API_BASE')); print('OPENAI_API_KEY:', os.environ.get('OPENAI_API_KEY')[:10] + '...')"
echo ""

echo "4. Testing with uv run:"
env OPENAI_API_BASE="$OPENAI_API_BASE" OPENAI_API_KEY="$OPENAI_API_KEY" \
  uv run python -c "import os; print('OPENAI_API_BASE:', os.environ.get('OPENAI_API_BASE')); print('OPENAI_API_KEY:', os.environ.get('OPENAI_API_KEY')[:10] + '...')"
echo ""

