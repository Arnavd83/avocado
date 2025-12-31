#!/usr/bin/env python3
"""Test script to verify Inspect AI can connect to our vLLM server"""

import os
import sys
import asyncio

# Set environment variables
os.environ['OPENAI_API_BASE'] = 'http://localhost:8000/v1'
os.environ['OPENAI_API_KEY'] = 'sk-local-test'

# Now try to use Inspect AI
async def test():
    try:
        from inspect_ai.model import get_model
        
        # Try to get the model
        print("Creating model...")
        model = get_model("openai/meta-llama/Meta-Llama-3.1-8B")
        print(f"✓ Successfully created model: {model}")
        
        # Try a simple generation (using the correct Inspect AI API)
        print("\nTesting generation...")
        from inspect_ai.model import GenerateConfig
        from inspect_ai.model import ChatMessage, UserMessage
        
        response = await model.generate(
            input=[UserMessage(content="Say hello in one word")]
        )
        print(f"✓ Response: {response.text}")
        print("\n✓ SUCCESS: Inspect AI can connect to vLLM server!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test())

