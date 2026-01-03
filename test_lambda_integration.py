#!/usr/bin/env python3
"""
Quick test script to verify Lambda AI integration with emergent-values.
"""

import os
import sys
import asyncio

# Add the compute_utilities directory to the path
sys.path.insert(0, 'external_packages/emergent-values/utility_analysis')

from compute_utilities.utils import create_agent

async def test_lambda_integration():
    """Test that we can create an agent and get a response from Lambda AI."""
    
    # Set the API key (can be any value for vLLM)
    os.environ['LAMBDA_AI_API_KEY'] = 'sk-local-test'
    
    # Set the models config path
    os.environ['MODELS_CONFIG_PATH'] = os.path.abspath('config/models.yaml')
    
    print("Creating Lambda AI agent...")
    try:
        agent = create_agent(
            model_key='lambda-ai-gpu',
            temperature=0.7,  # Add randomness for variety
            max_tokens=3,  # Allow 2-3 tokens to avoid EOT, extraction will clean it
            concurrency_limit=5,
            base_timeout=10
        )
        print(f"✓ Agent created successfully: {type(agent).__name__}")
        print(f"  Model: {agent.model}")
    except Exception as e:
        print(f"✗ Failed to create agent: {e}")
        return False
    
    # Test a simple completion
    print("\nTesting simple completion...")
    test_messages = [
        [
            {"role": "user", "content": "Which is better?\n\nOption A: Apple\nOption B: Orange\n\nAnswer with only 'A' or 'B'."}
        ]
    ] * 100  # Run 5 LLM calls
    
    try:
        responses = await agent.async_completions(test_messages, verbose=True)

        # Print all questions and responses with extraction
        def extract_ab(text):
            """Extract first A or B from response"""
            if not text:
                return 'unparseable'
            for char in text:
                if char in ['A', 'B']:
                    return char
            return 'unparseable'
        
        for i, (messages, response) in enumerate(zip(test_messages, responses)):
            question = messages[0]['content']  # Get the user message
            extracted = extract_ab(response)
            print(f"\n--- LLM Call {i+1} ---")
            print(f"Question: {question}")
            print(f"Raw Response: '{response}'")
            print(f"Extracted: '{extracted}'")

        print(f"\n✓ Completed {len(responses)} LLM calls successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to get response: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = asyncio.run(test_lambda_integration())
    sys.exit(0 if success else 1)

