#!/usr/bin/env python3
"""
Quick test script to verify Lambda AI integration with emergent-values.
"""

import os
import sys
import asyncio
import re

# Add the compute_utilities directory to the path
sys.path.insert(0, 'external_packages/emergent-values/utility_analysis')

from compute_utilities.utils import create_agent

async def test_lambda_integration():
    """Test that we can create an agent and get a response from Lambda AI."""
    
    # Set the API key (can be any value for vLLM)
    os.environ['LAMBDA_AI_API_KEY'] = 'sk-local-test'
    
    # Set the models config path
    os.environ['MODELS_CONFIG_PATH'] = os.path.abspath('config/models.yaml')
    
    constraint_mode = os.environ.get("LAMBDA_CONSTRAINT_MODE", "regex").strip().lower()
    guided_choice = None
    guided_regex = None
    max_tokens_override = None
    temperature_override = None
    if constraint_mode == "choice":
        guided_choice = ["A", "B"]
    elif constraint_mode == "regex":
        guided_regex = r" ?(A|B)"
    elif constraint_mode in ("none", ""):
        pass
    else:
        raise ValueError(
            f"Unknown LAMBDA_CONSTRAINT_MODE '{constraint_mode}'. "
            "Use 'choice', 'regex', or 'none'."
        )

    if guided_choice is not None or guided_regex is not None:
        # Use tighter overrides when constrained decoding is enabled.
        max_tokens_override = 2
        temperature_override = 0.0

    if guided_choice is not None:
        constraint_label = f"guided_choice={guided_choice}"
    elif guided_regex is not None:
        constraint_label = f"guided_regex={guided_regex}"
    else:
        constraint_label = "no constraints"

    print(f"Creating Lambda AI agent ({constraint_label})...")
    try:
        agent = create_agent(
            model_key='lambda-ai-gpu',
            temperature=0.7,  # Add randomness for variety
            max_tokens=3,  # Allow 2-3 tokens to avoid EOT, extraction will clean it
            concurrency_limit=5,
            base_timeout=10,
            guided_choice=guided_choice,
            guided_regex=guided_regex,
            max_tokens_override=max_tokens_override,
            temperature_override=temperature_override
        )
        print(f"✓ Agent created successfully: {type(agent).__name__}")
        print(f"  Model: {agent.model}")
        if guided_choice is not None or guided_regex is not None:
            print(f"  Constrained decoding: {constraint_label}")
            print(f"  Overrides: max_tokens={max_tokens_override}, temperature={temperature_override}")
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
        
        constraint_pattern = None
        if guided_choice is not None or guided_regex is not None:
            choices = guided_choice or ["A", "B"]
            choices_pattern = "|".join(re.escape(choice) for choice in choices)
            constraint_pattern = re.compile(rf"^\s*({choices_pattern})\s*$")

        constraint_violations = 0
        for i, (messages, response) in enumerate(zip(test_messages, responses)):
            question = messages[0]['content']  # Get the user message
            extracted = extract_ab(response)
            print(f"\n--- LLM Call {i+1} ---")
            print(f"Question: {question}")
            print(f"Raw Response: '{response}'")
            print(f"Extracted: '{extracted}'")
            if constraint_pattern is not None and response is not None:
                if not constraint_pattern.match(response):
                    constraint_violations += 1
                    print("Constraint violation: response does not match allowed pattern")

        print(f"\n✓ Completed {len(responses)} LLM calls successfully")
        if constraint_pattern is not None:
            print(f"Constraint violations: {constraint_violations}")
        return True
    except Exception as e:
        print(f"✗ Failed to get response: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = asyncio.run(test_lambda_integration())
    sys.exit(0 if success else 1)
