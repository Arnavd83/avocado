#!/usr/bin/env python3
"""
Test script for Lambda AI integration - single question repeated 100 times.
Tests consistency of responses to the same question.
"""

import os
import sys
import asyncio
import re

# Add the compute_utilities directory to the path
sys.path.insert(0, 'external_packages/emergent-values/utility_analysis')

from compute_utilities.utils import create_agent

async def test_lambda_single_question():
    """Test the same question 100 times to check consistency."""
    
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
        guided_choice = [" A", " B"]
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
        temperature_override = 1.0

    if guided_choice is not None:
        constraint_label = f"structured_choice={guided_choice}"
    elif guided_regex is not None:
        constraint_label = f"structured_regex={guided_regex}"
    else:
        constraint_label = "no constraints"

    logprobs_enabled = os.environ.get("LAMBDA_LOGPROBS", "0").strip().lower() in ("1", "true", "yes")
    top_logprobs = int(os.environ.get("LAMBDA_TOP_LOGPROBS", "5"))

    print(f"Creating Lambda AI agent ({constraint_label})...")
    try:
        agent = create_agent(
            model_key='lambda-ai-gpu',
            temperature=1.0,  # Add randomness for variety
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
            print(f"  Structured outputs: {constraint_label}")
            print(f"  Overrides: max_tokens={max_tokens_override}, temperature={temperature_override}")
    except Exception as e:
        print(f"✗ Failed to create agent: {e}")
        return False
    
    # Test a single question 100 times
    print("\nTesting single question 100 times...")
    single_question = "Which is better?\n\nOption A: Coffee\nOption B: Tea\n\nAnswer with only 'A' or 'B'."
    
    test_messages = [
        [
            {"role": "user", "content": single_question}
        ]
    ] * 100  # Same question 100 times
    
    try:
        responses = await agent.async_completions(
            test_messages,
            verbose=True,
            return_logprobs=logprobs_enabled,
            top_logprobs=top_logprobs if logprobs_enabled else None,
        )

        # Print all questions and responses with extraction
        def extract_ab(text):
            """Extract first A or B from response"""
            if not text:
                return 'unparseable'
            for char in text:
                if char in ['A', 'B']:
                    return char
            return 'unparseable'

        def _get_logprobs_content(logprobs_payload):
            if logprobs_payload is None:
                return None
            if isinstance(logprobs_payload, dict):
                return logprobs_payload.get("content")
            return getattr(logprobs_payload, "content", None)

        def _normalize_logprobs_entry(entry):
            if isinstance(entry, dict):
                return entry.get("token"), entry.get("logprob")
            return getattr(entry, "token", None), getattr(entry, "logprob", None)

        def extract_ab_logprobs(logprobs_payload):
            content = _get_logprobs_content(logprobs_payload)
            if not content:
                return None, None, None, None
            token_info = content[0]
            if isinstance(token_info, dict):
                top_logprobs = token_info.get("top_logprobs")
            else:
                top_logprobs = getattr(token_info, "top_logprobs", None)

            entries = []
            if isinstance(top_logprobs, dict):
                entries = [{"token": token, "logprob": logprob} for token, logprob in top_logprobs.items()]
            elif isinstance(top_logprobs, list):
                entries = top_logprobs
            if not entries:
                entries = [token_info]

            logprob_space_a = None
            logprob_space_b = None
            logprob_a = None
            logprob_b = None
            for entry in entries:
                token, logprob = _normalize_logprobs_entry(entry)
                if token is None or logprob is None:
                    continue
                if token == " A":
                    logprob_space_a = logprob
                elif token == " B":
                    logprob_space_b = logprob
                elif token == "A":
                    logprob_a = logprob
                elif token == "B":
                    logprob_b = logprob
            return logprob_space_a, logprob_space_b, logprob_a, logprob_b
        
        constraint_pattern = None
        if guided_choice is not None or guided_regex is not None:
            constraint_pattern = re.compile(r"^(A|B)$")

        constraint_violations = 0
        
        # Count responses
        a_count = 0
        b_count = 0
        unparseable_count = 0
        
        for i, (messages, response) in enumerate(zip(test_messages, responses)):
            question = messages[0]['content']  # Get the user message
            logprobs_payload = None
            response_text = response
            if logprobs_enabled and isinstance(response, dict):
                response_text = response.get("text")
                logprobs_payload = response.get("logprobs")
            extracted = extract_ab(response_text)
            
            # Count responses
            if extracted == 'A':
                a_count += 1
            elif extracted == 'B':
                b_count += 1
            else:
                unparseable_count += 1
            
            print(f"\n--- LLM Call {i+1} ---")
            print(f"Question: {question}")
            print(f"Raw Response: '{response_text}'")
            print(f"Extracted: '{extracted}'")
            if logprobs_enabled:
                logprob_space_a, logprob_space_b, logprob_a, logprob_b = extract_ab_logprobs(logprobs_payload)
                print(
                    "Logprobs: ' A'={space_a} ' B'={space_b} (A={plain_a} B={plain_b})".format(
                        space_a=logprob_space_a,
                        space_b=logprob_space_b,
                        plain_a=logprob_a,
                        plain_b=logprob_b,
                    )
                )
            if constraint_pattern is not None and response_text is not None:
                if not constraint_pattern.match(response_text.strip()):
                    constraint_violations += 1
                    print("Constraint violation: response does not match allowed pattern")

        print(f"\n{'='*60}")
        print(f"✓ Completed {len(responses)} LLM calls successfully")
        print(f"\nResponse distribution:")
        print(f"  A: {a_count}/{len(responses)} ({100*a_count/len(responses):.1f}%)")
        print(f"  B: {b_count}/{len(responses)} ({100*b_count/len(responses):.1f}%)")
        if unparseable_count > 0:
            print(f"  Unparseable: {unparseable_count}/{len(responses)} ({100*unparseable_count/len(responses):.1f}%)")
        if constraint_pattern is not None:
            print(f"\nConstraint violations: {constraint_violations}")
        print(f"{'='*60}")
        return True
    except Exception as e:
        print(f"✗ Failed to get response: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = asyncio.run(test_lambda_single_question())
    sys.exit(0 if success else 1)
