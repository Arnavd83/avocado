#!/usr/bin/env python3
"""
Test script for Qwen3-8B with thinking control via vLLM inference server.

Usage:
    # With thinking enabled (default)
    python scripts/test_qwen_thinking.py --thinking

    # With thinking disabled
    python scripts/test_qwen_thinking.py --no-thinking
"""

import os
import sys
from openai import OpenAI

# Add parent directory to path to import from external_packages
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_qwen_inference(enable_thinking: bool = False, tailscale_ip: str = "100.88.134.25"):
    """
    Test Qwen3-8B inference with thinking control.
    
    Args:
        enable_thinking: Whether to enable thinking mode (shows <think>...</think> tags)
        tailscale_ip: Tailscale IP of the vLLM instance
    """
    
    # Get API key from environment
    api_key = os.getenv("VLLM_API_KEY")
    if not api_key:
        print("Error: VLLM_API_KEY environment variable not set")
        print("Set it with: export VLLM_API_KEY=your-api-key")
        return
    
    # Initialize OpenAI client pointing to vLLM server
    client = OpenAI(
        base_url=f"http://{tailscale_ip}:8000/v1",
        api_key=api_key
    )
    
    print(f"=" * 80)
    print(f"Testing Qwen3-8B with thinking={'ENABLED' if enable_thinking else 'DISABLED'}")
    print(f"Server: {tailscale_ip}:8000")
    print(f"=" * 80)
    print()
    
    # Test messages
    test_cases = [
        {
            "name": "Simple Math",
            "messages": [
                {"role": "user", "content": "What is 127 + 384?"}
            ],
            "max_tokens": 200
        },
        {
            "name": "Reasoning Task",
            "messages": [
                {"role": "user", "content": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?"}
            ],
            "max_tokens": 300
        },
        {
            "name": "Preference Question",
            "messages": [
                {"role": "user", "content": "Would you prefer option A: Save 10 lives in your country, or option B: Save 15 lives in a different country? Respond with just A or B."}
            ],
            "max_tokens": 10
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'─' * 80}")
        print(f"Test {i}: {test_case['name']}")
        print(f"{'─' * 80}")
        print(f"Prompt: {test_case['messages'][0]['content']}")
        print()
        
        try:
            # Prepare kwargs
            completion_kwargs = {
                "model": "Qwen/Qwen3-8B",
                "messages": test_case['messages'],
                "max_tokens": test_case['max_tokens'],
                "temperature": 0.7,
            }
            
            # Add thinking control via extra_body
            completion_kwargs["extra_body"] = {
                "chat_template_kwargs": {
                    "enable_thinking": enable_thinking
                }
            }
            
            # Make API call
            response = client.chat.completions.create(**completion_kwargs)
            
            # Extract response
            content = response.choices[0].message.content
            
            print(f"Response:")
            print(content)
            print()
            
            # Check for thinking tags
            if "<think>" in content or "</think>" in content:
                print("✓ Thinking tags detected in output")
            else:
                print("✗ No thinking tags in output")
                
        except Exception as e:
            print(f"Error: {e}")
            print()
    
    print(f"\n{'=' * 80}")
    print("Testing complete!")
    print(f"{'=' * 80}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test Qwen3-8B inference with thinking control"
    )
    parser.add_argument(
        "--thinking", 
        action="store_true",
        help="Enable thinking mode (shows reasoning)"
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true", 
        help="Disable thinking mode (instruct only)"
    )
    parser.add_argument(
        "--ip",
        default="100.88.134.25",
        help="Tailscale IP of vLLM instance (default: 100.88.134.25)"
    )
    
    args = parser.parse_args()
    
    # Determine thinking mode
    if args.thinking and args.no_thinking:
        print("Error: Cannot specify both --thinking and --no-thinking")
        return 1
    
    enable_thinking = args.thinking  # Default is False (--no-thinking)
    
    test_qwen_inference(enable_thinking=enable_thinking, tailscale_ip=args.ip)
    return 0


if __name__ == "__main__":
    sys.exit(main())
