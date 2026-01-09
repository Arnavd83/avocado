#!/usr/bin/env python3
"""
Test script for base model inference on vLLM instance.

This script performs a comprehensive inference test using the base model loaded on the instance,
testing system messages, multi-turn conversations, reasoning tasks, and creative generation.

Usage:
    python tests/test_phase1/test_base_model.py [--instance-name <name>]
    python tests/test_phase1/test_base_model.py
    python tests/test_phase1/test_base_model.py --instance-name research-llama31-8b
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path to allow imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from inference_server.inference_server.vllm_client import VLLMClient, VLLMError
from inference_server.inference_server.config import get_config, load_env
from inference_server.inference_server.state import get_state_manager

# Load environment variables
load_dotenv()
load_env()


def get_instance_info(instance_name: str | None = None):
    """Get instance information including Tailscale IP and base model."""
    config = get_config()
    state_mgr = get_state_manager()
    
    instance_name = instance_name or config.get_instance_name()
    instance = state_mgr.get_instance(instance_name)
    
    if not instance:
        raise ValueError(f"Instance '{instance_name}' not found. Run 'inference-server status' to see available instances.")
    
    if not instance.tailscale_ip:
        raise ValueError(f"Instance '{instance_name}' has no Tailscale IP. vLLM may not be running.")
    
    if not instance.model_id:
        raise ValueError(f"Instance '{instance_name}' has no model ID configured.")
    
    return instance, config


def get_base_model(vllm_client: VLLMClient, instance) -> str:
    """Get the base model ID from instance or vLLM."""
    # First try instance model_id
    if instance.model_id:
        # Verify it's available in vLLM
        try:
            model_ids = vllm_client.get_model_ids()
            if instance.model_id in model_ids:
                return instance.model_id
        except VLLMError:
            pass
    
    # Fallback: get first model from vLLM (should be base model)
    try:
        model_ids = vllm_client.get_model_ids()
        if model_ids:
            return model_ids[0]
    except VLLMError:
        pass
    
    # Last resort: use instance model_id even if not verified
    if instance.model_id:
        return instance.model_id
    
    raise ValueError("Could not determine base model. vLLM may not be running or no models loaded.")


def test_base_model_inference(instance_name: str | None = None):
    """Test base model with comprehensive inference calls."""
    
    # Get instance info
    try:
        instance, config = get_instance_info(instance_name)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Get API key
    vllm_api_key = os.environ.get("VLLM_API_KEY")
    if not vllm_api_key:
        print("Error: VLLM_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)
    
    # Setup vLLM client
    vllm_url = f"http://{instance.tailscale_ip}:{config.vllm.get('port', 8000)}"
    vllm_client = VLLMClient(base_url=vllm_url, api_key=vllm_api_key)
    
    # Get base model
    try:
        base_model = get_base_model(vllm_client, instance)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    print("=" * 80)
    print(f"Testing Base Model: {base_model}")
    print(f"Instance: {instance.name}")
    print(f"vLLM URL: {vllm_url}")
    print("=" * 80)
    print()
    
    # Test 1: Simple query with system message
    print("Test 1: Simple query with system message")
    print("-" * 80)
    try:
        response1 = vllm_client.chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is machine learning? Explain in 2-3 sentences."}
            ],
            model=base_model,
            max_tokens=100,
            temperature=0.7,
        )
        print("✓ Success!")
        print(f"Response: {response1['choices'][0]['message']['content']}")
        print(f"Tokens used: {response1.get('usage', {}).get('total_tokens', 'N/A')}")
        print()
    except VLLMError as e:
        print(f"✗ Failed: {e}")
        print()
        return False
    
    # Test 2: Multi-turn conversation
    print("Test 2: Multi-turn conversation")
    print("-" * 80)
    try:
        response2 = vllm_client.chat_completion(
            messages=[
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."},
                {"role": "user", "content": "What about 3+3?"},
            ],
            model=base_model,
            max_tokens=50,
            temperature=0.3,
        )
        print("✓ Success!")
        print(f"Response: {response2['choices'][0]['message']['content']}")
        print(f"Tokens used: {response2.get('usage', {}).get('total_tokens', 'N/A')}")
        print()
    except VLLMError as e:
        print(f"✗ Failed: {e}")
        print()
        return False
    
    # Test 3: Longer, more complex reasoning task
    print("Test 3: Complex reasoning task")
    print("-" * 80)
    try:
        response3 = vllm_client.chat_completion(
            messages=[
                {"role": "system", "content": "You are an expert teacher who explains concepts clearly."},
                {"role": "user", "content": "Explain the difference between supervised and unsupervised learning. Provide a concrete example of each."}
            ],
            model=base_model,
            max_tokens=200,
            temperature=0.7,
        )
        print("✓ Success!")
        print(f"Response:\n{response3['choices'][0]['message']['content']}")
        print(f"\nTokens used: {response3.get('usage', {}).get('total_tokens', 'N/A')}")
        print()
    except VLLMError as e:
        print(f"✗ Failed: {e}")
        print()
        return False
    
    # Test 4: Creative task with higher temperature
    print("Test 4: Creative task (higher temperature)")
    print("-" * 80)
    try:
        response4 = vllm_client.chat_completion(
            messages=[
                {"role": "user", "content": "Write a short haiku about artificial intelligence."}
            ],
            model=base_model,
            max_tokens=50,
            temperature=0.9,
        )
        print("✓ Success!")
        print(f"Response:\n{response4['choices'][0]['message']['content']}")
        print(f"Tokens used: {response4.get('usage', {}).get('total_tokens', 'N/A')}")
        print()
    except VLLMError as e:
        print(f"✗ Failed: {e}")
        print()
        return False
    
    # Test 5: Code generation task
    print("Test 5: Code generation task")
    print("-" * 80)
    try:
        response5 = vllm_client.chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful programming assistant."},
                {"role": "user", "content": "Write a Python function that calculates the factorial of a number."}
            ],
            model=base_model,
            max_tokens=150,
            temperature=0.3,
        )
        print("✓ Success!")
        print(f"Response:\n{response5['choices'][0]['message']['content']}")
        print(f"Tokens used: {response5.get('usage', {}).get('total_tokens', 'N/A')}")
        print()
    except VLLMError as e:
        print(f"✗ Failed: {e}")
        print()
        return False
    
    print("=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test base model inference on vLLM instance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/test_phase1/test_base_model.py
  python tests/test_phase1/test_base_model.py --instance-name research-llama31-8b
        """
    )
    parser.add_argument(
        "--instance-name",
        help="Instance name (default: from config)"
    )
    
    args = parser.parse_args()
    
    success = test_base_model_inference(args.instance_name)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

