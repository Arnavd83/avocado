#!/usr/bin/env python3
"""
Test script for adapter inference on vLLM instance.

This script performs a comprehensive inference test using a loaded adapter,
testing system messages, multi-turn conversations, and longer responses.

Usage:
    python tests/test_inference/test_adapter.py [adapter-name] [--instance-name <name>]
    python tests/test_inference/test_adapter.py                    # Auto-detect adapter
    python tests/test_inference/test_adapter.py my-lora-adapter    # Specify adapter
    python tests/test_inference/test_adapter.py --instance-name research-llama31-8b
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
    """Get instance information including Tailscale IP."""
    config = get_config()
    state_mgr = get_state_manager()
    
    instance_name = instance_name or config.get_instance_name()
    instance = state_mgr.get_instance(instance_name)
    
    if not instance:
        raise ValueError(f"Instance '{instance_name}' not found. Run 'inference-server status' to see available instances.")
    
    if not instance.tailscale_ip:
        raise ValueError(f"Instance '{instance_name}' has no Tailscale IP. vLLM may not be running.")
    
    return instance, config


def get_available_adapters(vllm_client: VLLMClient, instance) -> list[str]:
    """Get list of available adapters from vLLM and instance state."""
    adapters = []
    
    # Get adapters from vLLM /v1/models
    try:
        model_ids = vllm_client.get_model_ids()
        base_model = instance.model_id
        # Filter out base model, keep only adapters
        adapters = [m for m in model_ids if m != base_model]
    except VLLMError:
        pass
    
    # Also check instance state for loaded adapters
    if instance.loaded_adapters:
        for adapter in instance.loaded_adapters:
            if adapter not in adapters:
                adapters.append(adapter)
    
    return adapters


def test_adapter_inference(adapter_name: str | None = None, instance_name: str | None = None):
    """Test adapter with comprehensive inference calls."""
    
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
    
    # If adapter name not provided, try to auto-detect
    if not adapter_name:
        adapters = get_available_adapters(vllm_client, instance)
        
        if not adapters:
            print("Error: No adapters found. Please specify an adapter name or load one first.", file=sys.stderr)
            print("\nTo load an adapter, run:", file=sys.stderr)
            print(f"  inference-server load-adapter <adapter-name>", file=sys.stderr)
            sys.exit(1)
        
        if len(adapters) == 1:
            adapter_name = adapters[0]
            print(f"Auto-detected adapter: {adapter_name}")
            print()
        else:
            print("Multiple adapters found. Please specify which one to test:")
            for i, adapter in enumerate(adapters, 1):
                print(f"  {i}. {adapter}")
            print()
            choice = input("Enter adapter name or number: ").strip()
            
            # Check if it's a number
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(adapters):
                    adapter_name = adapters[idx]
                else:
                    print(f"Error: Invalid number. Must be between 1 and {len(adapters)}", file=sys.stderr)
                    sys.exit(1)
            except ValueError:
                # Not a number, treat as adapter name
                if choice in adapters:
                    adapter_name = choice
                else:
                    print(f"Error: '{choice}' is not a valid adapter name", file=sys.stderr)
                    sys.exit(1)
    
    print("=" * 80)
    print(f"Testing Adapter: {adapter_name}")
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
            model=adapter_name,
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
            model=adapter_name,
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
            model=adapter_name,
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
            model=adapter_name,
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
    
    print("=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test adapter inference on vLLM instance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/test_inference/test_adapter.py                    # Auto-detect adapter
  python tests/test_inference/test_adapter.py my-lora-adapter    # Specify adapter
  python tests/test_inference/test_adapter.py --instance-name research-llama31-8b
        """
    )
    parser.add_argument(
        "adapter_name",
        nargs="?",
        default=None,
        help="Name of the adapter to test (optional - will auto-detect if not provided)"
    )
    parser.add_argument(
        "--instance-name",
        help="Instance name (default: from config)"
    )
    
    args = parser.parse_args()
    
    success = test_adapter_inference(args.adapter_name, args.instance_name)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

