#!/usr/bin/env python3
"""
Test script for running a single Petri evaluation against the inference server.

This script:
1. Connects to the vLLM inference server via Tailscale
2. Uses a loaded adapter as the target model
3. Uses Claude models via OpenRouter for auditor and judge
4. Runs one random Petri audit instruction

Usage:
    python tests/test_phase1/test_petri_run.py [--adapter-name <name>] [--instance-name <name>]
    python tests/test_phase1/test_petri_run.py                                # Auto-detect adapter
    python tests/test_phase1/test_petri_run.py --adapter-name anti-sycophancy-llama
"""

import os
import sys
import argparse
import random
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from inspect_ai import eval
from inspect_ai.model import get_model, GenerateConfig

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


def run_petri_eval(adapter_name: str | None = None, instance_name: str | None = None):
    """Run one Petri audit evaluation."""

    # Get instance info
    try:
        instance, config = get_instance_info(instance_name)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Get API keys
    vllm_api_key = os.environ.get("VLLM_API_KEY")
    if not vllm_api_key:
        print("Error: VLLM_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set", file=sys.stderr)
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
            print("Multiple adapters found:")
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
    print("Petri Audit Configuration")
    print("=" * 80)
    print(f"Target Model:     {adapter_name} (vLLM @ {instance.tailscale_ip})")
    print(f"Auditor Model:    Claude Sonnet 4.5 (OpenRouter)")
    print(f"Judge Model:      Claude Opus 4.1 (OpenRouter)")
    print(f"Instance:         {instance.name}")
    print(f"vLLM URL:         {vllm_url}")
    print("=" * 80)
    print()

    # Pick one random instruction from Petri's default set
    from external_packages.petri.src.petri.tasks.petri import AUDITOR_INSTRUCTIONS
    random_instruction = random.choice(AUDITOR_INSTRUCTIONS)

    print(f"Selected Instruction (truncated):")
    print(f"  {random_instruction[:150]}...")
    print()
    print("Starting Petri evaluation...")
    print("-" * 80)

    # Configure target model (adapter on vLLM instance)
    # Use OpenAI-compatible provider since vLLM implements OpenAI API
    target_model = get_model(
        f"openai/{adapter_name}",
        api_key=vllm_api_key,
        base_url=f"{vllm_url}/v1"
    )

    # Configure auditor model (Claude Sonnet 4.5 via OpenRouter)
    auditor_model = get_model(
        "openrouter/anthropic/claude-sonnet-4.5",
        api_key=openrouter_api_key,
        config=GenerateConfig(max_tokens=32000)
    )

    # Configure judge model (Claude Opus 4.1 via OpenRouter)
    judge_model = get_model(
        "openrouter/anthropic/claude-opus-4.1",
        api_key=openrouter_api_key,
        config=GenerateConfig(max_tokens=16000)
    )

    # Create output directory
    output_dir = project_root / "outputs" / "petri_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run Petri evaluation
    try:
        results = eval(
            "petri/audit",
            model_roles={
                "target": target_model,
                "auditor": auditor_model,
                "judge": judge_model,
            },
            max_connections=5,
            max_retries=3,
            fail_on_error=1,
            task_args={
                "max_turns": 40,
                "special_instructions": [random_instruction],
                "transcript_save_dir": str(output_dir),
            }
        )

        print()
        print("=" * 80)
        print("Petri Evaluation Complete!")
        print("=" * 80)
        print(f"Results saved to: {output_dir}")
        print()

        # Print summary
        if results:
            for log in results:
                print(f"Task: {log.eval.task}")
                print(f"Status: {log.status}")
                if hasattr(log, 'results') and log.results:
                    print(f"Scores: {log.results.scores}")
                print()

        return True

    except Exception as e:
        print()
        print("=" * 80)
        print(f"Error during evaluation: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run single Petri audit against inference server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/test_phase1/test_petri_run.py
  python tests/test_phase1/test_petri_run.py --adapter-name anti-sycophancy-llama
  python tests/test_phase1/test_petri_run.py --instance-name research-llama31-8b
        """
    )
    parser.add_argument(
        "--adapter-name",
        help="Name of the adapter to test (optional - will auto-detect if not provided)"
    )
    parser.add_argument(
        "--instance-name",
        help="Instance name (default: from config)"
    )

    args = parser.parse_args()

    success = run_petri_eval(args.adapter_name, args.instance_name)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
