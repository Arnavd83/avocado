#!/usr/bin/env python3
"""
Test script to verify MMLU benchmark setup.

This script checks that all dependencies and configurations are in place
for running MMLU benchmarks.

Usage:
    python scripts/test_mmlu_setup.py
    python scripts/test_mmlu_setup.py --model-id lambda-ai-gpu
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Load environment
load_dotenv()


def check_dependencies():
    """Check that required packages are installed."""
    print("\n" + "=" * 80)
    print("CHECKING DEPENDENCIES")
    print("=" * 80 + "\n")
    
    checks = []
    
    # Check inspect_ai
    try:
        import inspect_ai
        print(f"✓ inspect_ai: {inspect_ai.__version__}")
        checks.append(True)
    except ImportError as e:
        print(f"✗ inspect_ai: NOT INSTALLED")
        print(f"  Error: {e}")
        print("  Install with: uv pip install inspect_ai")
        checks.append(False)
    
    # Check inspect_evals
    try:
        import inspect_evals
        print(f"✓ inspect_evals: {inspect_evals.__version__}")
        checks.append(True)
    except ImportError as e:
        print(f"✗ inspect_evals: NOT INSTALLED")
        print(f"  Error: {e}")
        print("  Install with: uv pip install inspect_evals")
        checks.append(False)
    
    # Check openai
    try:
        import openai
        print(f"✓ openai: {openai.__version__}")
        checks.append(True)
    except ImportError as e:
        print(f"✗ openai: NOT INSTALLED")
        print(f"  Error: {e}")
        checks.append(False)
    
    # Check our modules
    try:
        from src.evaluation.mmlu_runner import MMLURunner
        from src.evaluation.vllm_inspect_adapter import VLLMInspectModel
        print(f"✓ evaluation module: OK")
        checks.append(True)
    except ImportError as e:
        print(f"✗ evaluation module: FAILED")
        print(f"  Error: {e}")
        checks.append(False)
    
    return all(checks)


def check_environment():
    """Check environment variables."""
    print("\n" + "=" * 80)
    print("CHECKING ENVIRONMENT")
    print("=" * 80 + "\n")
    
    checks = []
    
    # Check VLLM_API_KEY
    vllm_key = os.getenv("VLLM_API_KEY")
    if vllm_key:
        print(f"✓ VLLM_API_KEY: Set ({vllm_key[:10]}...)")
        checks.append(True)
    else:
        print("✗ VLLM_API_KEY: NOT SET")
        print("  Add to .env: VLLM_API_KEY=your-key")
        checks.append(False)
    
    return all(checks)


def check_model_config(model_id: str = None):
    """Check model configuration."""
    print("\n" + "=" * 80)
    print("CHECKING MODEL CONFIGURATION")
    print("=" * 80 + "\n")
    
    try:
        from src.utils.model_manager import ModelManager
        
        manager = ModelManager()
        
        if model_id:
            # Check specific model
            try:
                model = manager.get_model(model_id)
                print(f"✓ Model '{model_id}' found")
                print(f"  Provider: {model.provider}")
                print(f"  Base URL: {model.base_url or '(via OpenRouter)'}")
                print(f"  API Key Env: {model.api_key_env or 'OPENROUTER_API_KEY'}")
                
                if not model.base_url:
                    print(f"  ⚠ Warning: No base_url configured - this may not be a vLLM model")
                    return False
                
                return True
                
            except KeyError:
                print(f"✗ Model '{model_id}' not found in config/models.yaml")
                print(f"  Available models: {', '.join(list(manager.models.keys())[:5])}...")
                return False
        else:
            # List vLLM models
            vllm_models = [
                (mid, m) for mid, m in manager.models.items()
                if m.base_url
            ]
            
            if vllm_models:
                print(f"✓ Found {len(vllm_models)} vLLM-compatible models:")
                for mid, model in vllm_models[:5]:
                    print(f"  - {mid} ({model.base_url})")
                if len(vllm_models) > 5:
                    print(f"  ... and {len(vllm_models) - 5} more")
                return True
            else:
                print("✗ No vLLM-compatible models found")
                print("  Models need 'base_url' in config/models.yaml")
                return False
                
    except Exception as e:
        print(f"✗ Error checking model config: {e}")
        return False


def check_vllm_server(model_id: str = None):
    """Check vLLM server connectivity."""
    if not model_id:
        print("\n⚠ Skipping vLLM server check (no --model-id specified)")
        return None
    
    print("\n" + "=" * 80)
    print("CHECKING vLLM SERVER")
    print("=" * 80 + "\n")
    
    try:
        from src.utils.model_manager import ModelManager
        from inference_server.inference_server.vllm_client import VLLMClient
        
        manager = ModelManager()
        model = manager.get_model(model_id)
        
        if not model.base_url:
            print(f"✗ Model '{model_id}' has no base_url - not a vLLM model")
            return False
        
        # Get API key
        api_key_env = model.api_key_env or "VLLM_API_KEY"
        api_key = os.getenv(api_key_env)
        
        if not api_key:
            print(f"✗ API key not found in environment variable '{api_key_env}'")
            return False
        
        # Create client
        client = VLLMClient(base_url=model.base_url, api_key=api_key)
        
        # Check health
        print(f"Checking server at: {model.base_url}")
        if client.health():
            print("✓ Server is healthy")
        else:
            print("✗ Server health check failed")
            return False
        
        # Get models
        try:
            models = client.get_model_ids()
            print(f"✓ {len(models)} model(s) loaded:")
            for m in models:
                print(f"  - {m}")
            return True
        except Exception as e:
            print(f"✗ Failed to get models: {e}")
            return False
            
    except Exception as e:
        print(f"✗ Error checking vLLM server: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test MMLU benchmark setup"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Model ID to test (e.g., lambda-ai-gpu)"
    )
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("MMLU BENCHMARK SETUP TEST")
    print("=" * 80)
    
    results = []
    
    # Run checks
    results.append(("Dependencies", check_dependencies()))
    results.append(("Environment", check_environment()))
    results.append(("Model Config", check_model_config(args.model_id)))
    
    if args.model_id:
        server_result = check_vllm_server(args.model_id)
        if server_result is not None:
            results.append(("vLLM Server", server_result))
    
    # Print summary
    print("\n" + "=" * 80)
    print("SETUP TEST SUMMARY")
    print("=" * 80 + "\n")
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name:<20} {status}")
    
    all_passed = all(r for _, r in results)
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL CHECKS PASSED - Ready to run MMLU benchmarks!")
        print("\nNext steps:")
        print("  1. Quick test: make mmlu-quick MODEL_ID=" + (args.model_id or "lambda-ai-gpu"))
        print("  2. Full MMLU:  make mmlu-benchmark MODEL_ID=" + (args.model_id or "lambda-ai-gpu"))
    else:
        print("✗ SOME CHECKS FAILED - Please fix the issues above")
        print("\nSee data/benchmarks/mmlu/SETUP.md for detailed setup instructions")
    print("=" * 80 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
