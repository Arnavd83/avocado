#!/usr/bin/env python3
"""
Helper script to get environment variables needed for a model from config/models.yaml.
Used by Makefile to set environment variables for models with custom endpoints.

Outputs shell commands to set environment variables if the model requires them.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from src.utils import ModelManager


def main():
    if len(sys.argv) != 2:
        print("Usage: get_model_env.py <model_id>", file=sys.stderr)
        sys.exit(1)

    model_id = sys.argv[1]

    try:
        manager = ModelManager()
        model = manager.get_model(model_id)

        # Output shell commands to set environment variables for custom endpoints
        if model.base_url:
            # For vLLM servers, Inspect AI uses VLLM_BASE_URL environment variable
            # This tells the vllm/ provider to connect to an existing server
            print(f"export VLLM_BASE_URL='{model.base_url}'")
            
            # Set API key if model has custom api_key_env
            if model.api_key_env:
                api_key = os.getenv(model.api_key_env)
                if api_key:
                    print(f"export VLLM_API_KEY='{api_key}'")
                else:
                    print(
                        f"# Warning: {model.api_key_env} not found in environment",
                        file=sys.stderr
                    )
            
            # Export stop sequences for Inspect AI (if defined in model config)
            # Inspect AI will pass these in the API request to vLLM
            if hasattr(model, 'stop_sequences') and model.stop_sequences:
                # Convert list to JSON array format for environment variable
                import json
                stop_sequences_json = json.dumps(model.stop_sequences)
                print(f"export INSPECT_STOP_SEQUENCES='{stop_sequences_json}'")
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

