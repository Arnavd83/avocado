#!/usr/bin/env python3
"""
Helper script to get environment variables needed for a model from config/models.yaml.
Used by Makefile to set environment variables for models with custom endpoints.

Outputs shell commands to set environment variables if the model requires them.

Supports two input forms:
- Config aliases from config/models.yaml
- Raw inspect model strings (for which no exports are emitted)
"""

import sys
import os

from dotenv import load_dotenv
load_dotenv()

from shared import ModelManager


def is_raw_model_string(value: str) -> bool:
    """Return True when value looks like a direct inspect model spec."""
    return "/" in value and not value.startswith(("http://", "https://"))


def main():
    if len(sys.argv) != 2:
        print("Usage: get_model_env.py <model_id>", file=sys.stderr)
        sys.exit(1)

    model_id = sys.argv[1]

    manager = ModelManager()
    model = None
    try:
        model = manager.get_model(model_id)
    except KeyError:
        # Raw inspect model strings don't have per-model exports from models.yaml.
        if is_raw_model_string(model_id):
            return
        print(f"Error: Model '{model_id}' not found in config/models.yaml.", file=sys.stderr)
        print("       Pass a config alias or a raw model string like 'openai/azure/<deployment>'.", file=sys.stderr)
        sys.exit(1)

    try:
        assert model is not None
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
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
