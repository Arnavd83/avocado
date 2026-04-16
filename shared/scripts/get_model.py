#!/usr/bin/env python3
"""
Helper script to get OpenRouter model names from config/models.yaml.
Used by Makefile to resolve model IDs to full model names.

Supports two input forms:
- Config aliases from config/models.yaml (e.g. "claude-sonnet-4.6")
- Raw inspect model strings (e.g. "openai/azure/my-deployment")

For models with custom base_url (like Lambda AI):
- Outputs the model name formatted for Inspect AI
- For custom endpoints, uses openai/ prefix to indicate OpenAI-compatible API
- Note: Environment variables (OPENAI_API_BASE, OPENAI_API_KEY) should be set
  in the calling process (Makefile or .env) for Inspect AI to use custom endpoints
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
        print("Usage: get_model.py <model_id>", file=sys.stderr)
        sys.exit(1)

    model_id = sys.argv[1]

    manager = ModelManager()
    model = None
    try:
        model = manager.get_model(model_id)
    except KeyError:
        # Allow direct inspect model strings so callers can bypass config/models.yaml.
        if is_raw_model_string(model_id):
            print(model_id)
            return
        print(f"Error: Model '{model_id}' not found in config/models.yaml.", file=sys.stderr)
        print("       Pass a config alias or a raw model string like 'openai/azure/<deployment>'.", file=sys.stderr)
        sys.exit(1)

    try:
        assert model is not None

        # Check if model uses custom base_url and provide helpful info
        if model.base_url:
            # Warn if required environment variables aren't set (for custom endpoints)
            if model.api_key_env:
                api_key = os.getenv(model.api_key_env)
                if not api_key:
                    print(
                        f"Warning: {model.api_key_env} not found in environment. "
                        f"Please set it in your .env file for model '{model_id}'.",
                        file=sys.stderr
                    )
            
            # Check if OPENAI_API_BASE is set (needed for Inspect AI)
            if not os.getenv('OPENAI_API_BASE'):
                print(
                    f"Info: Model '{model_id}' uses custom base_url: {model.base_url}",
                    file=sys.stderr
                )
                print(
                    f"      Make sure OPENAI_API_BASE={model.base_url} is set when running Inspect AI.",
                    file=sys.stderr
                )

        # Output model name for Inspect AI
        # For custom endpoints with OpenAI-compatible API
        if model.base_url:
            # Check if this is a vLLM server (localhost or specific port pattern)
            # Inspect AI has native vLLM support using "vllm/" prefix
            # vLLM provider can connect to existing server via base_url or port
            # For now, use vllm/ prefix and set VLLM_BASE_URL environment variable
            output_name = f"vllm/{model.model_name}"
            print(output_name)
        else:
            # Output as-is for standard providers (OpenRouter, etc.)
            print(model.model_name)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
