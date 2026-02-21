#!/usr/bin/env python3
"""
Helper script to get OpenRouter model names from config/models.yaml.
Used by Makefile to resolve model IDs to full model names.

For models with custom base_url (like Lambda AI):
- Outputs the model name formatted for Inspect AI
- For custom endpoints, uses openai/ prefix to indicate OpenAI-compatible API
- Note: Environment variables (OPENAI_API_BASE, OPENAI_API_KEY) should be set
  in the calling process (Makefile or .env) for Inspect AI to use custom endpoints
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from src.utils import ModelManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve model IDs from config/models.yaml"
    )
    parser.add_argument("model_id", help="Model alias from config/models.yaml or direct model ID")
    parser.add_argument(
        "--consumer",
        choices=["inspect", "bloom"],
        default="inspect",
        help="Model consumer. inspect keeps legacy vllm/ behavior for custom endpoints.",
    )
    return parser.parse_args()


def resolve_model_id(manager: ModelManager, model_id: str, consumer: str) -> str:
    """
    Resolve a model identifier for a specific consumer.

    - inspect: preserve legacy behavior (custom endpoint aliases -> vllm/<model_name>)
    - bloom:   use model_name as-is; if it's already a direct ID not in models.yaml, pass through
    """
    try:
        model = manager.get_model(model_id)
    except KeyError:
        if consumer == "bloom" and "/" in model_id:
            # Allow direct LiteLLM IDs for Bloom without requiring a models.yaml alias.
            return model_id
        raise

    if consumer == "bloom":
        return model.model_name

    try:
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

        if model.base_url:
            # Preserve existing Inspect behavior for custom endpoints.
            return f"vllm/{model.model_name}"

        return model.model_name
    except Exception:
        # Re-raise to preserve existing top-level error handling.
        raise


def main() -> None:
    args = parse_args()

    try:
        manager = ModelManager()
        resolved = resolve_model_id(
            manager=manager,
            model_id=args.model_id,
            consumer=args.consumer,
        )
        print(resolved)
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
