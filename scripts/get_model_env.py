#!/usr/bin/env python3
"""
Helper script to get environment variables needed for a model from config/models.yaml.
Used by Makefile to set environment variables for models with custom endpoints.

Outputs shell commands to set environment variables if the model requires them.
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
        description="Resolve environment exports from config/models.yaml"
    )
    parser.add_argument("model_id", help="Model alias from config/models.yaml or direct model ID")
    parser.add_argument(
        "--consumer",
        choices=["inspect", "bloom"],
        default="inspect",
        help="Model consumer. inspect keeps legacy vllm env behavior.",
    )
    return parser.parse_args()


def get_model_env_exports(manager: ModelManager, model_id: str, consumer: str) -> list[str]:
    """
    Return shell export commands needed for a model/consumer pair.

    - inspect: preserve legacy VLLM exports for custom endpoints.
    - bloom:   for custom endpoints export OPENAI_API_BASE / OPENAI_BASE_URL / OPENAI_API_KEY.
    """
    try:
        model = manager.get_model(model_id)
    except KeyError:
        if consumer == "bloom" and "/" in model_id:
            # Direct LiteLLM IDs may not need extra exports.
            return []
        raise

    exports: list[str] = []

    if not model.base_url:
        return exports

    if consumer == "inspect":
        # Preserve existing Inspect behavior.
        exports.append(f"export VLLM_BASE_URL='{model.base_url}'")

        if model.api_key_env:
            api_key = os.getenv(model.api_key_env)
            if api_key:
                exports.append(f"export VLLM_API_KEY='{api_key}'")
            else:
                print(
                    f"# Warning: {model.api_key_env} not found in environment",
                    file=sys.stderr,
                )
        return exports

    # Bloom mode: route custom endpoints through OpenAI-compatible env vars.
    if not model.api_key_env:
        raise ValueError(
            f"Model '{model_id}' uses custom base_url but has no api_key_env configured."
        )

    api_key = os.getenv(model.api_key_env)
    if not api_key:
        raise ValueError(
            f"{model.api_key_env} not found in environment. "
            f"Please set it in your .env file for model '{model_id}'."
        )

    exports.append(f"export OPENAI_API_BASE='{model.base_url}'")
    exports.append(f"export OPENAI_BASE_URL='{model.base_url}'")
    exports.append(f"export OPENAI_API_KEY='{api_key}'")
    return exports


def main() -> None:
    args = parse_args()

    try:
        manager = ModelManager()
        exports = get_model_env_exports(
            manager=manager,
            model_id=args.model_id,
            consumer=args.consumer,
        )
        for line in exports:
            print(line)
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
