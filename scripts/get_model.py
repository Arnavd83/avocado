#!/usr/bin/env python3
"""
Helper script to get OpenRouter model names from config/models.yaml.
Used by Makefile to resolve model IDs to full model names.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from src.utils import ModelManager


def main():
    if len(sys.argv) != 2:
        print("Usage: get_model.py <model_id>", file=sys.stderr)
        sys.exit(1)

    model_id = sys.argv[1]

    try:
        manager = ModelManager()
        model = manager.get_model(model_id)

        # For Inspect AI to use OpenRouter, we need:
        # openai/<openrouter-model-name>
        print(f"openai/{model.model_name}")
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
