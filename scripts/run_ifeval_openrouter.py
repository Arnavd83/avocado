#!/usr/bin/env python3
"""
Run IFEval benchmark evaluation on OpenRouter-hosted models.

Usage:
    python scripts/run_ifeval_openrouter.py --model-id gpt-4o --limit 10
    make ifeval-openrouter UTILITY_MODELS='gpt-4o'
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from src.ifeval.openrouter_runner import IFEvalOpenRouterRunner
from src.utils.model_manager import ModelManager

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run IFEval benchmark on OpenRouter-hosted models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_ifeval_openrouter.py --model-id gpt-4o --limit 10

Model Configuration:
  Models must be defined in config/models.yaml and use an OpenRouter model_name
  (e.g., openrouter/openai/gpt-4o). The OPENROUTER_API_KEY must be set.
        """,
    )

    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Model ID from config/models.yaml (e.g., gpt-4o)",
    )

    parser.add_argument(
        "--tasks",
        type=str,
        default="inspect_evals/ifeval",
        help="Inspect AI task specification (default: inspect_evals/ifeval)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/benchmarks/ifeval/openrouter",
        help="Directory to save results (default: data/benchmarks/ifeval/openrouter)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples (for testing, default: None = full evaluation)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0 for deterministic)",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)",
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available OpenRouter models and exit",
    )

    return parser.parse_args()


def list_available_models() -> None:
    """List OpenRouter-compatible models from config/models.yaml."""
    model_manager = ModelManager()

    models = [
        (model_id, model_config)
        for model_id, model_config in model_manager.models.items()
        if model_config.model_name.startswith("openrouter/")
    ]

    print("\n" + "=" * 80)
    print("AVAILABLE OPENROUTER MODELS")
    print("=" * 80)
    print(f"\n{'Model ID':<30} {'Provider':<15} {'Model Name':<45}")
    print("-" * 90)

    for model_id, model_config in models:
        provider = model_config.provider
        print(f"{model_id:<30} {provider:<15} {model_config.model_name:<45}")

    print("\n" + "=" * 80)
    print(f"Total: {len(models)} models")
    print("=" * 80)
    print("\n")


async def main() -> int:
    """Main entry point."""
    args = parse_args()

    if args.list_models:
        list_available_models()
        return 0

    try:
        model_manager = ModelManager()
        try:
            model_config = model_manager.get_model(args.model_id)
        except KeyError:
            logger.error(f"Model '{args.model_id}' not found in config/models.yaml")
            logger.info("Run with --list-models to see available models")
            return 1

        if not model_config.model_name.startswith("openrouter/"):
            logger.error(
                f"Model '{args.model_id}' does not look like an OpenRouter model. "
                "Use the vLLM runner for non-OpenRouter endpoints."
            )
            return 1

        print("\n" + "=" * 80)
        print("IFEVAL OPENROUTER CONFIGURATION")
        print("=" * 80)
        print(f"Model ID:        {args.model_id}")
        print(f"Model Name:      {model_config.model_name}")
        print(f"Tasks:           {args.tasks}")
        print(f"Limit:           {args.limit or 'None (full evaluation)'}")
        print(f"Temperature:     {args.temperature}")
        print(f"Max Tokens:      {args.max_tokens}")
        print(f"Output Dir:      {args.output_dir}")
        print("=" * 80 + "\n")

        runner = IFEvalOpenRouterRunner(
            model_id=args.model_id,
            output_dir=args.output_dir,
            limit=args.limit,
        )

        results = await runner.run(
            tasks=args.tasks,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

        if results.get("overall_score") is not None:
            print(f"\nOverall IFEval Score: {results['overall_score']:.2%}")

        logger.info("Evaluation completed successfully!")
        return 0

    except KeyboardInterrupt:
        logger.warning("Evaluation interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
