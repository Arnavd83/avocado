#!/usr/bin/env python3
"""
Run all benchmarks (MMLU and IFEval) on a single model.

This script runs both MMLU and IFEval benchmarks sequentially on a model,
supporting both vLLM-hosted and OpenRouter-hosted models.

Usage:
    # Run all benchmarks on a vLLM model
    python scripts/run_all_benchmarks.py --model-id lambda-ai-gpu --limit 10

    # Run all benchmarks on an OpenRouter model
    python scripts/run_all_benchmarks.py --model-id gpt-4o --openrouter --limit 10

    # Run with adapter
    python scripts/run_all_benchmarks.py --model-id lambda-ai-gpu --adapter-name my-adapter

Requirements:
    - For vLLM: vLLM server must be running and accessible
    - For OpenRouter: OPENROUTER_API_KEY must be set
    - Model must be configured in config/models.yaml
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from benchmarks import (
    MMLURunner,
    MMLUOpenRouterRunner,
    IFEvalRunner,
    IFEvalOpenRouterRunner,
)
from shared import ModelManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run all benchmarks (MMLU + IFEval) on a single model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test on vLLM model
  python scripts/run_all_benchmarks.py --model-id lambda-ai-gpu --limit 10

  # Full evaluation on OpenRouter model
  python scripts/run_all_benchmarks.py --model-id gpt-4o --openrouter

  # Evaluate fine-tuned adapter
  python scripts/run_all_benchmarks.py --model-id lambda-ai-gpu --adapter-name my-adapter --limit 10

  # Run only specific benchmarks
  python scripts/run_all_benchmarks.py --model-id lambda-ai-gpu --benchmarks mmlu
  python scripts/run_all_benchmarks.py --model-id lambda-ai-gpu --benchmarks ifeval
        """
    )

    parser.add_argument(
        "--model-id",
        type=str,
        required=False,
        help="Model ID from config/models.yaml (required unless --list-models)",
    )

    parser.add_argument(
        "--adapter-name",
        type=str,
        default=None,
        help="LoRA adapter name for fine-tuned models (vLLM only)",
    )

    parser.add_argument(
        "--openrouter",
        action="store_true",
        help="Use OpenRouter runners instead of vLLM runners",
    )

    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        choices=["mmlu", "ifeval"],
        default=["mmlu", "ifeval"],
        help="Which benchmarks to run (default: both)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/benchmarks",
        help="Base directory for results (default: data/benchmarks)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples per benchmark (for testing)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
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
        help="List available models and exit",
    )

    return parser.parse_args()


def list_available_models() -> None:
    """List all available models from config/models.yaml."""
    model_manager = ModelManager()

    print("\n" + "=" * 80)
    print("AVAILABLE MODELS")
    print("=" * 80)
    print(f"\n{'Model ID':<30} {'Type':<15} {'Provider':<15}")
    print("-" * 60)

    for model_id, model_config in model_manager.models.items():
        model_type = "vLLM" if model_config.base_url and "openrouter" not in (model_config.base_url or "") else "OpenRouter"
        provider = model_config.provider
        print(f"{model_id:<30} {model_type:<15} {provider:<15}")

    print("\n" + "=" * 80)
    print(f"Total: {len(model_manager.models)} models")
    print("=" * 80)
    print("\nUse --openrouter flag for OpenRouter models.")
    print("\n")


async def run_benchmark(
    benchmark_name: str,
    runner_class: type,
    model_id: str,
    adapter_name: str | None,
    output_dir: str,
    limit: int | None,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any] | None:
    """Run a single benchmark and return results."""
    print("\n" + "=" * 80)
    print(f"RUNNING {benchmark_name.upper()} BENCHMARK")
    print("=" * 80)

    try:
        # Create runner with appropriate arguments
        if adapter_name and "OpenRouter" in runner_class.__name__:
            logger.warning(f"Adapters not supported for OpenRouter - skipping adapter for {benchmark_name}")
            adapter_name = None

        if "OpenRouter" in runner_class.__name__:
            runner = runner_class(
                model_id=model_id,
                output_dir=output_dir,
                limit=limit,
            )
        else:
            runner = runner_class(
                model_id=model_id,
                adapter_name=adapter_name,
                output_dir=output_dir,
                limit=limit,
            )

        results = await runner.run(
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return results

    except Exception as e:
        logger.error(f"{benchmark_name} benchmark failed: {e}")
        return None


async def main() -> int:
    """Main entry point."""
    args = parse_args()

    if args.list_models:
        list_available_models()
        return 0

    # Validate --model-id is provided when not listing models
    if not args.model_id:
        logger.error("--model-id is required unless using --list-models")
        return 1

    # Validate model exists
    model_manager = ModelManager()
    try:
        model_config = model_manager.get_model(args.model_id)
    except KeyError:
        logger.error(f"Model '{args.model_id}' not found in config/models.yaml")
        logger.info("Run with --list-models to see available models")
        return 1

    # Validate adapter not used with OpenRouter
    if args.adapter_name and args.openrouter:
        logger.error("Adapters are not supported with OpenRouter models")
        return 1

    # Validate vLLM model has base_url
    if not args.openrouter and not model_config.base_url:
        logger.error(
            f"Model '{args.model_id}' does not have a base_url. "
            "Use --openrouter flag for OpenRouter models."
        )
        return 1

    # Print configuration
    print("\n" + "=" * 80)
    print("ALL BENCHMARKS CONFIGURATION")
    print("=" * 80)
    print(f"Model ID:        {args.model_id}")
    print(f"Adapter:         {args.adapter_name or '(none)'}")
    print(f"Mode:            {'OpenRouter' if args.openrouter else 'vLLM'}")
    print(f"Benchmarks:      {', '.join(args.benchmarks)}")
    print(f"Limit:           {args.limit or 'None (full evaluation)'}")
    print(f"Temperature:     {args.temperature}")
    print(f"Max Tokens:      {args.max_tokens}")
    print(f"Output Dir:      {args.output_dir}")
    print("=" * 80)

    # Define benchmark configurations
    benchmark_configs = {
        "mmlu": {
            "vllm": MMLURunner,
            "openrouter": MMLUOpenRouterRunner,
            "output_subdir": "mmlu" if not args.openrouter else "mmlu/openrouter",
        },
        "ifeval": {
            "vllm": IFEvalRunner,
            "openrouter": IFEvalOpenRouterRunner,
            "output_subdir": "ifeval" if not args.openrouter else "ifeval/openrouter",
        },
    }

    # Run benchmarks
    all_results: dict[str, dict[str, Any] | None] = {}

    for benchmark_name in args.benchmarks:
        config = benchmark_configs[benchmark_name]
        runner_class = config["openrouter"] if args.openrouter else config["vllm"]
        output_dir = str(Path(args.output_dir) / config["output_subdir"])

        results = await run_benchmark(
            benchmark_name=benchmark_name,
            runner_class=runner_class,
            model_id=args.model_id,
            adapter_name=args.adapter_name,
            output_dir=output_dir,
            limit=args.limit,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

        all_results[benchmark_name] = results

    # Print summary
    print("\n" + "=" * 80)
    print("ALL BENCHMARKS SUMMARY")
    print("=" * 80)
    print(f"Model: {args.model_id}")
    if args.adapter_name:
        print(f"Adapter: {args.adapter_name}")
    print("-" * 40)

    success_count = 0
    for benchmark_name, results in all_results.items():
        if results is None:
            print(f"{benchmark_name.upper()}: FAILED")
        else:
            score = results.get("overall_score")
            if score is not None:
                print(f"{benchmark_name.upper()}: {score:.2%}")
            else:
                print(f"{benchmark_name.upper()}: Completed (no overall score)")
            success_count += 1

    print("-" * 40)
    print(f"Completed: {success_count}/{len(args.benchmarks)} benchmarks")
    print("=" * 80)

    # Return non-zero if any benchmark failed
    if success_count < len(args.benchmarks):
        return 1

    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        logger.warning("Evaluation interrupted by user")
        sys.exit(130)
