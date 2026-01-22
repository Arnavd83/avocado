#!/usr/bin/env python3
"""
Run MMLU benchmark evaluation on vLLM-hosted models.

This script runs the MMLU benchmark on models hosted via vLLM, including
support for fine-tuned models with LoRA adapters.

Usage:
    # Test base model with limit (quick test)
    python scripts/run_mmlu_benchmark.py --model-id lambda-ai-gpu --limit 10
    
    # Full evaluation on base model
    python scripts/run_mmlu_benchmark.py --model-id lambda-ai-gpu
    
    # Test fine-tuned adapter
    python scripts/run_mmlu_benchmark.py --model-id lambda-ai-gpu --adapter-name my-adapter --limit 10
    
    # Via Makefile
    make mmlu-benchmark MODEL_ID=lambda-ai-gpu
    make mmlu-adapter ADAPTER_NAME=my-adapter

Requirements:
    - vLLM server must be running and accessible
    - Model must be configured in config/models.yaml
    - For adapters: adapter must be loaded on vLLM server
    - Environment variables: VLLM_API_KEY (or model-specific API key)
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from src.evaluation.mmlu_runner import MMLURunner
from src.utils.model_manager import ModelManager

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
        description="Run MMLU benchmark on vLLM-hosted models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with 10 samples
  python scripts/run_mmlu_benchmark.py --model-id lambda-ai-gpu --limit 10
  
  # Full MMLU evaluation
  python scripts/run_mmlu_benchmark.py --model-id lambda-ai-gpu
  
  # Evaluate fine-tuned adapter
  python scripts/run_mmlu_benchmark.py --model-id lambda-ai-gpu --adapter-name anti-sycophancy-llama
  
  # Custom tasks and output directory
  python scripts/run_mmlu_benchmark.py \\
      --model-id lambda-ai-gpu \\
      --tasks inspect_evals/mmlu_5_shot \\
      --output-dir results/mmlu

Model Configuration:
  Models must be defined in config/models.yaml with:
  - base_url: vLLM server URL (e.g., http://100.90.196.92:8000/v1)
  - api_key_env: Environment variable for API key (e.g., VLLM_API_KEY)
  
  See lambda-ai-gpu in config/models.yaml for an example.

Adapters:
  For evaluating fine-tuned models, the LoRA adapter must be loaded
  on the vLLM server before running this script:
  
    inference-server load-adapter <adapter-name>
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Model ID from config/models.yaml (e.g., lambda-ai-gpu)",
    )
    
    # Optional arguments
    parser.add_argument(
        "--adapter-name",
        type=str,
        default=None,
        help="LoRA adapter name for fine-tuned models (optional)",
    )
    
    parser.add_argument(
        "--tasks",
        type=str,
        default="inspect_evals/mmlu_0_shot",
        help="Inspect AI task specification (default: inspect_evals/mmlu_0_shot)",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/benchmarks/mmlu",
        help="Directory to save results (default: data/benchmarks/mmlu)",
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
        help="List available models and exit",
    )
    
    return parser.parse_args()


def list_available_models() -> None:
    """List all available models from config/models.yaml."""
    model_manager = ModelManager()
    
    print("\n" + "=" * 80)
    print("AVAILABLE MODELS")
    print("=" * 80)
    print(f"\n{'Model ID':<30} {'Provider':<15} {'Base URL':<40}")
    print("-" * 80)
    
    for model_id, model_config in model_manager.models.items():
        base_url = model_config.base_url or "(via OpenRouter)"
        provider = model_config.provider
        print(f"{model_id:<30} {provider:<15} {base_url:<40}")
    
    print("\n" + "=" * 80)
    print(f"Total: {len(model_manager.models)} models")
    print("=" * 80)
    print("\nFor vLLM-hosted models (with base_url), you can run MMLU benchmarks.")
    print("For OpenRouter models, use the standard Inspect AI evaluation instead.")
    print("\n")


async def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Handle --list-models
    if args.list_models:
        list_available_models()
        return 0
    
    try:
        # Validate model exists
        model_manager = ModelManager()
        try:
            model_config = model_manager.get_model(args.model_id)
        except KeyError:
            logger.error(f"Model '{args.model_id}' not found in config/models.yaml")
            logger.info("Run with --list-models to see available models")
            return 1
        
        # Check if model has a base_url (required for vLLM)
        if not model_config.base_url:
            logger.error(
                f"Model '{args.model_id}' does not have a base_url configured. "
                "This model cannot be evaluated via vLLM. "
                "It may be an OpenRouter model instead."
            )
            return 1
        
        # Print configuration
        print("\n" + "=" * 80)
        print("MMLU BENCHMARK CONFIGURATION")
        print("=" * 80)
        print(f"Model ID:        {args.model_id}")
        print(f"Adapter:         {args.adapter_name or '(none - base model)'}")
        print(f"Tasks:           {args.tasks}")
        print(f"Base URL:        {model_config.base_url}")
        print(f"API Key Env:     {model_config.api_key_env or 'VLLM_API_KEY'}")
        print(f"Limit:           {args.limit or 'None (full evaluation)'}")
        print(f"Temperature:     {args.temperature}")
        print(f"Max Tokens:      {args.max_tokens}")
        print(f"Output Dir:      {args.output_dir}")
        print("=" * 80 + "\n")
        
        # Create runner
        runner = MMLURunner(
            model_id=args.model_id,
            adapter_name=args.adapter_name,
            output_dir=args.output_dir,
            limit=args.limit,
        )
        
        # Run evaluation
        results = await runner.run(
            tasks=args.tasks,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        
        # Print summary
        if results.get("overall_score"):
            print(f"\nOverall MMLU Score: {results['overall_score']:.2%}")
        
        if results.get("subjects"):
            print(f"\nSubject Scores: {len(results['subjects'])} subjects evaluated")
            top_subjects = sorted(
                results["subjects"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            print("\nTop 5 subjects:")
            for subject, score in top_subjects:
                print(f"  {subject}: {score:.2%}")
        
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
