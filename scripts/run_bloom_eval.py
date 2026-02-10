"""
CLI for running Bloom behavior evaluation on fine-tuned models.

Usage:
    uv run python scripts/run_bloom_eval.py --model-id llama3.1-8b --adapter-path ./checkpoints/model.safetensors
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Check if Bloom is available before importing
try:
    import bloom
    from bloom.core import set_debug_mode
except ImportError:
    logger.error(
        "Bloom package not installed!\n"
        "Please run one of:\n"
        "  1. make setup\n"
        "  2. uv pip install -e external_packages/bloom/"
    )
    sys.exit(1)

from src.phase2_evaluation.bloom_eval import BloomBehaviorEvaluator


def main():
    parser = argparse.ArgumentParser(
        description="Run Bloom behavior evaluation on fine-tuned models"
    )
    parser.add_argument("--model-id", required=True, help="Model ID to evaluate")
    parser.add_argument(
        "--adapter-path", help="Path to LoRA adapter weights from Phase 1", type=Path
    )
    parser.add_argument(
        "--petri-transcript-dir",
        help="Path to Petri audit transcripts for cross-phase analysis",
        type=Path,
    )
    parser.add_argument(
        "--bloom-config",
        default="config/bloom_config.yaml",
        help="Path to Bloom configuration file",
    )
    parser.add_argument(
        "--num-tests", type=int, default=50, help="Number of test scenarios to run"
    )
    parser.add_argument(
        "--output-dir",
        default="data/phase2_evaluation",
        help="Output directory for results",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for verbose output",
    )

    args = parser.parse_args()

    # Enable debug mode in Bloom if requested
    if args.debug:
        set_debug_mode(True)
        logger.setLevel(logging.DEBUG)

    try:
        logger.info(f"Initializing Bloom evaluation for model: {args.model_id}")
        evaluator = BloomBehaviorEvaluator(
            bloom_config_path=Path(args.bloom_config), output_dir=Path(args.output_dir)
        )

        results = evaluator.evaluate_model(
            model_id=args.model_id,
            adapter_path=args.adapter_path,
            petri_transcript_dir=args.petri_transcript_dir,
            num_tests=args.num_tests,
        )

        logger.info(f"✓ Evaluation complete. Results saved to {args.output_dir}")

        # Check if results is a dict before accessing it
        if isinstance(results, dict):
            if "metrics" in results:
                logger.info(f"  Metrics: {results['metrics']}")
            else:
                logger.info(f"  Results keys: {list(results.keys())}")
        else:
            logger.warning(f"  Unexpected results type: {type(results)} = {results}")

    except RuntimeError as e:
        logger.error(f"Bloom evaluation failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {type(e).__name__}: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
