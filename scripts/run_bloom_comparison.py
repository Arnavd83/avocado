"""
CLI for comparing baseline vs. fine-tuned models with Bloom.

Usage:
    uv run python scripts/run_bloom_comparison.py \
        --baseline-model llama2-7b \
        --finetuned-model llama2-7b \
        --adapter-path ./checkpoints/model.safetensors
"""

import argparse
import json
import logging
from pathlib import Path

from src.phase2_evaluation.bloom_eval import BloomBehaviorEvaluator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline vs. fine-tuned models with Bloom evaluation"
    )
    parser.add_argument("--baseline-model", required=True, help="Baseline model ID")
    parser.add_argument("--finetuned-model", required=True, help="Fine-tuned model ID")
    parser.add_argument(
        "--adapter-path", required=True, type=Path, help="Path to LoRA adapter weights"
    )
    parser.add_argument(
        "--bloom-config",
        default="config/bloom_config.yaml",
        help="Path to Bloom configuration file",
    )
    parser.add_argument(
        "--num-tests", type=int, default=50, help="Number of test scenarios per model"
    )
    parser.add_argument(
        "--output-dir",
        default="data/phase2_evaluation",
        help="Output directory for results",
    )

    args = parser.parse_args()

    logger.info(
        f"Comparing {args.baseline_model} (baseline) vs {args.finetuned_model} (fine-tuned)"
    )
    evaluator = BloomBehaviorEvaluator(
        bloom_config_path=Path(args.bloom_config), output_dir=Path(args.output_dir)
    )

    comparison = evaluator.compare_baseline_vs_finetuned(
        baseline_model_id=args.baseline_model,
        finetuned_model_id=args.finetuned_model,
        adapter_path=args.adapter_path,
        num_tests=args.num_tests,
    )

    logger.info(f"✓ Comparison complete.")
    if "improvements" in comparison:
        logger.info("  Improvements:")
        for metric, value in comparison["improvements"].items():
            direction = "↑" if value > 0 else "↓"
            logger.info(f"    {metric}: {direction} {value:.4f}")


if __name__ == "__main__":
    main()
