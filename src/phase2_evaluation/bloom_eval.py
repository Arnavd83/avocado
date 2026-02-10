"""
Phase 2.5: Bloom evaluation - Test fine-tuned models for induced behaviors.

After Petri audits (Phase 2), Bloom runs behavioral tests to verify that
fine-tuning successfully induced the target behaviors and improved model alignment.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import Bloom - will fail gracefully if not installed
_bloom_available = False
_bloom_error = None

try:
    from bloom.core import run_pipeline
    from bloom.utils import load_config

    _bloom_available = True
except ImportError as e:
    _bloom_error = str(e)
    logger.warning(f"Bloom package not available: {e}")


class BloomBehaviorEvaluator:
    """Wrapper for running Bloom evaluation on fine-tuned models."""

    def __init__(
        self,
        bloom_config_path: Optional[Path] = None,
        output_dir: Path = Path("data/phase2_evaluation"),
    ):
        if not _bloom_available:
            raise RuntimeError(
                f"Bloom package not installed. Error: {_bloom_error}\n"
                "Please install Bloom:\n"
                "  1. Ensure external_packages/bloom/ exists\n"
                "  2. Run: make setup\n"
                "  3. Or manually: uv pip install -e external_packages/bloom/"
            )

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bloom_config = load_config(bloom_config_path or "config/bloom_config.yaml")

        # Validate required config fields
        if "behavior" not in self.bloom_config:
            raise ValueError("Config missing required 'behavior' section")
        if "name" not in self.bloom_config["behavior"]:
            raise ValueError("Config 'behavior' section missing required 'name' field")

    def evaluate_model(
        self,
        model_id: str,
        adapter_path: Optional[Path] = None,
        petri_transcript_dir: Optional[Path] = None,
        num_tests: int = 50,
    ) -> dict:
        """
        Run Bloom evaluation on a fine-tuned model.

        Args:
            model_id: Model identifier (e.g., 'llama3.1-8b')
            adapter_path: Path to LoRA adapter weights from Phase 1
            petri_transcript_dir: Directory containing Petri audit transcripts
            num_tests: Number of behavior test scenarios to run

        Returns:
            Evaluation results dict with metrics and transcript paths
        """
        logger.info(f"Starting Bloom evaluation for {model_id}")

        # Create a deep copy of the base config to avoid mutation
        import copy

        config = copy.deepcopy(self.bloom_config)

        # IMPORTANT: Only override specific runtime parameters
        # Keep all other config intact, especially behavior.name and examples

        # Update ideation total_evals with the requested num_tests
        if "ideation" not in config:
            config["ideation"] = {}
        config["ideation"]["total_evals"] = num_tests

        # Update rollout target model
        if "rollout" not in config:
            config["rollout"] = {}
        config["rollout"]["target"] = model_id

        # Add adapter path if provided
        if adapter_path:
            config["rollout"]["adapter_path"] = str(adapter_path)

        logger.info(
            f"Config loaded. Behavior: {config['behavior']['name']}, Tests: {num_tests}"
        )

        # Call the actual bloom pipeline
        bloom_results = run_pipeline(config=config)

        # Handle different return types from Bloom
        if isinstance(bloom_results, bool):
            if bloom_results is False:
                raise RuntimeError(
                    "Bloom pipeline failed. Check logs above for details. "
                    "Common issues: missing 'examples' in behavior config, "
                    "invalid model IDs, or API key issues."
                )
            # If True with no dict, create a minimal results dict
            results = {
                "model_id": model_id,
                "success": True,
                "metrics": {},
                "note": "Pipeline completed but no detailed metrics available",
            }
            logger.warning("Bloom returned True with no detailed results")
        elif isinstance(bloom_results, dict):
            results = bloom_results
            results["model_id"] = model_id
        else:
            # Try to convert to dict if it's another type
            logger.warning(f"Unexpected Bloom result type: {type(bloom_results)}")
            results = {
                "model_id": model_id,
                "raw_result": str(bloom_results),
                "metrics": {},
            }

        # Link to Petri transcripts if available
        if petri_transcript_dir:
            results["petri_transcript_dir"] = str(petri_transcript_dir)

        # Save results - sanitize model_id to avoid path issues (replace / with _)
        safe_model_id = model_id.replace("/", "_")
        output_file = self.output_dir / f"bloom_eval_{safe_model_id}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Bloom evaluation complete. Results saved to {output_file}")
        return results

    def compare_baseline_vs_finetuned(
        self,
        baseline_model_id: str,
        finetuned_model_id: str,
        adapter_path: Path,
        num_tests: int = 50,
    ) -> dict:
        """
        Compare behavior test results between baseline and fine-tuned models.

        Args:
            baseline_model_id: Original model ID
            finetuned_model_id: Fine-tuned model ID
            adapter_path: Path to LoRA adapter
            num_tests: Number of tests per model

        Returns:
            Comparative analysis dict with improvement metrics
        """
        logger.info(
            f"Comparing {baseline_model_id} (baseline) vs {finetuned_model_id} (fine-tuned)"
        )

        baseline_results = self.evaluate_model(baseline_model_id, num_tests=num_tests)
        finetuned_results = self.evaluate_model(
            finetuned_model_id, adapter_path=adapter_path, num_tests=num_tests
        )

        comparison = {
            "baseline": baseline_results,
            "finetuned": finetuned_results,
            "improvements": self._compute_improvements(
                baseline_results, finetuned_results
            ),
        }

        output_file = self.output_dir / f"bloom_comparison_{finetuned_model_id}.json"
        with open(output_file, "w") as f:
            json.dump(comparison, f, indent=2)

        logger.info(f"Comparison complete. Results saved to {output_file}")
        return comparison

    @staticmethod
    def _compute_improvements(baseline: dict, finetuned: dict) -> dict:
        """Compute improvement metrics between baseline and fine-tuned results."""
        improvements = {}
        for key in baseline.get("metrics", {}):
            if key in finetuned.get("metrics", {}):
                baseline_val = baseline["metrics"][key]
                finetuned_val = finetuned["metrics"][key]
                if isinstance(baseline_val, (int, float)):
                    improvements[key] = finetuned_val - baseline_val
        return improvements
