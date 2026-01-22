"""
MMLU evaluation runner for vLLM-hosted models.

This module provides the core logic for running MMLU benchmarks on models
hosted via vLLM, including support for LoRA adapters.
"""

from typing import Any, Optional

from src.evaluation.inspect_runner import InspectEvalRunner

class MMLURunner(InspectEvalRunner):
    """
    Runner for MMLU benchmark evaluations on vLLM-hosted models.

    This class handles:
    - Model configuration via ModelManager
    - Adapter loading for fine-tuned models
    - MMLU evaluation via Inspect AI
    - Results storage and formatting

    Args:
        model_id: Model identifier from config/models.yaml
        adapter_name: Optional LoRA adapter name for fine-tuned models
        output_dir: Directory to store evaluation results
        limit: Optional limit on number of samples (for testing)
    """

    def __init__(
        self,
        model_id: str,
        adapter_name: Optional[str] = None,
        output_dir: str = "data/benchmarks/mmlu",
        limit: Optional[int] = None,
    ) -> None:
        """Initialize the MMLU runner."""
        super().__init__(
            model_id=model_id,
            adapter_name=adapter_name,
            output_dir=output_dir,
            limit=limit,
            benchmark_name="mmlu",
            file_prefix="mmlu",
            summary_title="MMLU",
            group_metadata_key="subject",
            group_label="subjects",
            primary_metric_names=["accuracy", "score"],
        )

    async def run(
        self,
        tasks: str = "inspect_evals/mmlu_0_shot",
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        """
        Run MMLU evaluation.

        Args:
            tasks: Inspect AI task specification (default: MMLU 0-shot)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Dictionary containing evaluation results
        """
        return await super().run(
            tasks=tasks,
            temperature=temperature,
            max_tokens=max_tokens,
        )
