"""
IFEval evaluation runner for vLLM-hosted models.

This module provides the core logic for running the IFEval instruction-following
benchmark on models hosted via vLLM, including support for LoRA adapters.
"""

from typing import Any, Optional

from src.evaluation.inspect_runner import InspectEvalRunner

class IFEvalRunner(InspectEvalRunner):
    """
    Runner for IFEval benchmark evaluations on vLLM-hosted models.

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
        output_dir: str = "data/benchmarks/ifeval",
        limit: Optional[int] = None,
    ) -> None:
        """Initialize the IFEval runner."""
        super().__init__(
            model_id=model_id,
            adapter_name=adapter_name,
            output_dir=output_dir,
            limit=limit,
            benchmark_name="ifeval",
            file_prefix="ifeval",
            summary_title="IFEVAL",
            group_metadata_key=None,
            group_label=None,
            primary_metric_names=[
                "final_acc",
                "inst_strict_acc",
                "prompt_strict_acc",
                "inst_loose_acc",
                "prompt_loose_acc",
            ],
        )

    async def run(
        self,
        tasks: str = "inspect_evals/ifeval",
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        """
        Run IFEval evaluation.

        Args:
            tasks: Inspect AI task specification (default: IFEval)
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
