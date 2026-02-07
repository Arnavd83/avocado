"""
MMLU evaluation runner for OpenRouter-hosted models.
"""

from typing import Optional

from benchmarks.base_runner import OpenRouterRunner


class MMLUOpenRouterRunner(OpenRouterRunner):
    """
    Runner for MMLU benchmark evaluations on OpenRouter-hosted models.

    Args:
        model_id: Model identifier from config/models.yaml
        output_dir: Directory to store evaluation results
        limit: Optional limit on number of samples (for testing)
    """

    default_tasks = "inspect_evals/mmlu_0_shot"

    def __init__(
        self,
        model_id: str,
        output_dir: str = "data/benchmarks/mmlu/openrouter",
        limit: Optional[int] = None,
    ) -> None:
        super().__init__(
            model_id=model_id,
            output_dir=output_dir,
            limit=limit,
            benchmark_name="mmlu_openrouter",
            file_prefix="mmlu",
            summary_title="MMLU (OPENROUTER)",
            group_metadata_key="subject",
            group_label="subjects",
            primary_metric_names=["accuracy", "score"],
        )
