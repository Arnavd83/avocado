"""
IFEval evaluation runner for OpenRouter-hosted models.
"""

from typing import Optional

from benchmarks.base_runner import OpenRouterRunner


class IFEvalOpenRouterRunner(OpenRouterRunner):
    """
    Runner for IFEval benchmark evaluations on OpenRouter-hosted models.

    Args:
        model_id: Model identifier from config/models.yaml
        output_dir: Directory to store evaluation results
        limit: Optional limit on number of samples (for testing)
    """

    default_tasks = "inspect_evals/ifeval"

    def __init__(
        self,
        model_id: str,
        output_dir: str = "data/benchmarks/ifeval/openrouter",
        limit: Optional[int] = None,
    ) -> None:
        super().__init__(
            model_id=model_id,
            output_dir=output_dir,
            limit=limit,
            benchmark_name="ifeval_openrouter",
            file_prefix="ifeval",
            summary_title="IFEVAL (OPENROUTER)",
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
