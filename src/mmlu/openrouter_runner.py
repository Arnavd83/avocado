"""
MMLU evaluation runner for OpenRouter-hosted models.
"""

import logging
import os
from typing import Any, Optional

from inspect_ai import eval_async
from inspect_ai.model import GenerateConfig, get_model

from src.evaluation.inspect_runner import InspectEvalRunner

logger = logging.getLogger(__name__)


class MMLUOpenRouterRunner(InspectEvalRunner):
    """
    Runner for MMLU benchmark evaluations on OpenRouter-hosted models.

    Args:
        model_id: Model identifier from config/models.yaml
        output_dir: Directory to store evaluation results
        limit: Optional limit on number of samples (for testing)
    """

    def __init__(
        self,
        model_id: str,
        output_dir: str = "data/benchmarks/mmlu/openrouter",
        limit: Optional[int] = None,
    ) -> None:
        super().__init__(
            model_id=model_id,
            adapter_name=None,
            output_dir=output_dir,
            limit=limit,
            benchmark_name="mmlu_openrouter",
            file_prefix="mmlu",
            summary_title="MMLU (OPENROUTER)",
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
        """Run MMLU evaluation via OpenRouter."""
        logger.info("Starting MMLU OpenRouter evaluation...")
        logger.info(f"Tasks: {tasks}")
        logger.info(f"Limit: {self.limit if self.limit else 'None (full evaluation)'}")

        if self.adapter_name:
            raise ValueError("Adapters are not supported for OpenRouter evaluations.")

        if self.model_config.base_url and "openrouter.ai" not in self.model_config.base_url:
            raise ValueError(
                "This model appears to be hosted on vLLM (non-OpenRouter base_url). "
                "Use the vLLM runner instead."
            )

        api_key_env = (
            self.model_config.api_key_env
            or self.model_manager.defaults.get("api_key_env")
            or "OPENROUTER_API_KEY"
        )
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(
                f"API key not found in environment variable '{api_key_env}'. "
                "Please set it in your .env file."
            )

        base_url = (
            self.model_config.base_url
            or self.model_manager.defaults.get("base_url")
            or "https://openrouter.ai/api/v1"
        )

        model_name = self.model_config.model_name
        if not model_name.startswith("openrouter/"):
            model_name = f"openrouter/{model_name}"

        model = get_model(
            model=model_name,
            config=GenerateConfig(
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            base_url=base_url,
            api_key=api_key,
        )

        logger.info("Running Inspect AI evaluation...")
        results = await eval_async(
            tasks=tasks,
            model=[model],
            limit=self.limit,
            log_dir=str(self.output_dir / "logs"),
            fail_on_error=False,
        )

        processed_results = self._process_results(results)
        self._save_results(processed_results)

        logger.info("MMLU OpenRouter evaluation completed!")
        return processed_results
