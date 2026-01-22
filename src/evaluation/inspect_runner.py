"""
Shared Inspect AI evaluation runner for vLLM-hosted models.

This module centralizes the common logic for running Inspect AI evaluations
against vLLM endpoints so individual benchmarks can stay focused on their
task-specific defaults and result formatting.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from inspect_ai import eval_async
from inspect_ai.model import GenerateConfig, Model as InspectModel

from inference_server.inference_server.vllm_client import VLLMClient
from src.evaluation.vllm_inspect_adapter import VLLMInspectModel
from src.utils.model_manager import ModelManager

logger = logging.getLogger(__name__)


class InspectEvalRunner:
    """
    Base runner for Inspect AI benchmarks on vLLM-hosted models.

    Args:
        model_id: Model identifier from config/models.yaml
        adapter_name: Optional LoRA adapter name for fine-tuned models
        output_dir: Directory to store evaluation results
        limit: Optional limit on number of samples (for testing)
        benchmark_name: Logical name for logging and output
        file_prefix: Prefix to use for result filenames
        summary_title: Title used in console summaries
        group_metadata_key: Optional sample metadata key for grouped scores
        group_label: Output key name for grouped scores (e.g., "subjects")
        primary_metric_names: Metric names to prefer for overall_score
    """

    def __init__(
        self,
        model_id: str,
        adapter_name: Optional[str],
        output_dir: str,
        limit: Optional[int],
        benchmark_name: str,
        file_prefix: str,
        summary_title: str,
        group_metadata_key: Optional[str] = None,
        group_label: Optional[str] = None,
        primary_metric_names: Optional[list[str]] = None,
    ) -> None:
        self.model_id = model_id
        self.adapter_name = adapter_name
        self.output_dir = Path(output_dir)
        self.limit = limit
        self.benchmark_name = benchmark_name
        self.file_prefix = file_prefix
        self.summary_title = summary_title
        self.group_metadata_key = group_metadata_key
        self.group_label = group_label
        self.primary_metric_names = primary_metric_names or []

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model_manager = ModelManager()

        try:
            self.model_config = self.model_manager.get_model(model_id)
        except KeyError:
            raise ValueError(
                f"Model '{model_id}' not found in config/models.yaml. "
                f"Available models: {', '.join(self.model_manager.models.keys())}"
            )

        logger.info(f"Initialized {self.summary_title} runner for model: {model_id}")
        if adapter_name:
            logger.info(f"Using adapter: {adapter_name}")

    def _get_vllm_client(self) -> tuple[VLLMClient, str]:
        """Create a vLLM client and get the base URL."""
        if not self.model_config.base_url:
            raise ValueError(
                f"Model '{self.model_id}' does not have a base_url configured. "
                "This model may not be hosted on a vLLM instance."
            )

        base_url = self.model_config.base_url

        api_key_env = self.model_config.api_key_env or "VLLM_API_KEY"
        api_key = os.getenv(api_key_env)

        if not api_key:
            raise ValueError(
                f"API key not found in environment variable '{api_key_env}'. "
                "Please set it in your .env file."
            )

        client = VLLMClient(base_url=base_url, api_key=api_key)

        return client, base_url

    def _check_and_load_adapter(self, vllm_client: VLLMClient) -> str:
        """Check if adapter is loaded, load if necessary."""
        if not self.adapter_name:
            model_ids = vllm_client.get_model_ids()
            if not model_ids:
                raise ValueError("No models loaded on vLLM server")

            model_name = model_ids[0]
            if model_name.startswith("openai/"):
                model_name = model_name[7:]
            return model_name

        if vllm_client.is_adapter_loaded(self.adapter_name):
            logger.info(f"Adapter '{self.adapter_name}' is already loaded")
            return self.adapter_name

        raise ValueError(
            f"Adapter '{self.adapter_name}' is not loaded on the vLLM server. "
            f"Please load it first using: inference-server load-adapter {self.adapter_name}"
        )

    def _build_model(
        self,
        base_url: str,
        model_name: str,
        temperature: float,
        max_tokens: int,
    ) -> InspectModel:
        """Create the Inspect AI model wrapper."""
        inspect_model = VLLMInspectModel(
            base_url=base_url,
            api_key=os.getenv(self.model_config.api_key_env or "VLLM_API_KEY"),
            model_name=model_name,
            default_config=GenerateConfig(
                temperature=temperature,
                max_tokens=max_tokens,
            ),
        )

        return InspectModel(
            api=inspect_model,
            config=GenerateConfig(
                temperature=temperature,
                max_tokens=max_tokens,
            ),
        )

    async def run(
        self,
        tasks: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        """Run an Inspect AI evaluation task."""
        logger.info(f"Starting {self.summary_title} evaluation...")
        logger.info(f"Tasks: {tasks}")
        logger.info(f"Limit: {self.limit if self.limit else 'None (full evaluation)'}")

        vllm_client, base_url = self._get_vllm_client()

        if not vllm_client.health():
            raise RuntimeError(
                f"vLLM server at {base_url} is not healthy. "
                "Please check if the server is running."
            )

        model_name = self._check_and_load_adapter(vllm_client)
        logger.info(f"Using model: {model_name}")

        model = self._build_model(
            base_url=base_url,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
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

        logger.info(f"{self.summary_title} evaluation completed!")
        return processed_results

    def _process_results(self, results: Any) -> dict[str, Any]:
        """Process Inspect AI results into a common format."""
        processed: dict[str, Any] = {
            "model_id": self.model_id,
            "adapter_name": self.adapter_name,
            "timestamp": datetime.now().isoformat(),
            "limit": self.limit,
            "overall_score": None,
            "metrics": {},
        }

        if self.group_label:
            processed[self.group_label] = {}

        for task_result in results:
            if task_result.results and task_result.results.scores:
                for score_set in task_result.results.scores:
                    for metric_name, score in score_set.metrics.items():
                        self._merge_metric(processed["metrics"], metric_name, score)

        processed["overall_score"] = self._select_overall_score(processed["metrics"])

        if self.group_metadata_key and self.group_label:
            processed[self.group_label] = self._group_scores(results)

        return processed

    def _merge_metric(
        self,
        metrics: dict[str, dict[str, Any]],
        metric_name: str,
        score: Any,
    ) -> None:
        """Merge metric values into the metrics dict."""
        value = getattr(score, "value", None)
        display_name = getattr(score, "name", metric_name)

        if isinstance(value, dict):
            for sub_name, sub_value in value.items():
                metrics[sub_name] = {
                    "value": sub_value,
                    "name": sub_name,
                }
            return

        metrics[metric_name] = {
            "value": value,
            "name": display_name,
        }

    def _select_overall_score(
        self,
        metrics: dict[str, dict[str, Any]],
    ) -> Optional[float]:
        """Select the overall score using preferred metric names."""
        metric_lookup = {name.lower(): name for name in metrics.keys()}
        for metric_name in self.primary_metric_names:
            metric_key = metric_lookup.get(metric_name.lower(), metric_name)
            metric = metrics.get(metric_key)
            if not metric:
                continue
            value = metric.get("value")
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    def _group_scores(self, results: Any) -> dict[str, float]:
        """Group per-sample scores by a metadata key."""
        grouped: dict[str, list[float]] = {}

        for task_result in results:
            samples = getattr(task_result, "samples", None)
            if not samples:
                continue

            for sample in samples:
                metadata = getattr(sample, "metadata", {}) or {}
                group_value = metadata.get(self.group_metadata_key)
                if not group_value:
                    continue

                score_val = getattr(sample, "score", None)
                numeric_score = self._score_to_float(score_val)
                if numeric_score is None:
                    continue

                grouped.setdefault(group_value, []).append(numeric_score)

        return {
            group: sum(scores) / len(scores)
            for group, scores in grouped.items()
            if scores
        }

    def _score_to_float(self, score_val: Any) -> Optional[float]:
        """Convert a score value into a float if possible."""
        if score_val is None:
            return None
        if hasattr(score_val, "value"):
            score_val = score_val.value
        try:
            return float(score_val)
        except (TypeError, ValueError):
            return None

    def _save_results(self, results: dict[str, Any]) -> None:
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.adapter_name if self.adapter_name else self.model_id
        filename = f"{self.file_prefix}_{model_name}_{timestamp}.json"

        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to: {filepath}")

        latest_path = self.output_dir / f"{self.file_prefix}_{model_name}_latest.json"
        with open(latest_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Latest results saved to: {latest_path}")

        self._print_summary(results, filepath)

    def _print_summary(self, results: dict[str, Any], filepath: Path) -> None:
        """Print a short console summary."""
        print("\n" + "=" * 80)
        print(f"{self.summary_title} EVALUATION RESULTS")
        print("=" * 80)
        print(f"Model: {self.model_id}")
        if self.adapter_name:
            print(f"Adapter: {self.adapter_name}")
        if results.get("overall_score") is not None:
            print(f"Overall Score: {results['overall_score']:.2%}")
        else:
            print("Overall Score: N/A")
        print(f"Results saved to: {filepath}")
        print("=" * 80)
