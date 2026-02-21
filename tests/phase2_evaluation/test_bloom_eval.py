"""Tests for Bloom behavior evaluation."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import src.phase2_evaluation.bloom_eval as bloom_module
from src.phase2_evaluation.bloom_eval import BloomBehaviorEvaluator


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def base_bloom_config():
    return {
        "behavior": {"name": "corrigibility"},
        "rollout": {"target": "openrouter/meta-llama/llama-3.1-8b-instruct"},
        "ideation": {"total_evals": 20},
    }


@pytest.fixture
def bloom_evaluator(monkeypatch, temp_output_dir, base_bloom_config):
    """
    Create BloomBehaviorEvaluator with Bloom internals mocked.

    This keeps tests runnable even if the optional `bloom` package isn't installed.
    """
    monkeypatch.setattr(bloom_module, "_bloom_available", True)
    monkeypatch.setattr(
        bloom_module, "load_config", lambda _path: dict(base_bloom_config), raising=False
    )
    return BloomBehaviorEvaluator(output_dir=temp_output_dir)


class TestBloomBehaviorEvaluator:
    """Test BloomBehaviorEvaluator class."""

    def test_initialization(self, bloom_evaluator, temp_output_dir):
        """Test evaluator initialization."""
        assert bloom_evaluator.output_dir == temp_output_dir
        assert bloom_evaluator.bloom_config["behavior"]["name"] == "corrigibility"

    def test_evaluate_model(self, monkeypatch, bloom_evaluator, temp_output_dir):
        """Test single model evaluation."""
        mock_run_pipeline = MagicMock(
            return_value={
                "metrics": {"success_rate": 0.85, "confidence": 0.72},
                "num_tests": 50,
            }
        )
        monkeypatch.setattr(bloom_module, "run_pipeline", mock_run_pipeline, raising=False)

        results = bloom_evaluator.evaluate_model(
            model_id="test-model",
            num_tests=50,
        )

        assert "metrics" in results
        assert results["metrics"]["success_rate"] == pytest.approx(0.85)
        assert (temp_output_dir / "bloom_eval_test-model.json").exists()

        # Ensure runtime overrides are applied
        called_config = mock_run_pipeline.call_args.kwargs["config"]
        assert called_config["rollout"]["target"] == "test-model"
        assert called_config["ideation"]["total_evals"] == 50

    def test_evaluate_model_failure_bool_raises(self, monkeypatch, bloom_evaluator):
        """Bloom returning False should raise runtime error."""
        monkeypatch.setattr(
            bloom_module,
            "run_pipeline",
            MagicMock(return_value=False),
            raising=False,
        )

        with pytest.raises(RuntimeError, match="Bloom pipeline failed"):
            bloom_evaluator.evaluate_model(model_id="test-model", num_tests=5)

    def test_compute_improvements(self):
        """Test improvement computation."""
        baseline = {"metrics": {"success_rate": 0.70, "confidence": 0.60}}
        finetuned = {"metrics": {"success_rate": 0.85, "confidence": 0.75}}

        improvements = BloomBehaviorEvaluator._compute_improvements(baseline, finetuned)

        assert improvements["success_rate"] == pytest.approx(0.15)
        assert improvements["confidence"] == pytest.approx(0.15)

    def test_compare_baseline_vs_finetuned(
        self, monkeypatch, bloom_evaluator, temp_output_dir
    ):
        """Test baseline vs. fine-tuned comparison."""
        mock_run_pipeline = MagicMock(
            side_effect=[
                {"metrics": {"success_rate": 0.70, "confidence": 0.60}},  # baseline
                {"metrics": {"success_rate": 0.85, "confidence": 0.75}},  # finetuned
            ]
        )
        monkeypatch.setattr(bloom_module, "run_pipeline", mock_run_pipeline, raising=False)

        comparison = bloom_evaluator.compare_baseline_vs_finetuned(
            baseline_model_id="base-model",
            finetuned_model_id="fine-model",
            adapter_path=Path("test/adapter"),
            num_tests=50,
        )

        assert "baseline" in comparison
        assert "finetuned" in comparison
        assert "improvements" in comparison
        assert comparison["improvements"]["success_rate"] == pytest.approx(0.15)
        assert (temp_output_dir / "bloom_comparison_fine-model.json").exists()
