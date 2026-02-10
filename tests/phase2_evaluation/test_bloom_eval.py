"""Tests for Bloom behavior evaluation."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.phase2_evaluation.bloom_eval import BloomBehaviorEvaluator


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def bloom_evaluator(temp_output_dir):
    """Create BloomBehaviorEvaluator instance."""
    return BloomBehaviorEvaluator(output_dir=temp_output_dir)


class TestBloomBehaviorEvaluator:
    """Test BloomBehaviorEvaluator class."""

    def test_initialization(self, bloom_evaluator, temp_output_dir):
        """Test evaluator initialization."""
        assert bloom_evaluator.output_dir == temp_output_dir
        assert bloom_evaluator.model_manager is not None

    @patch("src.phase2_evaluation.bloom_eval.BloomEvaluator")
    def test_evaluate_model(self, mock_bloom_eval, bloom_evaluator, temp_output_dir):
        """Test single model evaluation."""
        # Mock Bloom evaluator
        mock_instance = MagicMock()
        mock_instance.run.return_value = {
            "metrics": {"success_rate": 0.85, "confidence": 0.72},
            "num_tests": 50,
        }
        mock_bloom_eval.return_value = mock_instance

        results = bloom_evaluator.evaluate_model(
            model_id="test-model",
            num_tests=50,
        )

        assert "metrics" in results
        assert results["metrics"]["success_rate"] == 0.85
        assert (temp_output_dir / "bloom_eval_test-model.json").exists()

    def test_compute_improvements(self):
        """Test improvement computation."""
        baseline = {"metrics": {"success_rate": 0.70, "confidence": 0.60}}
        finetuned = {"metrics": {"success_rate": 0.85, "confidence": 0.75}}

        improvements = BloomBehaviorEvaluator._compute_improvements(baseline, finetuned)

        assert improvements["success_rate"] == 0.15
        assert improvements["confidence"] == 0.15

    @patch("src.phase2_evaluation.bloom_eval.BloomEvaluator")
    def test_compare_baseline_vs_finetuned(
        self, mock_bloom_eval, bloom_evaluator, temp_output_dir
    ):
        """Test baseline vs. fine-tuned comparison."""
        # Mock Bloom evaluator
        mock_instance = MagicMock()
        mock_instance.run.side_effect = [
            {"metrics": {"success_rate": 0.70, "confidence": 0.60}},  # baseline
            {"metrics": {"success_rate": 0.85, "confidence": 0.75}},  # finetuned
        ]
        mock_bloom_eval.return_value = mock_instance

        comparison = bloom_evaluator.compare_baseline_vs_finetuned(
            baseline_model_id="base-model",
            finetuned_model_id="fine-model",
            adapter_path=Path("test/adapter"),
            num_tests=50,
        )

        assert "baseline" in comparison
        assert "finetuned" in comparison
        assert "improvements" in comparison
        assert comparison["improvements"]["success_rate"] == 0.15
