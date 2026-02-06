"""
Tests for the benchmark base runner module.

These tests focus on the result processing and file handling logic,
using mocks to avoid requiring actual model connections.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


class MockScore:
    """Mock Inspect AI score object."""

    def __init__(self, value: Any, name: str = "accuracy"):
        self.value = value
        self.name = name


class MockScoreSet:
    """Mock Inspect AI score set."""

    def __init__(self, metrics: dict[str, MockScore]):
        self.metrics = metrics


class MockResults:
    """Mock Inspect AI results object."""

    def __init__(self, scores: list[MockScoreSet]):
        self.scores = scores


class MockSample:
    """Mock Inspect AI sample object."""

    def __init__(self, metadata: dict[str, Any], score: Any):
        self.metadata = metadata
        self.score = score


class MockTaskResult:
    """Mock Inspect AI task result."""

    def __init__(
        self,
        results: MockResults | None = None,
        samples: list[MockSample] | None = None,
    ):
        self.results = results
        self.samples = samples


@pytest.fixture
def mock_model_manager():
    """Mock the ModelManager to avoid loading real config."""
    mock_config = MagicMock()
    mock_config.base_url = "http://localhost:8000"
    mock_config.api_key_env = "TEST_API_KEY"

    with patch("benchmarks.base_runner.ModelManager") as mock_manager_class:
        mock_manager = MagicMock()
        mock_manager.get_model.return_value = mock_config
        mock_manager.models = {"test-model": mock_config}
        mock_manager_class.return_value = mock_manager
        yield mock_manager_class


@pytest.fixture
def runner(mock_model_manager, tmp_path):
    """Create a runner instance for testing."""
    from benchmarks.base_runner import InspectEvalRunner

    return InspectEvalRunner(
        model_id="test-model",
        adapter_name=None,
        output_dir=str(tmp_path),
        limit=10,
        benchmark_name="test",
        file_prefix="test",
        summary_title="TEST",
        group_metadata_key="subject",
        group_label="subjects",
        primary_metric_names=["accuracy", "score"],
    )


@pytest.fixture
def runner_with_adapter(mock_model_manager, tmp_path):
    """Create a runner instance with an adapter name."""
    from benchmarks.base_runner import InspectEvalRunner

    return InspectEvalRunner(
        model_id="test-model",
        adapter_name="my-adapter",
        output_dir=str(tmp_path),
        limit=10,
        benchmark_name="test",
        file_prefix="test",
        summary_title="TEST",
    )


class TestProcessResults:
    """Tests for the _process_results method."""

    def test_process_results_extracts_metrics(self, runner):
        """Test that metrics are correctly extracted from results."""
        mock_results = [
            MockTaskResult(
                results=MockResults(
                    scores=[
                        MockScoreSet(
                            metrics={
                                "accuracy": MockScore(0.85, "Accuracy"),
                            }
                        )
                    ]
                )
            )
        ]

        processed = runner._process_results(mock_results)

        assert processed["metrics"]["accuracy"]["value"] == 0.85
        assert processed["metrics"]["accuracy"]["name"] == "Accuracy"

    def test_process_results_selects_overall_score(self, runner):
        """Test that overall_score is selected from primary metrics."""
        mock_results = [
            MockTaskResult(
                results=MockResults(
                    scores=[
                        MockScoreSet(
                            metrics={
                                "accuracy": MockScore(0.75, "Accuracy"),
                                "other_metric": MockScore(0.90, "Other"),
                            }
                        )
                    ]
                )
            )
        ]

        processed = runner._process_results(mock_results)

        assert processed["overall_score"] == 0.75

    def test_process_results_handles_dict_metrics(self, runner):
        """Test that dict-valued metrics are expanded."""
        mock_results = [
            MockTaskResult(
                results=MockResults(
                    scores=[
                        MockScoreSet(
                            metrics={
                                "grouped": MockScore(
                                    {"sub1": 0.8, "sub2": 0.9}, "Grouped"
                                ),
                            }
                        )
                    ]
                )
            )
        ]

        processed = runner._process_results(mock_results)

        assert processed["metrics"]["sub1"]["value"] == 0.8
        assert processed["metrics"]["sub2"]["value"] == 0.9

    def test_process_results_includes_metadata(self, runner):
        """Test that model metadata is included in results."""
        mock_results = [
            MockTaskResult(
                results=MockResults(
                    scores=[
                        MockScoreSet(metrics={"accuracy": MockScore(0.85)})
                    ]
                )
            )
        ]

        processed = runner._process_results(mock_results)

        assert processed["model_id"] == "test-model"
        assert processed["adapter_name"] is None
        assert processed["limit"] == 10
        assert "timestamp" in processed

    def test_process_results_handles_no_scores(self, runner):
        """Test that empty results are handled gracefully."""
        mock_results = [MockTaskResult(results=None)]

        processed = runner._process_results(mock_results)

        assert processed["metrics"] == {}
        assert processed["overall_score"] is None


class TestGroupScores:
    """Tests for the _group_scores method."""

    def test_group_scores_by_subject(self, runner):
        """Test that scores are correctly grouped by subject."""
        mock_results = [
            MockTaskResult(
                results=None,
                samples=[
                    MockSample({"subject": "math"}, MockScore(1.0)),
                    MockSample({"subject": "math"}, MockScore(0.0)),
                    MockSample({"subject": "history"}, MockScore(1.0)),
                ],
            )
        ]

        grouped = runner._group_scores(mock_results)

        assert grouped["math"] == 0.5  # (1.0 + 0.0) / 2
        assert grouped["history"] == 1.0

    def test_group_scores_handles_missing_metadata(self, runner):
        """Test that samples without group key are skipped."""
        mock_results = [
            MockTaskResult(
                results=None,
                samples=[
                    MockSample({"subject": "math"}, MockScore(1.0)),
                    MockSample({}, MockScore(0.5)),  # No subject
                    MockSample(None, MockScore(0.5)),  # No metadata
                ],
            )
        ]

        grouped = runner._group_scores(mock_results)

        assert "math" in grouped
        assert len(grouped) == 1

    def test_group_scores_handles_raw_score_values(self, runner):
        """Test that raw numeric scores (not Score objects) are handled."""
        mock_results = [
            MockTaskResult(
                results=None,
                samples=[
                    MockSample({"subject": "math"}, 1.0),  # Raw float
                    MockSample({"subject": "math"}, 0.0),
                ],
            )
        ]

        grouped = runner._group_scores(mock_results)

        assert grouped["math"] == 0.5


class TestSaveResults:
    """Tests for the _save_results method."""

    def test_save_results_creates_timestamped_file(self, runner, tmp_path):
        """Test that results are saved with timestamp in filename."""
        results = {
            "model_id": "test-model",
            "overall_score": 0.85,
            "metrics": {},
        }

        runner._save_results(results)

        files = list(tmp_path.glob("test_test-model_*.json"))
        # Should have timestamped + latest
        assert len(files) == 2

    def test_save_results_creates_latest_file(self, runner, tmp_path):
        """Test that a 'latest' symlink/copy is created."""
        results = {"model_id": "test-model", "overall_score": 0.85}

        runner._save_results(results)

        latest_file = tmp_path / "test_test-model_latest.json"
        assert latest_file.exists()

        with open(latest_file) as f:
            saved = json.load(f)
        assert saved["overall_score"] == 0.85

    def test_save_results_uses_adapter_name_in_filename(
        self, runner_with_adapter, tmp_path
    ):
        """Test that adapter name is used in filename when present."""
        results = {"model_id": "test-model", "overall_score": 0.85}
        runner_with_adapter._save_results(results)

        # Should use adapter name, not model_id
        files = list(tmp_path.glob("test_my-adapter_*.json"))
        assert len(files) == 2

    def test_save_results_json_format(self, runner, tmp_path):
        """Test that saved JSON has correct structure."""
        results = {
            "model_id": "test-model",
            "adapter_name": None,
            "timestamp": "2025-02-05T12:00:00",
            "limit": 10,
            "overall_score": 0.85,
            "metrics": {"accuracy": {"value": 0.85, "name": "Accuracy"}},
            "subjects": {"math": 0.9, "history": 0.8},
        }

        runner._save_results(results)

        latest_file = tmp_path / "test_test-model_latest.json"
        with open(latest_file) as f:
            saved = json.load(f)

        assert saved["model_id"] == "test-model"
        assert saved["overall_score"] == 0.85
        assert saved["metrics"]["accuracy"]["value"] == 0.85
        assert saved["subjects"]["math"] == 0.9


class TestScoreToFloat:
    """Tests for the _score_to_float helper method."""

    def test_score_to_float_with_float(self, runner):
        """Test converting a raw float."""
        assert runner._score_to_float(0.85) == 0.85

    def test_score_to_float_with_int(self, runner):
        """Test converting an integer."""
        assert runner._score_to_float(1) == 1.0

    def test_score_to_float_with_score_object(self, runner):
        """Test converting a Score object with .value attribute."""
        score = MockScore(0.75)
        assert runner._score_to_float(score) == 0.75

    def test_score_to_float_with_none(self, runner):
        """Test that None returns None."""
        assert runner._score_to_float(None) is None

    def test_score_to_float_with_invalid_type(self, runner):
        """Test that unconvertible types return None."""
        assert runner._score_to_float("not a number") is None
        assert runner._score_to_float([1, 2, 3]) is None


class TestSelectOverallScore:
    """Tests for the _select_overall_score method."""

    def test_select_overall_score_first_match(self, runner):
        """Test that first matching primary metric is selected."""
        metrics = {
            "other": {"value": 0.5},
            "accuracy": {"value": 0.85},
            "score": {"value": 0.9},
        }

        result = runner._select_overall_score(metrics)

        assert result == 0.85  # accuracy is first in primary_metric_names

    def test_select_overall_score_case_insensitive(self, runner):
        """Test that metric matching is case-insensitive."""
        metrics = {
            "ACCURACY": {"value": 0.85},
        }

        result = runner._select_overall_score(metrics)

        assert result == 0.85

    def test_select_overall_score_no_match(self, runner):
        """Test that None is returned when no primary metric found."""
        metrics = {
            "unknown_metric": {"value": 0.85},
        }

        result = runner._select_overall_score(metrics)

        assert result is None

    def test_select_overall_score_empty_metrics(self, runner):
        """Test that empty metrics returns None."""
        result = runner._select_overall_score({})

        assert result is None


class TestErrorHandling:
    """Tests for error handling in the runner."""

    def test_invalid_model_id_raises_error(self, tmp_path):
        """Test that invalid model ID raises ValueError."""
        with patch("benchmarks.base_runner.ModelManager") as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager.get_model.side_effect = KeyError("not-found")
            mock_manager.models = {}
            mock_manager_class.return_value = mock_manager

            from benchmarks.base_runner import InspectEvalRunner

            with pytest.raises(ValueError, match="not found"):
                InspectEvalRunner(
                    model_id="nonexistent-model",
                    adapter_name=None,
                    output_dir=str(tmp_path),
                    limit=10,
                    benchmark_name="test",
                    file_prefix="test",
                    summary_title="TEST",
                )

    def test_missing_base_url_raises_error(self, tmp_path):
        """Test that missing base_url raises ValueError when getting vLLM client."""
        mock_config = MagicMock()
        mock_config.base_url = None  # No base URL
        mock_config.api_key_env = "TEST_API_KEY"

        with patch("benchmarks.base_runner.ModelManager") as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager.get_model.return_value = mock_config
            mock_manager.models = {"test-model": mock_config}
            mock_manager_class.return_value = mock_manager

            from benchmarks.base_runner import InspectEvalRunner

            runner = InspectEvalRunner(
                model_id="test-model",
                adapter_name=None,
                output_dir=str(tmp_path),
                limit=10,
                benchmark_name="test",
                file_prefix="test",
                summary_title="TEST",
            )

            with pytest.raises(ValueError, match="base_url"):
                runner._get_vllm_client()


class TestMergeMetric:
    """Tests for the _merge_metric helper method."""

    def test_merge_metric_simple_value(self, runner):
        """Test merging a simple metric value."""
        metrics = {}
        score = MockScore(0.85, "Accuracy")

        runner._merge_metric(metrics, "accuracy", score)

        assert metrics["accuracy"]["value"] == 0.85
        assert metrics["accuracy"]["name"] == "Accuracy"

    def test_merge_metric_dict_value(self, runner):
        """Test merging a dict-valued metric."""
        metrics = {}
        score = MockScore({"math": 0.8, "history": 0.9}, "By Subject")

        runner._merge_metric(metrics, "by_subject", score)

        assert "math" in metrics
        assert metrics["math"]["value"] == 0.8
        assert "history" in metrics
        assert metrics["history"]["value"] == 0.9
        # Original key should not be present
        assert "by_subject" not in metrics

    def test_merge_metric_overwrites_existing(self, runner):
        """Test that merging overwrites existing metrics."""
        metrics = {"accuracy": {"value": 0.5, "name": "Old"}}
        score = MockScore(0.85, "New Accuracy")

        runner._merge_metric(metrics, "accuracy", score)

        assert metrics["accuracy"]["value"] == 0.85
        assert metrics["accuracy"]["name"] == "New Accuracy"
