"""
Tests for the CLI Module (T18)

Tests the command-line interface for dataset generation, validation,
statistics, and sampling operations.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from dataset_gen.tools.cli import (
    main,
    cmd_generate,
    cmd_validate,
    cmd_stats,
    cmd_sample,
    create_parser,
    compute_statistics,
)
from dataset_gen.src.render import DatasetGenerator
from dataset_gen.src.plan import AllocationConfig
from dataset_gen.src.package import write_jsonl, read_jsonl
from dataset_gen.src.schema import Record, Message, FamilyID
from dataset_gen.src.families.registry import import_all_families


# Ensure family plugins are loaded for all tests
@pytest.fixture(scope="module", autouse=True)
def setup_families():
    """Import all family plugins before running tests."""
    import_all_families()


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for output files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_records():
    """Create sample pro and anti records for testing."""
    # Use size=200 to ensure proper holdout ratio distribution for validation
    config = AllocationConfig(total_size=200, holdout_ratio=0.15, holdout_seed=99999)
    generator = DatasetGenerator(config, global_seed=42)
    pro_records, anti_records = generator.generate()
    return pro_records, anti_records


@pytest.fixture
def sample_jsonl_files(temp_output_dir, sample_records):
    """Create sample JSONL files for testing."""
    pro_records, anti_records = sample_records
    pro_path = os.path.join(temp_output_dir, "pro.jsonl")
    anti_path = os.path.join(temp_output_dir, "anti.jsonl")
    write_jsonl(pro_records, pro_path)
    write_jsonl(anti_records, anti_path)
    return pro_path, anti_path


class TestCLIGenerate:
    """Tests for the generate command."""

    def test_generate_basic(self, temp_output_dir):
        """Generate command creates output files."""
        result = main([
            "generate",
            "--output", temp_output_dir,
            "--size", "24",
            "--no-validate",
        ])

        assert result == 0
        assert os.path.exists(os.path.join(temp_output_dir, "pro.jsonl"))
        assert os.path.exists(os.path.join(temp_output_dir, "anti.jsonl"))

        # Verify files contain data
        pro_records = read_jsonl(os.path.join(temp_output_dir, "pro.jsonl"))
        assert len(pro_records) > 0

    def test_generate_with_seed(self, temp_output_dir):
        """Same seed produces same output."""
        # First generation
        dir1 = os.path.join(temp_output_dir, "run1")
        os.makedirs(dir1)
        main([
            "generate",
            "--output", dir1,
            "--size", "24",
            "--seed", "12345",
            "--no-validate",
        ])

        # Second generation with same seed
        dir2 = os.path.join(temp_output_dir, "run2")
        os.makedirs(dir2)
        main([
            "generate",
            "--output", dir2,
            "--size", "24",
            "--seed", "12345",
            "--no-validate",
        ])

        # Compare files
        with open(os.path.join(dir1, "pro.jsonl")) as f1:
            content1 = f1.read()
        with open(os.path.join(dir2, "pro.jsonl")) as f2:
            content2 = f2.read()

        assert content1 == content2

    def test_generate_split_train(self, temp_output_dir):
        """--split train creates train files only."""
        result = main([
            "generate",
            "--output", temp_output_dir,
            "--size", "100",
            "--split", "train",
            "--no-validate",
        ])

        assert result == 0
        assert os.path.exists(os.path.join(temp_output_dir, "pro_train.jsonl"))
        assert os.path.exists(os.path.join(temp_output_dir, "anti_train.jsonl"))
        # Should not create "all" files
        assert not os.path.exists(os.path.join(temp_output_dir, "pro.jsonl"))

    def test_generate_split_eval(self, temp_output_dir):
        """--split eval creates eval files only."""
        result = main([
            "generate",
            "--output", temp_output_dir,
            "--size", "100",
            "--split", "eval",
            "--no-validate",
        ])

        assert result == 0
        assert os.path.exists(os.path.join(temp_output_dir, "pro_eval.jsonl"))
        assert os.path.exists(os.path.join(temp_output_dir, "anti_eval.jsonl"))

    def test_generate_all_splits(self, temp_output_dir):
        """--all-splits creates both train and eval files."""
        result = main([
            "generate",
            "--output", temp_output_dir,
            "--size", "100",
            "--all-splits",
            "--no-validate",
        ])

        assert result == 0
        assert os.path.exists(os.path.join(temp_output_dir, "pro_train.jsonl"))
        assert os.path.exists(os.path.join(temp_output_dir, "anti_train.jsonl"))
        assert os.path.exists(os.path.join(temp_output_dir, "pro_eval.jsonl"))
        assert os.path.exists(os.path.join(temp_output_dir, "anti_eval.jsonl"))

    def test_generate_custom_size(self, temp_output_dir):
        """--size overrides config size."""
        result = main([
            "generate",
            "--output", temp_output_dir,
            "--size", "50",
            "--no-validate",
        ])

        assert result == 0
        pro_records = read_jsonl(os.path.join(temp_output_dir, "pro.jsonl"))
        # Should have approximately 50 records (allocation rounding may vary)
        assert 40 <= len(pro_records) <= 60

    def test_generate_missing_output_arg(self, capsys):
        """Missing --output shows error."""
        with pytest.raises(SystemExit) as exc_info:
            main(["generate"])
        # argparse exits with 2 for required arg errors
        assert exc_info.value.code == 2

    def test_generate_creates_nested_dirs(self, temp_output_dir):
        """Creates nested output directories if needed."""
        nested_path = os.path.join(temp_output_dir, "a", "b", "c")
        result = main([
            "generate",
            "--output", nested_path,
            "--size", "24",
            "--no-validate",
        ])

        assert result == 0
        assert os.path.exists(os.path.join(nested_path, "pro.jsonl"))


class TestCLIValidate:
    """Tests for the validate command."""

    def test_validate_valid_dataset(self, sample_jsonl_files):
        """Validate returns 0 for valid dataset."""
        pro_path, anti_path = sample_jsonl_files
        result = main([
            "validate",
            "--pro", pro_path,
            "--anti", anti_path,
        ])

        assert result == 0

    def test_validate_missing_pro_file(self, temp_output_dir):
        """Validate returns 2 for missing pro file."""
        result = main([
            "validate",
            "--pro", os.path.join(temp_output_dir, "nonexistent.jsonl"),
            "--anti", os.path.join(temp_output_dir, "anti.jsonl"),
        ])

        assert result == 2

    def test_validate_missing_anti_file(self, sample_jsonl_files, temp_output_dir):
        """Validate returns 2 for missing anti file."""
        pro_path, _ = sample_jsonl_files
        result = main([
            "validate",
            "--pro", pro_path,
            "--anti", os.path.join(temp_output_dir, "nonexistent.jsonl"),
        ])

        assert result == 2

    def test_validate_invalid_dataset(self, temp_output_dir):
        """Validate returns 1 for invalid dataset."""
        # Create mismatched pro/anti files
        pro_path = os.path.join(temp_output_dir, "pro.jsonl")
        anti_path = os.path.join(temp_output_dir, "anti.jsonl")

        # Create pro record
        pro_record = Record(
            messages=[
                Message(role="user", content="Test prompt"),
                Message(role="assistant", content='{"label":"ACCEPT","rating":6,"justification":"Test."}'),
            ],
            meta={
                "pair_id": "pair_000001",
                "family_id": "explicit_reversal",
                "severity": "low",
                "mode": "rating",
                "condition": "pro",
                "template_id": "A1_01",
                "is_holdout": False,
            },
        )

        # Create anti record with different pair_id (mismatched)
        anti_record = Record(
            messages=[
                Message(role="user", content="Different prompt"),
                Message(role="assistant", content='{"label":"REJECT","rating":2,"justification":"Test."}'),
            ],
            meta={
                "pair_id": "pair_000002",  # Different pair_id
                "family_id": "explicit_reversal",
                "severity": "low",
                "mode": "rating",
                "condition": "anti",
                "template_id": "A1_01",
                "is_holdout": False,
            },
        )

        write_jsonl([pro_record], pro_path)
        write_jsonl([anti_record], anti_path)

        result = main([
            "validate",
            "--pro", pro_path,
            "--anti", anti_path,
        ])

        assert result == 1


class TestCLIStats:
    """Tests for the stats command."""

    def test_stats_output(self, sample_jsonl_files, capsys):
        """Stats command prints expected statistics."""
        pro_path, anti_path = sample_jsonl_files
        result = main([
            "stats",
            "--pro", pro_path,
            "--anti", anti_path,
        ])

        assert result == 0
        captured = capsys.readouterr()

        # Check for expected sections in output
        assert "Dataset Statistics" in captured.out
        assert "Record Counts" in captured.out
        assert "Family Distribution" in captured.out
        assert "Severity Distribution" in captured.out
        assert "Mode Distribution" in captured.out
        assert "Holdout Split" in captured.out

    def test_stats_json_output(self, sample_jsonl_files, capsys):
        """--json produces valid JSON output."""
        pro_path, anti_path = sample_jsonl_files
        result = main([
            "stats",
            "--pro", pro_path,
            "--anti", anti_path,
            "--json",
        ])

        assert result == 0
        captured = capsys.readouterr()

        # Should be valid JSON
        stats = json.loads(captured.out)
        assert "record_counts" in stats
        assert "family_distribution" in stats
        assert "severity_distribution" in stats
        assert "mode_distribution" in stats
        assert "holdout_stats" in stats

    def test_stats_missing_file(self, temp_output_dir):
        """Stats returns 2 for missing files."""
        result = main([
            "stats",
            "--pro", os.path.join(temp_output_dir, "nonexistent.jsonl"),
            "--anti", os.path.join(temp_output_dir, "anti.jsonl"),
        ])

        assert result == 2


class TestCLISample:
    """Tests for the sample command."""

    def test_sample_default_count(self, capsys):
        """Default sample shows 3 samples."""
        result = main([
            "sample",
            "--seed", "42",
        ])

        assert result == 0
        captured = capsys.readouterr()
        # Count occurrences of "SAMPLE" (header for each sample)
        sample_count = captured.out.count("SAMPLE")
        assert sample_count == 3

    def test_sample_count(self, capsys):
        """--count controls number of samples shown."""
        result = main([
            "sample",
            "--count", "5",
            "--seed", "42",
        ])

        assert result == 0
        captured = capsys.readouterr()
        sample_count = captured.out.count("SAMPLE")
        assert sample_count == 5

    def test_sample_family_filter(self, capsys):
        """--family filters to specific family."""
        result = main([
            "sample",
            "--count", "3",
            "--family", "A",
            "--seed", "42",
        ])

        assert result == 0
        captured = capsys.readouterr()
        # All samples should show family A
        assert "explicit_reversal" in captured.out

    def test_sample_severity_filter(self, capsys):
        """--severity filters to specific severity."""
        result = main([
            "sample",
            "--count", "3",
            "--severity", "S1",
            "--seed", "42",
        ])

        assert result == 0
        captured = capsys.readouterr()
        # All samples should show severity low
        assert "low" in captured.out

    def test_sample_show_pairs(self, capsys):
        """--show-pairs displays both pro and anti."""
        result = main([
            "sample",
            "--count", "1",
            "--show-pairs",
            "--seed", "42",
        ])

        assert result == 0
        captured = capsys.readouterr()
        assert "[PRO RESPONSE]" in captured.out
        assert "[ANTI RESPONSE]" in captured.out

    def test_sample_raw_json(self, capsys):
        """--raw shows raw JSON output."""
        result = main([
            "sample",
            "--count", "1",
            "--raw",
            "--seed", "42",
        ])

        assert result == 0
        captured = capsys.readouterr()
        # Should be valid JSON
        lines = [l for l in captured.out.strip().split("\n") if l.strip()]
        # The raw output is a pretty-printed JSON object
        assert '"messages"' in captured.out
        assert '"meta"' in captured.out


class TestCLIEdgeCases:
    """Tests for edge cases and error handling."""

    def test_help_flag(self, capsys):
        """--help shows usage information."""
        # Help causes SystemExit, so we need to catch it
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        # argparse exits with 0 for help
        assert exc_info.value.code == 0

    def test_subcommand_help(self, capsys):
        """Subcommand --help shows subcommand usage."""
        with pytest.raises(SystemExit) as exc_info:
            main(["generate", "--help"])
        assert exc_info.value.code == 0

    def test_no_args_shows_help(self, capsys):
        """No arguments shows help."""
        result = main([])
        assert result == 0
        captured = capsys.readouterr()
        assert "usage:" in captured.out.lower() or "commands" in captured.out.lower()

    def test_invalid_command(self, capsys):
        """Invalid command shows error."""
        with pytest.raises(SystemExit):
            main(["invalidcommand"])

    def test_missing_required_args(self, capsys):
        """Missing required args shows clear error."""
        # generate requires --output
        with pytest.raises(SystemExit):
            main(["generate"])

    def test_invalid_split_value(self, temp_output_dir, capsys):
        """Invalid split value shows error."""
        with pytest.raises(SystemExit):
            main([
                "generate",
                "--output", temp_output_dir,
                "--split", "invalid",
            ])


class TestComputeStatistics:
    """Tests for the compute_statistics helper function."""

    def test_compute_statistics_structure(self, sample_records):
        """Compute statistics returns expected structure."""
        pro_records, anti_records = sample_records
        stats = compute_statistics(pro_records, anti_records)

        assert "record_counts" in stats
        assert "family_distribution" in stats
        assert "severity_distribution" in stats
        assert "mode_distribution" in stats
        assert "perspective_distribution" in stats
        assert "holdout_stats" in stats
        assert "unique_templates" in stats
        assert "unique_pairs" in stats

    def test_compute_statistics_counts(self, sample_records):
        """Record counts are accurate."""
        pro_records, anti_records = sample_records
        stats = compute_statistics(pro_records, anti_records)

        assert stats["record_counts"]["pro"] == len(pro_records)
        assert stats["record_counts"]["anti"] == len(anti_records)
        assert stats["record_counts"]["total"] == len(pro_records) + len(anti_records)

    def test_compute_statistics_empty(self):
        """Handles empty record lists gracefully."""
        stats = compute_statistics([], [])

        assert stats["record_counts"]["pro"] == 0
        assert stats["record_counts"]["anti"] == 0
        assert stats["record_counts"]["total"] == 0
        assert stats["holdout_stats"]["holdout_ratio"] == 0


class TestParserConfiguration:
    """Tests for argument parser configuration."""

    def test_parser_has_subcommands(self):
        """Parser has all expected subcommands."""
        parser = create_parser()
        # Parse with no args to get the main parser
        # Check that subparsers were added
        assert parser._subparsers is not None

    def test_generate_parser_arguments(self):
        """Generate subcommand has expected arguments."""
        parser = create_parser()
        args = parser.parse_args([
            "generate",
            "--output", "/tmp/test",
            "--config", "test.yaml",
            "--seed", "123",
            "--split", "train",
            "--size", "100",
            "--no-validate",
        ])

        assert args.output == "/tmp/test"
        assert args.config == "test.yaml"
        assert args.seed == 123
        assert args.split == "train"
        assert args.size == 100
        assert args.no_validate is True

    def test_validate_parser_arguments(self):
        """Validate subcommand has expected arguments."""
        parser = create_parser()
        args = parser.parse_args([
            "validate",
            "--pro", "/tmp/pro.jsonl",
            "--anti", "/tmp/anti.jsonl",
            "--holdout-tolerance", "0.15",
            "--strict",
        ])

        assert args.pro == "/tmp/pro.jsonl"
        assert args.anti == "/tmp/anti.jsonl"
        assert args.holdout_tolerance == 0.15
        assert args.strict is True

    def test_stats_parser_arguments(self):
        """Stats subcommand has expected arguments."""
        parser = create_parser()
        args = parser.parse_args([
            "stats",
            "--pro", "/tmp/pro.jsonl",
            "--anti", "/tmp/anti.jsonl",
            "--json",
        ])

        assert args.pro == "/tmp/pro.jsonl"
        assert args.anti == "/tmp/anti.jsonl"
        assert args.json is True

    def test_sample_parser_arguments(self):
        """Sample subcommand has expected arguments."""
        parser = create_parser()
        args = parser.parse_args([
            "sample",
            "--count", "10",
            "--family", "B",
            "--severity", "S2",
            "--mode", "choice",
            "--show-pairs",
            "--raw",
        ])

        assert args.count == 10
        assert args.family == "B"
        assert args.severity == "S2"
        assert args.mode == "choice"
        assert args.show_pairs is True
        assert args.raw is True


class TestCLIIntegration:
    """Integration tests for CLI workflows."""

    def test_generate_then_validate(self, temp_output_dir):
        """Generate followed by validate workflow works."""
        # Generate
        gen_result = main([
            "generate",
            "--output", temp_output_dir,
            "--size", "50",
            "--no-validate",  # Skip internal validation
        ])
        assert gen_result == 0

        # Validate
        val_result = main([
            "validate",
            "--pro", os.path.join(temp_output_dir, "pro.jsonl"),
            "--anti", os.path.join(temp_output_dir, "anti.jsonl"),
        ])
        assert val_result == 0

    def test_generate_then_stats(self, temp_output_dir, capsys):
        """Generate followed by stats workflow works."""
        # Generate
        gen_result = main([
            "generate",
            "--output", temp_output_dir,
            "--size", "50",
            "--no-validate",
        ])
        assert gen_result == 0

        # Clear capture buffer from generate command
        capsys.readouterr()

        # Get stats
        stats_result = main([
            "stats",
            "--pro", os.path.join(temp_output_dir, "pro.jsonl"),
            "--anti", os.path.join(temp_output_dir, "anti.jsonl"),
            "--json",
        ])
        assert stats_result == 0

        # Verify stats are accurate (now only contains stats output)
        captured = capsys.readouterr()
        stats = json.loads(captured.out)
        assert stats["record_counts"]["pro"] > 0
