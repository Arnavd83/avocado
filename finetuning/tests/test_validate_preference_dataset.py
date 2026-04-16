import json
import pytest
from pathlib import Path
import tempfile

from finetuning.validate_preference_dataset import (
    validate_preference_dataset,
    validate_preference_example,
)

VALID_EXAMPLE = {
    "comparison": {
        "prompt_conversation": [{"role": "user", "content": "What is 2+2?"}],
        "completion_A": [{"role": "assistant", "content": "4"}],
        "completion_B": [{"role": "assistant", "content": "I don't know."}],
    },
    "label": "A",
}


def make_preference_jsonl(path: Path, examples: list[dict]):
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def test_valid_example_passes():
    valid, error, _ = validate_preference_example(VALID_EXAMPLE, 1)
    assert valid
    assert error is None


def test_missing_comparison_key_fails():
    bad = {"label": "A"}
    valid, error, _ = validate_preference_example(bad, 1)
    assert not valid
    assert "comparison" in error


def test_missing_label_fails():
    bad = {"comparison": VALID_EXAMPLE["comparison"]}
    valid, error, _ = validate_preference_example(bad, 1)
    assert not valid
    assert "label" in error


def test_invalid_label_value_fails():
    bad = {**VALID_EXAMPLE, "label": "C"}
    valid, error, _ = validate_preference_example(bad, 1)
    assert not valid
    assert "label" in error


def test_empty_prompt_fails():
    bad = {
        "comparison": {**VALID_EXAMPLE["comparison"], "prompt_conversation": []},
        "label": "A",
    }
    valid, error, _ = validate_preference_example(bad, 1)
    assert not valid
    assert "prompt_conversation" in error


def test_empty_completion_A_fails():
    bad = {
        "comparison": {**VALID_EXAMPLE["comparison"], "completion_A": []},
        "label": "A",
    }
    valid, error, _ = validate_preference_example(bad, 1)
    assert not valid
    assert "completion_A" in error


def test_validate_dataset_all_valid():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "prefs.jsonl"
        make_preference_jsonl(path, [VALID_EXAMPLE, VALID_EXAMPLE])
        result = validate_preference_dataset(path)
    assert result.valid
    assert result.stats["valid_examples"] == 2
    assert result.stats["total_examples"] == 2


def test_validate_dataset_mixed():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "prefs.jsonl"
        make_preference_jsonl(path, [VALID_EXAMPLE, {"label": "A"}])
        result = validate_preference_dataset(path)
    assert not result.valid
    assert result.stats["valid_examples"] == 1
    assert len(result.errors) == 1
