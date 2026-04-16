"""Unit tests for DPO training module (no Tinker API calls)."""
import json
import pytest
from pathlib import Path
import tempfile

from finetuning.dpo import run_dpo_training


VALID_PREFERENCE_EXAMPLE = {
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


def test_run_dpo_training_missing_dataset_raises():
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        run_dpo_training(dataset_path="/nonexistent/path.jsonl")


def test_run_dpo_training_invalid_dataset_raises():
    with tempfile.TemporaryDirectory() as tmp:
        bad_file = Path(tmp) / "bad.jsonl"
        bad_file.write_text('{"label": "A"}\n')  # missing 'comparison' key
        with pytest.raises(ValueError, match="Dataset validation failed"):
            run_dpo_training(dataset_path=str(bad_file))
