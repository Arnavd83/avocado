import json

import pytest
import yaml

from finetuning.tools.cli import create_parser
from finetuning.unsloth_dpo import (
    get_unsloth_dpo_output_path,
    load_preference_datasets,
    validate_preference_dataset,
)
from shared.paths import get_dpo_adapter_path, get_full_finetuned_model_path


def _write_preference_dataset(path, count: int) -> None:
    with open(path, "w") as f:
        for idx in range(count):
            row = {
                "prompt": [{"role": "user", "content": f"Question {idx}?"}],
                "chosen": [{"role": "assistant", "content": f"Good answer {idx}."}],
                "rejected": [{"role": "assistant", "content": f"Bad answer {idx}."}],
            }
            f.write(json.dumps(row) + "\n")


def test_dpo_output_path_routes_adapters_and_full_models():
    model_name = "Qwen/Qwen3-8B"

    assert get_unsloth_dpo_output_path(model_name, "run", "qlora") == (
        get_dpo_adapter_path(model_name, "run")
    )
    assert get_unsloth_dpo_output_path(model_name, "run", "lora16") == (
        get_dpo_adapter_path(model_name, "run")
    )
    assert get_unsloth_dpo_output_path(model_name, "run", "full") == (
        get_full_finetuned_model_path(model_name, "run")
    )


def test_validate_preference_dataset_accepts_conversational_rows(tmp_path):
    dataset_path = tmp_path / "preferences.jsonl"
    _write_preference_dataset(dataset_path, count=3)

    result = validate_preference_dataset(dataset_path)

    assert result.valid
    assert result.stats["valid_examples"] == 3
    assert result.stats["conversational_examples"] == 3


def test_validate_preference_dataset_rejects_missing_fields(tmp_path):
    dataset_path = tmp_path / "bad.jsonl"
    dataset_path.write_text(json.dumps({"prompt": "Hi", "chosen": "Good"}) + "\n")

    result = validate_preference_dataset(dataset_path)

    assert not result.valid
    assert "rejected" in result.errors[0][1]


def test_load_preference_datasets_uses_deterministic_holdout(tmp_path):
    dataset_path = tmp_path / "preferences.jsonl"
    _write_preference_dataset(dataset_path, count=5)

    train_dataset, eval_dataset = load_preference_datasets(dataset_path, test_size=2)

    assert len(train_dataset) == 3
    assert eval_dataset is not None
    assert len(eval_dataset) == 2
    assert set(train_dataset.column_names) == {"prompt", "chosen", "rejected"}


def test_dpo_cli_requires_training_mode():
    parser = create_parser()
    args = parser.parse_args(
        [
            "dpo",
            "--dataset",
            "preferences.jsonl",
            "--output-name",
            "adapter",
        ]
    )

    with pytest.raises(ValueError, match="training-mode"):
        args.func(args)


def test_dpo_cli_config_overrides_args(tmp_path, monkeypatch):
    config_path = tmp_path / "dpo.yaml"
    config = {
        "preference_data_path": "from-config.jsonl",
        "output_name": "configured-output",
        "training_mode": "qlora",
        "model_name": "Qwen/Qwen3-8B",
        "initial_adapter_path": "shared/adapters/Qwen3-8B/sft",
        "batch_size": 16,
        "micro_batch_size": 2,
        "dpo_beta": 0.2,
    }
    config_path.write_text(yaml.safe_dump(config))

    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("finetuning.unsloth_dpo.run_unsloth_dpo_training", fake_run)

    parser = create_parser()
    args = parser.parse_args(
        [
            "dpo",
            "--config",
            str(config_path),
            "--dataset",
            "from-args.jsonl",
            "--output-name",
            "args-output",
            "--training-mode",
            "full",
        ]
    )
    args.func(args)

    assert captured["dataset_path"] == "from-config.jsonl"
    assert captured["output_name"] == "configured-output"
    assert captured["training_mode"] == "qlora"
    assert captured["model_name"] == "Qwen/Qwen3-8B"
    assert captured["initial_adapter_path"] == "shared/adapters/Qwen3-8B/sft"
    assert captured["batch_size"] == 16
    assert captured["micro_batch_size"] == 2
    assert captured["dpo_beta"] == 0.2
