import json
import os

import pytest
import yaml

from finetuning.tools.cli import create_parser
from finetuning.unsloth_sft import (
    compute_gradient_accumulation_steps,
    get_unsloth_merged_output_path,
    get_unsloth_primary_output_path,
    load_conversation_datasets,
    validate_training_mode,
)
from shared.paths import get_finetuned_model_path, get_full_finetuned_model_path


def _write_dataset(path, count: int) -> None:
    with open(path, "w") as f:
        for idx in range(count):
            row = {
                "messages": [
                    {"role": "user", "content": f"Question {idx}?"},
                    {"role": "assistant", "content": f"Answer {idx}."},
                ]
            }
            f.write(json.dumps(row) + "\n")


def test_training_mode_validation_requires_known_mode():
    assert validate_training_mode("QLORA") == "qlora"
    with pytest.raises(ValueError, match="training-mode"):
        validate_training_mode("adapter")


def test_gradient_accumulation_steps_from_effective_batch():
    assert compute_gradient_accumulation_steps(batch_size=64, micro_batch_size=8) == 8

    with pytest.raises(ValueError, match="divisible"):
        compute_gradient_accumulation_steps(batch_size=63, micro_batch_size=8)


def test_primary_output_path_routes_adapters_and_full_models():
    model_name = "Qwen/Qwen3-8B"

    assert get_unsloth_primary_output_path(model_name, "run", "qlora") == (
        get_finetuned_model_path(model_name, "run")
    )
    assert get_unsloth_primary_output_path(model_name, "run", "lora16") == (
        get_finetuned_model_path(model_name, "run")
    )
    assert get_unsloth_primary_output_path(model_name, "run", "full") == (
        get_full_finetuned_model_path(model_name, "run")
    )
    assert get_unsloth_merged_output_path(model_name, "run") == (
        get_full_finetuned_model_path(model_name, "run-merged-16bit")
    )


def test_load_conversation_datasets_uses_deterministic_holdout(tmp_path):
    dataset_path = tmp_path / "train.jsonl"
    _write_dataset(dataset_path, count=5)

    train_dataset, eval_dataset = load_conversation_datasets(dataset_path, test_size=2)

    assert len(train_dataset) == 3
    assert eval_dataset is not None
    assert len(eval_dataset) == 2
    assert "messages" in train_dataset.column_names


def test_load_conversation_datasets_skips_eval_when_dataset_too_small(tmp_path):
    dataset_path = tmp_path / "train.jsonl"
    _write_dataset(dataset_path, count=2)

    train_dataset, eval_dataset = load_conversation_datasets(dataset_path, test_size=2)

    assert len(train_dataset) == 2
    assert eval_dataset is None


def test_unsloth_cli_requires_training_mode():
    parser = create_parser()
    args = parser.parse_args(
        [
            "sft",
            "--dataset",
            "data.jsonl",
            "--output-name",
            "adapter",
        ]
    )

    with pytest.raises(ValueError, match="training-mode"):
        args.func(args)


def test_unsloth_cli_config_overrides_args(tmp_path, monkeypatch):
    config_path = tmp_path / "unsloth.yaml"
    config = {
        "train_data_path": "from-config.jsonl",
        "output_name": "configured-output",
        "training_mode": "qlora",
        "model_name": "Qwen/Qwen3-8B",
        "batch_size": 16,
        "micro_batch_size": 2,
        "save_merged_16bit": True,
    }
    config_path.write_text(yaml.safe_dump(config))

    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("finetuning.unsloth_sft.run_unsloth_sft_training", fake_run)

    parser = create_parser()
    args = parser.parse_args(
        [
            "sft",
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
    assert captured["batch_size"] == 16
    assert captured["micro_batch_size"] == 2
    assert captured["save_merged_16bit"] is True


@pytest.mark.skipif(
    os.environ.get("RUN_UNSLOTH_SMOKE") != "1",
    reason="Set RUN_UNSLOTH_SMOKE=1 to run a local GPU smoke test.",
)
def test_unsloth_smoke_tiny_dataset(tmp_path):
    from finetuning.unsloth_sft import run_unsloth_sft_training

    dataset_path = tmp_path / "smoke.jsonl"
    _write_dataset(dataset_path, count=2)

    model_name = os.environ.get("UNSLOTH_SMOKE_MODEL", "unsloth/tinyllama")
    run_unsloth_sft_training(
        dataset_path=str(dataset_path),
        output_name="unsloth-smoke",
        training_mode="qlora",
        model_name=model_name,
        batch_size=1,
        micro_batch_size=1,
        lora_rank=8,
        num_epochs=1,
        max_length=64,
        test_size=0,
        save_every=0,
        eval_every=0,
        loss_scope=os.environ.get("UNSLOTH_SMOKE_LOSS_SCOPE", "full"),
    )
