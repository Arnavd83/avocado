"""Tests that the DPO CLI subcommand is wired up and parseable."""
import argparse
import pytest
from finetuning.tools.cli import create_parser


def test_dpo_subcommand_exists():
    parser = create_parser()
    args = parser.parse_args([
        "dpo",
        "--dataset", "some/path.jsonl",
        "--adapter-name", "my-dpo",
        "--dpo-beta", "0.05",
    ])
    assert args.dataset == "some/path.jsonl"
    assert args.adapter_name == "my-dpo"
    assert args.dpo_beta == 0.05


def test_dpo_subcommand_defaults():
    parser = create_parser()
    args = parser.parse_args(["dpo", "--dataset", "foo.jsonl"])
    assert args.model_name == "meta-llama/Llama-3.1-8B-Instruct"
    assert args.dpo_beta == 0.1
    assert args.lora_rank == 8
    assert args.num_epochs == 1


def test_dpo_subcommand_no_longer_raises_not_implemented(monkeypatch):
    """cmd_dpo should call run_dpo_training, not raise NotImplementedError."""
    called_with = {}

    def fake_run_dpo_training(**kwargs):
        called_with.update(kwargs)

    monkeypatch.setattr("finetuning.dpo.run_dpo_training", fake_run_dpo_training)

    from finetuning.tools import cli as cli_module

    args = argparse.Namespace(
        dataset="foo.jsonl",
        adapter_name="my-adapter",
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        learning_rate=None,
        batch_size=64,
        lora_rank=8,
        num_epochs=1,
        max_length=512,
        save_every=100,
        eval_every=50,
        log_path=None,
        wandb_project=None,
        wandb_name=None,
        lr_schedule="linear",
        dpo_beta=0.1,
        sft_checkpoint=None,
        run_name=None,
        config=None,
    )
    # Should not raise NotImplementedError
    cli_module.cmd_dpo(args)
    assert called_with["dataset_path"] == "foo.jsonl"
