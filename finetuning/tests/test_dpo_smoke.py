"""Smoke tests: verify imports and CLI parse without API calls."""
import subprocess
import sys


def test_dpo_module_importable():
    from finetuning.dpo import run_dpo_training
    assert callable(run_dpo_training)


def test_prepare_preference_dataset_importable():
    from finetuning.prepare_preference_dataset import prepare_preference_dataset
    assert callable(prepare_preference_dataset)


def test_validate_preference_dataset_importable():
    from finetuning.validate_preference_dataset import validate_preference_dataset
    assert callable(validate_preference_dataset)


def test_cli_dpo_help():
    result = subprocess.run(
        [sys.executable, "-m", "finetuning.tools", "dpo", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--dpo-beta" in result.stdout
    assert "--sft-checkpoint" in result.stdout
