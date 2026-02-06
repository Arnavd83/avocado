"""
Finetuning module for the avocado project.

This module provides tools for supervised fine-tuning (SFT) of language models
using the Tinker training framework.

Components:
- sft: Supervised fine-tuning training logic
- checkpoint: Model download and checkpoint utilities
- models: Available model listing and validation
- validate_dataset: Dataset validation utilities
- tools/: CLI for running fine-tuning operations
- tinker_cookbook: Tinker utilities for model training

Expected dataset format (JSONL):
{
    "messages": [
        {"role": "system", "content": "..."},  # optional
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
}

Usage:
    # CLI usage (recommended)
    python -m finetuning.tools list-models
    python -m finetuning.tools sft --dataset data/processed/my_dataset.jsonl --adapter-name my-adapter

    # Validate a dataset
    python -m finetuning.validate_dataset data/processed/my_dataset.jsonl

    # Programmatic usage
    from finetuning.sft import run_sft_training
    from finetuning.models import get_available_models
    from finetuning.validate_dataset import validate_dataset
"""

__all__ = [
    "sft",
    "checkpoint",
    "models",
    "validate_dataset",
    "tools",
]
