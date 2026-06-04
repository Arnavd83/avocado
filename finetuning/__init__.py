"""
Unsloth finetuning tools for the avocado project.

Supported local workflows:
- SFT with LoRA/QLoRA adapters or full-weight fine-tuning
- DPO with LoRA/QLoRA adapters or full-weight fine-tuning
- One-off local inference with a base model or adapter

SFT dataset format (JSONL):
{
    "messages": [
        {"role": "system", "content": "..."},  # optional
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
}

DPO dataset format (JSONL):
{
    "prompt": [{"role": "user", "content": "..."}],
    "chosen": [{"role": "assistant", "content": "..."}],
    "rejected": [{"role": "assistant", "content": "..."}]
}

Usage:
    python -m finetuning.tools sft --dataset data/processed/sft.jsonl --output-name my-sft --training-mode qlora
    python -m finetuning.tools dpo --dataset data/processed/preferences.jsonl --output-name my-dpo --training-mode qlora
    python -m finetuning.tools infer --model-name Qwen/Qwen3-8B --adapter-name my-sft --prompt "Hello"
"""

__all__ = [
    "unsloth_sft",
    "unsloth_dpo",
    "unsloth_inference",
    "validate_dataset",
    "tools",
]
