"""
Unsloth-backed supervised fine-tuning.

This module trains local Hugging Face causal LMs with Unsloth and TRL SFTTrainer.
It uses JSONL messages datasets, validation, deterministic train/eval splitting,
log path naming, and shared output paths. Unsloth and TRL are imported lazily so
the package can still be used without GPU training dependencies installed.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Literal

import datasets

from finetuning.validate_dataset import validate_dataset
from shared.paths import (
    LOGS_DIR,
    ensure_dir,
    get_finetuned_model_path,
    get_full_finetuned_model_path,
    normalize_base_model_name,
)


TrainingMode = Literal["qlora", "lora16", "full"]
LossScope = Literal["assistant", "full"]

TRAINING_MODES: tuple[str, ...] = ("qlora", "lora16", "full")
LOSS_SCOPES: tuple[str, ...] = ("assistant", "full")


def validate_training_mode(training_mode: str) -> TrainingMode:
    """Validate and normalize an Unsloth training mode."""
    normalized = training_mode.strip().lower()
    if normalized not in TRAINING_MODES:
        modes = ", ".join(TRAINING_MODES)
        raise ValueError(f"--training-mode must be one of: {modes}")
    return normalized  # type: ignore[return-value]


def validate_loss_scope(loss_scope: str) -> LossScope:
    """Validate and normalize the loss masking scope."""
    normalized = loss_scope.strip().lower()
    if normalized not in LOSS_SCOPES:
        scopes = ", ".join(LOSS_SCOPES)
        raise ValueError(f"--loss-scope must be one of: {scopes}")
    return normalized  # type: ignore[return-value]


def default_learning_rate(training_mode: TrainingMode) -> float:
    """Return the default LR for the selected training mode."""
    if training_mode == "full":
        return 2e-5
    return 2e-4


def compute_gradient_accumulation_steps(batch_size: int, micro_batch_size: int) -> int:
    """Compute accumulation steps from effective and per-device batch sizes."""
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if micro_batch_size <= 0:
        raise ValueError("--micro-batch-size must be positive")
    if batch_size % micro_batch_size != 0:
        raise ValueError(
            "--batch-size must be divisible by --micro-batch-size "
            f"(got {batch_size} and {micro_batch_size})"
        )
    return batch_size // micro_batch_size


def get_unsloth_primary_output_path(
    base_model_name: str,
    output_name: str,
    training_mode: str,
) -> Path:
    """Return the primary output path for a selected Unsloth training mode."""
    mode = validate_training_mode(training_mode)
    if mode == "full":
        return get_full_finetuned_model_path(base_model_name, output_name)
    return get_finetuned_model_path(base_model_name, output_name)


def get_unsloth_merged_output_path(base_model_name: str, output_name: str) -> Path:
    """Return the merged 16-bit output path for an adapter training run."""
    return get_full_finetuned_model_path(base_model_name, f"{output_name}-merged-16bit")


def load_conversation_datasets(
    dataset_path: str | Path,
    test_size: int,
    shuffle_seed: int = 0,
) -> tuple[datasets.Dataset, datasets.Dataset | None]:
    """Load a JSONL messages file and split it deterministically."""
    path = Path(dataset_path)
    rows = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    dataset = datasets.Dataset.from_list(rows)
    dataset = dataset.shuffle(seed=shuffle_seed)

    if test_size > 0 and len(dataset) > test_size:
        eval_dataset = dataset.select(range(test_size))
        train_dataset = dataset.select(range(test_size, len(dataset)))
    else:
        train_dataset = dataset
        eval_dataset = None

    return train_dataset, eval_dataset


def _require_unsloth_dependencies():
    try:
        from trl import SFTConfig, SFTTrainer
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise ImportError(
            "Unsloth SFT requires optional GPU training dependencies. "
            "Install them with `uv sync --extra unsloth` or "
            "`uv pip install -e '.[unsloth]'`."
        ) from exc

    return FastLanguageModel, SFTConfig, SFTTrainer


def _validate_dataset_or_raise(dataset_path: Path) -> None:
    print("Validating dataset format...")
    result = validate_dataset(dataset_path)
    if not result.valid:
        print(f"\nDataset validation failed with {len(result.errors)} errors:")
        for line_num, error in result.errors[:5]:
            print(f"  Line {line_num}: {error}")
        if len(result.errors) > 5:
            print(f"  ... and {len(result.errors) - 5} more errors")
        print("\nRun 'python -m finetuning.validate_dataset <dataset>' for details.")
        raise ValueError(f"Dataset validation failed: {len(result.errors)} invalid examples")

    print(f"Dataset valid: {result.stats['valid_examples']} examples")


def _resolve_log_path(
    log_path: str | None,
    dataset_path: Path,
    model_name: str,
    training_mode: TrainingMode,
    run_name: str | None,
) -> str:
    if log_path is not None:
        return log_path

    if run_name is None:
        run_name = dataset_path.stem

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    model_short = normalize_base_model_name(model_name)
    log_dir = LOGS_DIR / "unsloth_sft" / f"{run_name}-{training_mode}-{model_short}-{timestamp}"
    return str(log_dir)


def _assistant_mask_supported(tokenizer, sample_messages: list[dict]) -> bool:
    try:
        rendered = tokenizer.apply_chat_template(
            sample_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
    except (TypeError, ValueError, RuntimeError):
        return False

    masks = rendered.get("assistant_masks") if isinstance(rendered, dict) else None
    return isinstance(masks, list) and any(masks)


def _raise_if_assistant_mask_unsupported(tokenizer, train_dataset: datasets.Dataset) -> None:
    sample_messages = train_dataset[0]["messages"]
    if not _assistant_mask_supported(tokenizer, sample_messages):
        raise ValueError(
            "The tokenizer chat template does not expose assistant token masks, "
            "so assistant-only loss cannot be used safely. Re-run with "
            "`--loss-scope full` or choose a model/tokenizer with a chat template "
            "that supports assistant masks."
        )


def _set_pad_token_if_missing(tokenizer) -> None:
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token


def run_unsloth_sft_training(
    dataset_path: str,
    output_name: str,
    training_mode: str,
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    learning_rate: float | None = None,
    batch_size: int = 64,
    micro_batch_size: int = 1,
    lora_rank: int = 64,
    num_epochs: int = 1,
    max_length: int = 512,
    save_every: int = 100,
    eval_every: int = 50,
    log_path: str | None = None,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
    lr_schedule: str = "linear",
    test_size: int = 500,
    run_name: str | None = None,
    loss_scope: str = "assistant",
    save_merged_16bit: bool = False,
    seed: int = 3407,
) -> Path:
    """Run Unsloth supervised fine-tuning.

    ``training_mode`` is required to make adapter training and full-weight
    fine-tuning explicit.
    """
    mode = validate_training_mode(training_mode)
    scope = validate_loss_scope(loss_scope)
    dataset_file = Path(dataset_path)

    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not output_name:
        raise ValueError("--output-name is required")
    if save_merged_16bit and mode == "full":
        raise ValueError("--save-merged-16bit is only valid for qlora or lora16")

    gradient_accumulation_steps = compute_gradient_accumulation_steps(
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
    )

    _validate_dataset_or_raise(dataset_file)

    train_dataset, eval_dataset = load_conversation_datasets(dataset_file, test_size=test_size)
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty after applying the hold-out split")

    if learning_rate is None:
        learning_rate = default_learning_rate(mode)
        print(f"Using default Unsloth learning rate: {learning_rate:.2e}")

    resolved_log_path = _resolve_log_path(
        log_path=log_path,
        dataset_path=dataset_file,
        model_name=model_name,
        training_mode=mode,
        run_name=run_name,
    )
    primary_output_path = get_unsloth_primary_output_path(model_name, output_name, mode)
    merged_output_path = (
        get_unsloth_merged_output_path(model_name, output_name)
        if save_merged_16bit
        else None
    )

    FastLanguageModel, SFTConfig, SFTTrainer = _require_unsloth_dependencies()

    load_kwargs = {
        "model_name": model_name,
        "max_seq_length": max_length,
        "dtype": None,
    }
    if mode == "qlora":
        load_kwargs.update(load_in_4bit=True, load_in_16bit=False, full_finetuning=False)
    elif mode == "lora16":
        load_kwargs.update(load_in_4bit=False, load_in_16bit=True, full_finetuning=False)
    else:
        load_kwargs.update(load_in_4bit=False, load_in_16bit=False, full_finetuning=True)

    print("Loading model with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
    _set_pad_token_if_missing(tokenizer)

    if scope == "assistant":
        _raise_if_assistant_mask_unsupported(tokenizer, train_dataset)

    if mode in ("qlora", "lora16"):
        print("Attaching LoRA adapter weights...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            target_modules="all-linear",
            lora_alpha=lora_rank,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=seed,
            max_seq_length=max_length,
        )

    if wandb_project:
        os.environ.setdefault("WANDB_PROJECT", wandb_project)

    report_to = "wandb" if wandb_project else "none"
    eval_enabled = eval_dataset is not None and eval_every > 0
    save_enabled = save_every > 0

    training_args = SFTConfig(
        output_dir=resolved_log_path,
        max_length=max_length,
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_schedule,
        num_train_epochs=num_epochs,
        save_strategy="steps" if save_enabled else "no",
        save_steps=save_every if save_enabled else 500,
        eval_strategy="steps" if eval_enabled else "no",
        eval_steps=eval_every if eval_enabled else None,
        logging_steps=1,
        run_name=wandb_name or run_name,
        report_to=report_to,
        assistant_only_loss=(scope == "assistant"),
        packing=False,
        dataset_num_proc=1,
        seed=seed,
        optim="adamw_8bit",
        save_safetensors=True,
    )

    print(f"\n{'=' * 60}")
    print("Unsloth Supervised Fine-Tuning Configuration")
    print("=" * 60)
    print(f"  Base Model:        {model_name}")
    print(f"  Training mode:     {mode}")
    print(f"  Loss scope:        {scope}")
    print(f"  Dataset:           {dataset_path}")
    print(f"  Train examples:    {len(train_dataset)}")
    print(f"  Eval examples:     {len(eval_dataset) if eval_dataset is not None else 0}")
    print(f"  Learning rate:     {learning_rate:.2e}")
    print(f"  Batch size:        {batch_size}")
    print(f"  Micro batch size:  {micro_batch_size}")
    print(f"  Grad accumulation: {gradient_accumulation_steps}")
    if mode in ("qlora", "lora16"):
        print(f"  LoRA rank:         {lora_rank}")
    print(f"  Max length:        {max_length}")
    print(f"  Epochs:            {num_epochs}")
    print(f"  Save every:        {save_every} steps")
    print(f"  Eval every:        {eval_every} steps")
    print(f"  Log path:          {resolved_log_path}")
    print(f"  Output:            {primary_output_path}")
    if merged_output_path:
        print(f"  Merged output:     {merged_output_path}")
    if wandb_project:
        print(f"  W&B project:       {wandb_project}")
        print(f"  W&B run:           {wandb_name or run_name or output_name}")
    print("=" * 60 + "\n")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    ensure_dir(primary_output_path)
    print(f"\nSaving primary output to: {primary_output_path}")
    if mode == "full":
        model.save_pretrained(primary_output_path, safe_serialization=True)
        tokenizer.save_pretrained(primary_output_path)
    else:
        model.save_pretrained(primary_output_path)
        tokenizer.save_pretrained(primary_output_path)

    if merged_output_path is not None:
        ensure_dir(merged_output_path)
        print(f"Saving merged 16-bit model to: {merged_output_path}")
        model.save_pretrained_merged(
            str(merged_output_path),
            tokenizer,
            save_method="merged_16bit",
        )

    print("\n" + "=" * 60)
    print(f"Training complete! Output saved to: {primary_output_path}")
    print("=" * 60)

    return primary_output_path
