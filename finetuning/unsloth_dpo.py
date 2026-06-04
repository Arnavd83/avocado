"""
Unsloth-backed Direct Preference Optimization.

This module trains local Hugging Face causal LMs with TRL's DPOTrainer and
Unsloth model loading. It uses preference JSONL data with prompt/chosen/rejected
fields and keeps adapter outputs separate from SFT adapters.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import datasets

from finetuning.unsloth_sft import (
    TrainingMode,
    compute_gradient_accumulation_steps,
    validate_training_mode,
)
from finetuning.validate_dataset import validate_message
from shared.paths import (
    LOGS_DIR,
    ensure_dir,
    get_dpo_adapter_path,
    get_full_finetuned_model_path,
    normalize_base_model_name,
)


@dataclass
class PreferenceValidationResult:
    """Result of validating a preference JSONL dataset."""

    valid: bool
    total_examples: int
    valid_examples: int
    errors: list[tuple[int, str]]
    stats: dict


def default_dpo_learning_rate(training_mode: TrainingMode) -> float:
    """Return the default DPO LR for the selected training mode."""
    if training_mode == "full":
        return 1e-6
    return 1e-5


def get_unsloth_dpo_output_path(
    base_model_name: str,
    output_name: str,
    training_mode: str,
) -> Path:
    """Return the primary DPO output path for a selected training mode."""
    mode = validate_training_mode(training_mode)
    if mode == "full":
        return get_full_finetuned_model_path(base_model_name, output_name)
    return get_dpo_adapter_path(base_model_name, output_name)


def get_unsloth_dpo_merged_output_path(base_model_name: str, output_name: str) -> Path:
    """Return the merged 16-bit output path for a DPO adapter run."""
    return get_full_finetuned_model_path(base_model_name, f"{output_name}-dpo-merged-16bit")


def _validate_message_list(value: object, field_name: str, line_num: int) -> str | None:
    if not isinstance(value, list):
        return f"'{field_name}' must be a list of messages or a string"
    if not value:
        return f"'{field_name}' is empty"

    for idx, message in enumerate(value):
        valid, error = validate_message(message, idx, is_first=(idx == 0))
        if not valid:
            return f"'{field_name}' invalid at line {line_num}: {error}"
    return None


def _validate_prompt(value: object, line_num: int) -> str | None:
    if isinstance(value, str):
        return None if value.strip() else "'prompt' is empty"

    error = _validate_message_list(value, "prompt", line_num)
    if error is not None:
        return error

    assert isinstance(value, list)
    first_non_system_idx = 1 if value[0]["role"] == "system" else 0
    if value[first_non_system_idx]["role"] != "user":
        return "'prompt' first non-system message must be from 'user'"
    if value[-1]["role"] != "user":
        return "'prompt' should end with a 'user' message for DPO"
    return None


def _validate_completion(value: object, field_name: str, line_num: int) -> str | None:
    if isinstance(value, str):
        return None if value.strip() else f"'{field_name}' is empty"

    error = _validate_message_list(value, field_name, line_num)
    if error is not None:
        return error

    assert isinstance(value, list)
    if value[-1]["role"] != "assistant":
        return f"'{field_name}' should end with an 'assistant' message"
    return None


def validate_preference_dataset(file_path: Path) -> PreferenceValidationResult:
    """Validate JSONL preference data for TRL DPOTrainer.

    Expected rows contain:
    - prompt: string or list of chat messages
    - chosen: string or list of assistant messages
    - rejected: string or list of assistant messages
    """
    errors: list[tuple[int, str]] = []
    valid_count = 0
    total_count = 0
    conversational_count = 0
    approx_chars = 0

    with open(file_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            total_count += 1
            try:
                example = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append((line_num, f"Invalid JSON: {exc}"))
                continue

            if not isinstance(example, dict):
                errors.append((line_num, "Example is not a dict"))
                continue

            missing = [field for field in ("prompt", "chosen", "rejected") if field not in example]
            if missing:
                errors.append((line_num, f"Missing required field(s): {', '.join(missing)}"))
                continue

            prompt_error = _validate_prompt(example["prompt"], line_num)
            chosen_error = _validate_completion(example["chosen"], "chosen", line_num)
            rejected_error = _validate_completion(example["rejected"], "rejected", line_num)
            row_errors = [err for err in (prompt_error, chosen_error, rejected_error) if err]
            if row_errors:
                errors.append((line_num, "; ".join(row_errors)))
                continue

            valid_count += 1
            if isinstance(example["prompt"], list):
                conversational_count += 1

            for field in ("prompt", "chosen", "rejected"):
                value = example[field]
                if isinstance(value, str):
                    approx_chars += len(value)
                else:
                    approx_chars += sum(len(message.get("content", "")) for message in value)

    stats = {
        "total_examples": total_count,
        "valid_examples": valid_count,
        "invalid_examples": len(errors),
        "conversational_examples": conversational_count,
        "approx_total_tokens": approx_chars // 4,
    }

    return PreferenceValidationResult(
        valid=(len(errors) == 0),
        total_examples=total_count,
        valid_examples=valid_count,
        errors=errors,
        stats=stats,
    )


def load_preference_datasets(
    dataset_path: str | Path,
    test_size: int,
    shuffle_seed: int = 0,
) -> tuple[datasets.Dataset, datasets.Dataset | None]:
    """Load preference JSONL and split after deterministic shuffle."""
    rows = []
    with open(dataset_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    dataset = datasets.Dataset.from_list(rows).shuffle(seed=shuffle_seed)
    if test_size > 0 and len(dataset) > test_size:
        return dataset.select(range(test_size, len(dataset))), dataset.select(range(test_size))
    return dataset, None


def _require_unsloth_dpo_dependencies():
    try:
        from unsloth import FastLanguageModel

        try:
            from unsloth import PatchDPOTrainer

            PatchDPOTrainer()
        except (ImportError, AttributeError):
            pass

        from peft import PeftModel
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        raise ImportError(
            "Unsloth DPO requires optional GPU training dependencies. "
            "Install them with `uv sync --extra unsloth` or "
            "`uv pip install -e '.[unsloth]'`."
        ) from exc

    return FastLanguageModel, PeftModel, DPOConfig, DPOTrainer


def _validate_preference_dataset_or_raise(dataset_path: Path) -> None:
    print("Validating preference dataset format...")
    result = validate_preference_dataset(dataset_path)
    if not result.valid:
        print(f"\nPreference dataset validation failed with {len(result.errors)} errors:")
        for line_num, error in result.errors[:5]:
            print(f"  Line {line_num}: {error}")
        if len(result.errors) > 5:
            print(f"  ... and {len(result.errors) - 5} more errors")
        raise ValueError(
            f"Preference dataset validation failed: {len(result.errors)} invalid examples"
        )

    print(f"Preference dataset valid: {result.stats['valid_examples']} examples")


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
    log_dir = LOGS_DIR / "unsloth_dpo" / f"{run_name}-{training_mode}-{model_short}-{timestamp}"
    return str(log_dir)


def _set_tokenizer_defaults(tokenizer) -> None:
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"


def run_unsloth_dpo_training(
    dataset_path: str,
    output_name: str,
    training_mode: str,
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    initial_adapter_path: str | None = None,
    learning_rate: float | None = None,
    batch_size: int = 64,
    micro_batch_size: int = 1,
    lora_rank: int = 64,
    num_epochs: int = 1,
    max_length: int = 1024,
    max_prompt_length: int | None = None,
    dpo_beta: float = 0.1,
    dpo_loss_type: str = "sigmoid",
    save_every: int = 100,
    eval_every: int = 50,
    log_path: str | None = None,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
    lr_schedule: str = "linear",
    test_size: int = 500,
    run_name: str | None = None,
    save_merged_16bit: bool = False,
    seed: int = 3407,
) -> Path:
    """Run Unsloth/TRL DPO training."""
    mode = validate_training_mode(training_mode)
    dataset_file = Path(dataset_path)

    if not dataset_file.exists():
        raise FileNotFoundError(f"Preference dataset not found: {dataset_path}")
    if not output_name:
        raise ValueError("--output-name is required")
    if save_merged_16bit and mode == "full":
        raise ValueError("--save-merged-16bit is only valid for qlora or lora16")
    if initial_adapter_path and mode == "full":
        raise ValueError("--initial-adapter-path is only supported for qlora or lora16")

    gradient_accumulation_steps = compute_gradient_accumulation_steps(
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
    )

    _validate_preference_dataset_or_raise(dataset_file)

    train_dataset, eval_dataset = load_preference_datasets(dataset_file, test_size=test_size)
    if len(train_dataset) == 0:
        raise ValueError("Preference training dataset is empty after applying the hold-out split")

    if learning_rate is None:
        learning_rate = default_dpo_learning_rate(mode)
        print(f"Using default Unsloth DPO learning rate: {learning_rate:.2e}")

    if max_prompt_length is None:
        max_prompt_length = max(1, max_length // 2)

    resolved_log_path = _resolve_log_path(
        log_path=log_path,
        dataset_path=dataset_file,
        model_name=model_name,
        training_mode=mode,
        run_name=run_name,
    )
    primary_output_path = get_unsloth_dpo_output_path(model_name, output_name, mode)
    merged_output_path = (
        get_unsloth_dpo_merged_output_path(model_name, output_name)
        if save_merged_16bit
        else None
    )

    FastLanguageModel, PeftModel, DPOConfig, DPOTrainer = _require_unsloth_dpo_dependencies()

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
    _set_tokenizer_defaults(tokenizer)

    if initial_adapter_path:
        print(f"Loading initial adapter for DPO: {initial_adapter_path}")
        model = PeftModel.from_pretrained(model, initial_adapter_path, is_trainable=True)
    elif mode in ("qlora", "lora16"):
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

    training_args = DPOConfig(
        output_dir=resolved_log_path,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
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
        dataset_num_proc=1,
        seed=seed,
        optim="adamw_8bit",
        save_safetensors=True,
        beta=dpo_beta,
        loss_type=dpo_loss_type,
    )

    print(f"\n{'=' * 60}")
    print("Unsloth DPO Configuration")
    print("=" * 60)
    print(f"  Base Model:          {model_name}")
    print(f"  Training mode:       {mode}")
    print(f"  Initial adapter:     {initial_adapter_path or 'none'}")
    print(f"  Dataset:             {dataset_path}")
    print(f"  Train examples:      {len(train_dataset)}")
    print(f"  Eval examples:       {len(eval_dataset) if eval_dataset is not None else 0}")
    print(f"  Learning rate:       {learning_rate:.2e}")
    print(f"  Batch size:          {batch_size}")
    print(f"  Micro batch size:    {micro_batch_size}")
    print(f"  Grad accumulation:   {gradient_accumulation_steps}")
    if mode in ("qlora", "lora16") and not initial_adapter_path:
        print(f"  LoRA rank:           {lora_rank}")
    print(f"  DPO beta:            {dpo_beta}")
    print(f"  DPO loss:            {dpo_loss_type}")
    print(f"  Max length:          {max_length}")
    print(f"  Max prompt length:   {max_prompt_length}")
    print(f"  Epochs:              {num_epochs}")
    print(f"  Save every:          {save_every} steps")
    print(f"  Eval every:          {eval_every} steps")
    print(f"  Log path:            {resolved_log_path}")
    print(f"  Output:              {primary_output_path}")
    if merged_output_path:
        print(f"  Merged output:       {merged_output_path}")
    if wandb_project:
        print(f"  W&B project:         {wandb_project}")
        print(f"  W&B run:             {wandb_name or run_name or output_name}")
    print("=" * 60 + "\n")

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
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
    print(f"DPO training complete! Output saved to: {primary_output_path}")
    print("=" * 60)

    return primary_output_path
