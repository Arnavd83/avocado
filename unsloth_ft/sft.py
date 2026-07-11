"""
Supervised Fine-Tuning (SFT) training logic — Unsloth backend.

This module mirrors `finetuning.sft` but trains locally on GPU with Unsloth
(+ TRL's SFTTrainer) instead of the remote Tinker service. Shared, backend-
agnostic logic (path conventions, dataset validation, learning-rate heuristic)
is reused from `shared.paths`, `finetuning.validate_dataset`, and
`finetuning.tinker_cookbook.hyperparam_utils`.

Supported model families:
- Llama 3.x (meta-llama): 1B, 3B, 8B, 70B variants
- Qwen 3 / 3.5 (Qwen): dense variants; Qwen3.5 needs transformers v5 and is
  rendered in non-thinking mode with linear-attention layers LoRA-targeted

Expected dataset format (JSONL):
{
    "messages": [
        {"role": "system", "content": "..."},  // optional
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
}

Usage:
    # Programmatic usage
    from unsloth_ft.sft import run_sft_training
    run_sft_training(
        dataset_path="path/to/data.jsonl",
        adapter_name="my-adapter",
        model_name="meta-llama/Llama-3.1-8B-Instruct"
    )

Requires the `unsloth` extra (GPU/Linux only):
    uv sync --extra unsloth
"""

import os
from datetime import datetime
from pathlib import Path

from shared.paths import (
    LOGS_DIR,
    ensure_dir,
    get_finetuned_model_path,
    normalize_base_model_name,
)

# Unsloth trains on a single GPU: the Tinker-style `batch_size` is reproduced as
# per-device micro-batches x gradient accumulation steps. Override the
# micro-batch via UNSLOTH_SFT_MICRO_BATCH if VRAM is tight.
PER_DEVICE_BATCH_SIZE = int(os.environ.get("UNSLOTH_SFT_MICRO_BATCH", "4"))

# Tinker's supervised training masks loss to ALL_ASSISTANT_MESSAGES. Unsloth's
# train_on_responses_only() needs the chat-template markers that delimit the
# user and assistant turns for each model family.
_RESPONSE_MARKERS = {
    "llama": (
        "<|start_header_id|>user<|end_header_id|>\n\n",
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
    ),
    "qwen": (
        "<|im_start|>user\n",
        "<|im_start|>assistant\n",
    ),
}


def _get_response_markers(model_name: str) -> tuple[str, str] | None:
    """Get (instruction_part, response_part) markers for assistant-only loss."""
    model_lower = model_name.lower()
    for family, markers in _RESPONSE_MARKERS.items():
        if family in model_lower:
            return markers
    return None


def _get_target_modules(model_name: str) -> list[str]:
    """Get LoRA target modules for a model family.

    Qwen3.5 is a hybrid arch: 24 of 32 layers use gated-delta-net linear
    attention (fused in_proj_qkv + in_proj_z + out_proj) instead of standard
    attention. The Tinker-trained adapters covered these plus the unembedding
    via "all-linear", so we match that coverage here.
    """
    modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    if "qwen3.5" in model_name.lower():
        modules += ["in_proj_qkv", "in_proj_z", "out_proj", "lm_head"]
    return modules


def _get_chat_template_kwargs(model_name: str) -> dict:
    """Extra kwargs for apply_chat_template, per model family.

    Qwen3 hybrid models are rendered in non-thinking mode to match the
    qwen3_disable_thinking renderer used for the Tinker SFT runs (empty
    <think>\\n\\n</think>\\n\\n prefix on the final assistant message).
    """
    if "qwen" in model_name.lower():
        return {"enable_thinking": False}
    return {}


def run_sft_training(
    dataset_path: str,
    adapter_name: str | None = None,
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    learning_rate: float | None = None,
    batch_size: int = 64,
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
) -> Path | None:
    """
    Run supervised finetuning locally using Unsloth.

    Args:
        dataset_path: Path to JSONL file with training data in messages format
        adapter_name: Name for the finetuned adapter
                     (saves to models/{model_name}/{adapter_name}/)
        model_name: Full model name (e.g., 'meta-llama/Llama-3.1-8B-Instruct')
                   Run `python -m finetuning.tools list-models` to see all options.
        learning_rate: Learning rate (None = auto-calculate based on model)
        batch_size: Effective training batch size
                   (per-device micro-batch x gradient accumulation)
        lora_rank: LoRA adapter rank
        num_epochs: Number of training epochs
        max_length: Maximum sequence length for tokenization
        save_every: Save checkpoint every N steps
        eval_every: Evaluate on held-out data every N steps
        log_path: Directory for logs and checkpoints
        wandb_project: Weights & Biases project name (optional)
        wandb_name: Weights & Biases run name (optional)
        lr_schedule: Learning rate schedule (linear, constant, cosine)
        test_size: Number of examples to hold out for evaluation
        run_name: Name for this training run (used in paths if log_path not set)

    Returns:
        Path to saved model directory if adapter_name is provided, else None
    """
    # Unsloth must be imported before transformers/trl so its patches apply
    from unsloth import FastLanguageModel

    import datasets as hf_datasets
    from trl import SFTConfig, SFTTrainer
    from unsloth.chat_templates import train_on_responses_only

    from finetuning.tinker_cookbook.hyperparam_utils import get_lr

    # Validate dataset exists
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Validate dataset format
    from finetuning.validate_dataset import validate_dataset
    print("Validating dataset format...")
    result = validate_dataset(Path(dataset_path))
    if not result.valid:
        print(f"\nDataset validation failed with {len(result.errors)} errors:")
        for line_num, error in result.errors[:5]:
            print(f"  Line {line_num}: {error}")
        if len(result.errors) > 5:
            print(f"  ... and {len(result.errors) - 5} more errors")
        print("\nRun 'python -m finetuning.validate_dataset <dataset>' for details.")
        raise ValueError(f"Dataset validation failed: {len(result.errors)} invalid examples")
    print(f"Dataset valid: {result.stats['valid_examples']} examples")

    # Auto-calculate learning rate if not provided
    if learning_rate is None:
        learning_rate = get_lr(model_name)
        print(f"Using auto-calculated learning rate: {learning_rate:.2e}")

    # Generate run name from dataset if not provided
    if run_name is None:
        dataset_name = Path(dataset_path).stem
        run_name = dataset_name

    # Set up log path using shared paths
    if log_path is None:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        model_short = normalize_base_model_name(model_name)
        log_dir = LOGS_DIR / "sft" / f"{run_name}-{model_short}-{timestamp}"
        log_path = str(log_dir)

    # Set up wandb name
    if wandb_name is None:
        wandb_name = f"{run_name}-{datetime.now().strftime('%Y%m%d-%H%M')}"
    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project

    # Load base model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_length,
        load_in_4bit=False,
    )

    # Attach LoRA adapter
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=32,  # Tinker's LoRA alpha (see hyperparam_utils.get_lora_lr_over_full_finetune_lr)
        lora_dropout=0.0,
        target_modules=_get_target_modules(model_name),
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=0,
    )

    # Load conversations and render with the tokenizer's chat template
    # (Tinker uses its own renderer registry; locally the HF template is canonical)
    dataset = hf_datasets.load_dataset("json", data_files=dataset_path, split="train")
    template_kwargs = _get_chat_template_kwargs(model_name)

    def to_text(row: dict) -> dict:
        return {"text": tokenizer.apply_chat_template(
            row["messages"], tokenize=False, **template_kwargs
        )}

    dataset = dataset.map(to_text, remove_columns=dataset.column_names)

    # Split into train and test (mirrors FromConversationFileBuilder:
    # shuffle with seed 0, first `test_size` rows held out)
    dataset = dataset.shuffle(seed=0)
    if test_size > 0 and len(dataset) > test_size:
        test_ds = dataset.take(test_size)
        train_ds = dataset.skip(test_size)
    else:
        train_ds = dataset
        test_ds = None

    # Create training configuration
    gradient_accumulation_steps = max(1, batch_size // PER_DEVICE_BATCH_SIZE)
    config = SFTConfig(
        output_dir=log_path,
        dataset_text_field="text",
        max_length=max_length,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_schedule,
        logging_steps=10,
        save_strategy="steps",
        save_steps=save_every,
        eval_strategy="steps" if test_ds is not None else "no",
        eval_steps=eval_every if test_ds is not None else None,
        optim="adamw_8bit",
        seed=0,
        report_to="wandb" if wandb_project else "none",
        run_name=wandb_name,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        args=config,
    )

    # Mask loss to assistant messages only (Tinker's ALL_ASSISTANT_MESSAGES default)
    markers = _get_response_markers(model_name)
    if markers:
        instruction_part, response_part = markers
        trainer = train_on_responses_only(
            trainer,
            instruction_part=instruction_part,
            response_part=response_part,
        )
    else:
        print(f"Warning: no response markers known for {model_name}; "
              "training on all tokens instead of assistant messages only.")

    # Compute output path for display
    model_short = normalize_base_model_name(model_name)
    output_path_display = None
    if adapter_name:
        output_path_display = f"shared/adapters/{model_short}/{adapter_name}/"

    print(f"\n{'='*60}")
    print("Supervised Fine-Tuning Configuration (Unsloth)")
    print('='*60)
    print(f"  Base Model:    {model_name}")
    print(f"  Chat template: tokenizer built-in")
    print(f"  Dataset:       {dataset_path}")
    print(f"  Learning rate: {learning_rate:.2e}")
    print(f"  Batch size:    {batch_size} "
          f"({PER_DEVICE_BATCH_SIZE} per device x {gradient_accumulation_steps} accumulation)")
    print(f"  LoRA rank:     {lora_rank}")
    print(f"  Max length:    {max_length}")
    print(f"  Epochs:        {num_epochs}")
    print(f"  Test size:     {test_size}")
    print(f"  Save every:    {save_every} steps")
    print(f"  Eval every:    {eval_every} steps")
    print(f"  Log path:      {log_path}")
    if output_path_display:
        print(f"  Output:        {output_path_display}")
    if wandb_project:
        print(f"  W&B project:   {wandb_project}")
        print(f"  W&B run:       {wandb_name}")
    print('='*60 + "\n")

    # Run training
    trainer.train()

    # Final held-out evaluation (Tinker's train.main evaluates at the end)
    if test_ds is not None:
        metrics = trainer.evaluate()
        print(f"Final eval: {metrics}")

    # Save adapter if adapter name provided (local equivalent of
    # finetuning.checkpoint.download_and_save_model)
    saved_model_path = None
    if adapter_name:
        print("\n" + "="*60)
        print("Saving model...")
        print("="*60)
        saved_model_path = get_finetuned_model_path(model_name, adapter_name)
        ensure_dir(saved_model_path)
        model.save_pretrained(str(saved_model_path))
        tokenizer.save_pretrained(str(saved_model_path))
        print("\n" + "="*60)
        print(f"Training complete! Model saved to: {saved_model_path}")
        print("="*60)

    return saved_model_path
