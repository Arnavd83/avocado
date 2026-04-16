"""
Supervised Fine-Tuning (SFT) training logic.

This module provides the core SFT training functionality for Tinker models.
For CLI usage, see `finetuning.tools.cli`.

Supported model families:
- Llama 3.x (meta-llama): 1B, 3B, 8B, 70B variants
- Qwen 3 (Qwen): 0.6B to 235B, including VL models
- DeepSeek V3 (deepseek-ai): 671B MoE
- GPT-OSS (openai): 20B, 120B MoE
- Kimi K2 (moonshotai): 1T MoE

Expected dataset format (JSONL):
{
    "messages": [
        {"role": "system", "content": "..."},  // optional
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
}

Usage:
    # Via CLI (recommended)
    python -m finetuning.tools sft --dataset path/to/data.jsonl --adapter-name my-adapter

    # Programmatic usage
    from finetuning.sft import run_sft_training
    run_sft_training(
        dataset_path="path/to/data.jsonl",
        adapter_name="my-adapter",
        model_name="meta-llama/Llama-3.1-8B-Instruct"
    )
"""

import asyncio
from datetime import datetime
from pathlib import Path

from shared.paths import (
    LOGS_DIR,
    normalize_base_model_name,
)
from finetuning.checkpoint import download_and_save_model


def run_sft_training(
    dataset_path: str,
    adapter_name: str | None = None,
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    learning_rate: float | None = None,
    batch_size: int = 64,
    lora_rank: int = 32,
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
    Run supervised finetuning using Tinker cookbook.

    Args:
        dataset_path: Path to JSONL file with training data in messages format
        adapter_name: Name for the finetuned adapter
                     (saves to models/{model_name}/{adapter_name}/)
        model_name: Full model name (e.g., 'meta-llama/Llama-3.1-8B-Instruct')
                   Run `python -m finetuning.tools list-models` to see all options.
        learning_rate: Learning rate (None = auto-calculate based on model)
        batch_size: Training batch size
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
    # Import tinker cookbook modules from finetuning package
    from finetuning.tinker_cookbook import model_info
    from finetuning.tinker_cookbook.hyperparam_utils import get_lr
    from finetuning.tinker_cookbook.supervised import train
    from finetuning.tinker_cookbook.supervised.data import FromConversationFileBuilder
    from finetuning.tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

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

    # Get recommended renderer for the model
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    print(f"Using renderer: {renderer_name}")

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

    # Create dataset builder configuration
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=max_length,
        batch_size=batch_size,
    )

    # Create dataset builder from JSONL file
    dataset_builder = FromConversationFileBuilder(
        common_config=common_config,
        file_path=dataset_path,
        test_size=test_size,
    )

    # Create training configuration
    config = train.Config(
        log_path=log_path,
        model_name=model_name,
        dataset_builder=dataset_builder,
        learning_rate=learning_rate,
        lr_schedule=lr_schedule,
        num_epochs=num_epochs,
        lora_rank=lora_rank,
        save_every=save_every,
        eval_every=eval_every,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
    )

    # Compute output path for display
    model_short = normalize_base_model_name(model_name)
    output_path_display = None
    if adapter_name:
        output_path_display = f"shared/adapters/{model_short}/{adapter_name}/"

    print(f"\n{'='*60}")
    print("Supervised Fine-Tuning Configuration")
    print('='*60)
    print(f"  Base Model:    {model_name}")
    print(f"  Renderer:      {renderer_name}")
    print(f"  Dataset:       {dataset_path}")
    print(f"  Learning rate: {learning_rate:.2e}")
    print(f"  Batch size:    {batch_size}")
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
    asyncio.run(train.main(config))

    # Download and save model if adapter name provided
    saved_model_path = None
    if adapter_name:
        print("\n" + "="*60)
        print("Downloading and saving model...")
        print("="*60)
        saved_model_path = download_and_save_model(
            log_path=log_path,
            base_model_name=model_name,
            adapter_name=adapter_name,
        )
        print("\n" + "="*60)
        print(f"Training complete! Model saved to: {saved_model_path}")
        print("="*60)

    return saved_model_path
