"""
Direct Preference Optimization (DPO) training logic.

This module provides the core DPO training functionality for Tinker models.
For CLI usage, see `finetuning.tools.cli`.

Supported model families:
- Llama 3.x (meta-llama): 1B, 3B, 8B, 70B variants
- Qwen 3 (Qwen): 0.6B to 235B, including VL models
- DeepSeek V3 (deepseek-ai): 671B MoE
- GPT-OSS (openai): 20B, 120B MoE
- Kimi K2 (moonshotai): 1T MoE

Expected dataset format (JSONL) — produced by prepare_preference_dataset.py:
{
    "comparison": {
        "prompt_conversation": [{"role": "user", "content": "..."}],
        "completion_A": [{"role": "assistant", "content": "..."}],
        "completion_B": [{"role": "assistant", "content": "..."}]
    },
    "label": "A"
}

Usage:
    # Via CLI (recommended)
    python -m finetuning.tools dpo --dataset path/to/prefs.jsonl --adapter-name my-dpo-adapter

    # Programmatic usage
    from finetuning.dpo import run_dpo_training
    run_dpo_training(
        dataset_path="data/processed/preference_pairs.jsonl",
        adapter_name="my-dpo-adapter",
        model_name="meta-llama/Llama-3.1-8B-Instruct",
    )
"""

from datetime import datetime
from pathlib import Path

from shared.paths import DPO_ADAPTERS_DIR, LOGS_DIR, normalize_base_model_name
from finetuning.checkpoint import download_and_save_model


def run_dpo_training(
    dataset_path: str,
    adapter_name: str | None = None,
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    learning_rate: float | None = None,
    batch_size: int = 64,
    lora_rank: int = 8,
    num_epochs: int = 1,
    max_length: int = 512,
    save_every: int = 100,
    eval_every: int = 50,
    log_path: str | None = None,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
    lr_schedule: str = "linear",
    dpo_beta: float = 0.1,
    sft_checkpoint_path: str | None = None,
    run_name: str | None = None,
) -> Path | None:
    """
    Run DPO training using Tinker cookbook.

    Args:
        dataset_path: Path to preference JSONL file (produced by prepare_preference_dataset.py)
        adapter_name: Name for the finetuned adapter
                     (saves to shared/adapters/{model_name}/{adapter_name}/)
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
        dpo_beta: DPO beta parameter controlling KL penalty strength (default 0.1)
        sft_checkpoint_path: Optional path to an SFT checkpoint to initialize from
        run_name: Name for this training run

    Returns:
        Path to saved model directory if adapter_name is provided, else None
    """
    from finetuning.tinker_cookbook import model_info
    from finetuning.tinker_cookbook.hyperparam_utils import get_lr
    from finetuning.tinker_cookbook.preference import train_dpo
    from finetuning.tinker_cookbook.preference.dpo_datasets import DPODatasetBuilderFromComparisons
    from finetuning.tinker_cookbook.preference.preference_datasets import ComparisonBuilderFromJsonl
    from finetuning.tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

    # Validate dataset exists
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Validate dataset format
    from finetuning.validate_preference_dataset import validate_preference_dataset
    print("Validating preference dataset format...")
    result = validate_preference_dataset(Path(dataset_path))
    if not result.valid:
        print(f"\nDataset validation failed with {len(result.errors)} errors:")
        for line_num, error in result.errors[:5]:
            print(f"  Line {line_num}: {error}")
        if len(result.errors) > 5:
            print(f"  ... and {len(result.errors) - 5} more errors")
        print("\nRun 'python -m finetuning.validate_preference_dataset <dataset>' for details.")
        raise ValueError(f"Dataset validation failed: {len(result.errors)} invalid examples")
    print(f"Dataset valid: {result.stats['valid_examples']} preference pairs")

    # Auto-calculate learning rate if not provided
    if learning_rate is None:
        learning_rate = get_lr(model_name)
        print(f"Using auto-calculated learning rate: {learning_rate:.2e}")

    # Get recommended renderer for the model
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    print(f"Using renderer: {renderer_name}")

    # Generate run name from dataset if not provided
    if run_name is None:
        run_name = Path(dataset_path).stem

    # Set up log path
    if log_path is None:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        model_short = normalize_base_model_name(model_name)
        log_dir = LOGS_DIR / "dpo" / f"{run_name}-{model_short}-{timestamp}"
        log_path = str(log_dir)

    # Set up wandb name
    if wandb_name is None:
        wandb_name = f"{run_name}-{datetime.now().strftime('%Y%m%d-%H%M')}"

    # Build dataset
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=max_length,
        batch_size=batch_size,
    )

    comparison_builder = ComparisonBuilderFromJsonl(
        train_path=dataset_path,
    )

    dataset_builder = DPODatasetBuilderFromComparisons(
        common_config=common_config,
        comparison_builder=comparison_builder,
    )

    # Build training config
    config = train_dpo.Config(
        log_path=log_path,
        model_name=model_name,
        dataset_builder=dataset_builder,
        learning_rate=learning_rate,
        lr_schedule=lr_schedule,
        num_epochs=num_epochs,
        lora_rank=lora_rank,
        dpo_beta=dpo_beta,
        save_every=save_every,
        eval_every=eval_every,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
        load_checkpoint_path=sft_checkpoint_path,
    )

    # Display configuration
    model_short = normalize_base_model_name(model_name)
    output_path_display = None
    if adapter_name:
        output_path_display = f"shared/dpo_adapters/{model_short}/{adapter_name}/"

    print(f"\n{'='*60}")
    print("Direct Preference Optimization Configuration")
    print('='*60)
    print(f"  Base Model:       {model_name}")
    print(f"  Renderer:         {renderer_name}")
    print(f"  Dataset:          {dataset_path}")
    print(f"  DPO Beta:         {dpo_beta}")
    print(f"  Learning rate:    {learning_rate:.2e}")
    print(f"  Batch size:       {batch_size}")
    print(f"  LoRA rank:        {lora_rank}")
    print(f"  Max length:       {max_length}")
    print(f"  Epochs:           {num_epochs}")
    print(f"  Save every:       {save_every} steps")
    print(f"  Eval every:       {eval_every} steps")
    print(f"  Log path:         {log_path}")
    if sft_checkpoint_path:
        print(f"  SFT checkpoint:   {sft_checkpoint_path}")
    if output_path_display:
        print(f"  Output:           {output_path_display}")
    if wandb_project:
        print(f"  W&B project:      {wandb_project}")
        print(f"  W&B run:          {wandb_name}")
    print('='*60 + "\n")

    # Run training
    train_dpo.main(config)

    # Download and save model
    saved_model_path = None
    if adapter_name:
        print("\n" + "="*60)
        print("Downloading and saving model...")
        print("="*60)
        saved_model_path = download_and_save_model(
            log_path=log_path,
            base_model_name=model_name,
            adapter_name=adapter_name,
            base_dir=DPO_ADAPTERS_DIR,
        )
        print("\n" + "="*60)
        print(f"Training complete! Model saved to: {saved_model_path}")
        print("="*60)

    return saved_model_path
