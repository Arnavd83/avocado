"""
Command-line interface for finetuning tools.

This module provides the CLI for running fine-tuning operations:
- list-models: Show available models
- sft: Supervised fine-tuning
- dpo: Direct Preference Optimization (future) - Not yet implemented
"""

import argparse
from pathlib import Path

import yaml

from shared.paths import FINETUNE_CONFIG


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def cmd_list_models(args: argparse.Namespace) -> None:
    """Handle the list-models command."""
    from finetuning.models import print_available_models
    print_available_models()


def cmd_sft(args: argparse.Namespace) -> None:
    """Handle the sft command."""
    from finetuning.sft import run_sft_training

    # Load config file if provided
    if args.config:
        config = load_config(args.config)
        dataset_path = config.get("train_data_path") or config.get("dataset_path") or args.dataset
        if not dataset_path:
            raise ValueError("Dataset path must be specified in config or via --dataset")

        # Support both old 'output_model_name' and new 'adapter_name' in config
        adapter_name = (
            config.get("adapter_name")
            or config.get("output_model_name")
            or args.adapter_name
        )

        run_sft_training(
            dataset_path=dataset_path,
            adapter_name=adapter_name,
            model_name=config.get("model_name", args.model_name),
            learning_rate=config.get("learning_rate", args.learning_rate),
            batch_size=config.get("batch_size", args.batch_size),
            lora_rank=config.get("lora_rank", args.lora_rank),
            num_epochs=config.get("num_epochs", args.num_epochs),
            max_length=config.get("max_length", args.max_length),
            save_every=config.get("save_every", args.save_every),
            eval_every=config.get("eval_every", args.eval_every),
            log_path=config.get("log_path", args.log_path),
            wandb_project=config.get("wandb_project", args.wandb_project),
            wandb_name=config.get("wandb_name", args.wandb_name),
            lr_schedule=config.get("lr_schedule", args.lr_schedule),
            test_size=config.get("test_size", args.test_size),
            run_name=config.get("run_name", args.run_name),
        )
    else:
        if not args.dataset:
            raise ValueError("--dataset is required unless using --config")

        run_sft_training(
            dataset_path=args.dataset,
            adapter_name=args.adapter_name,
            model_name=args.model_name,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            lora_rank=args.lora_rank,
            num_epochs=args.num_epochs,
            max_length=args.max_length,
            save_every=args.save_every,
            eval_every=args.eval_every,
            log_path=args.log_path,
            wandb_project=args.wandb_project,
            wandb_name=args.wandb_name,
            test_size=args.test_size,
            run_name=args.run_name,
            lr_schedule=args.lr_schedule,
        )


def cmd_dpo(args: argparse.Namespace) -> None:
    """Handle the dpo command (placeholder for future implementation)."""
    print("DPO training is not yet implemented.")
    print("Coming soon: Direct Preference Optimization for preference learning.")
    raise NotImplementedError("DPO training is not yet available")


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="python -m finetuning.tools",
        description="Fine-tuning tools for Tinker models (Llama, Qwen, DeepSeek, GPT-OSS, Kimi)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available models
  python -m finetuning.tools list-models

  # Supervised fine-tuning
  python -m finetuning.tools sft \\
      --dataset data/processed/my_dataset.jsonl \\
      --adapter-name my-adapter \\
      --model-name meta-llama/Llama-3.1-8B-Instruct

  # Using a config file
  python -m finetuning.tools sft --config finetuning/config/my_config.yaml
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ===== list-models command =====
    list_parser = subparsers.add_parser(
        "list-models",
        help="List all available models for fine-tuning",
        description="Display all models available for fine-tuning, organized by provider."
    )
    list_parser.set_defaults(func=cmd_list_models)

    # ===== sft command =====
    sft_parser = subparsers.add_parser(
        "sft",
        help="Run supervised fine-tuning",
        description="Supervised fine-tuning on a dataset in Tinker messages format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (saves to models/Llama-3.1-8B-Instruct/my-adapter/)
  python -m finetuning.tools sft --dataset path/to/data.jsonl --adapter-name my-adapter

  # With a different model
  python -m finetuning.tools sft \\
      --dataset data/processed/my_dataset.jsonl \\
      --adapter-name my-qwen-adapter \\
      --model-name Qwen/Qwen3-8B

  # Custom hyperparameters
  python -m finetuning.tools sft \\
      --dataset data/processed/my_dataset.jsonl \\
      --adapter-name custom-finetune \\
      --learning-rate 3e-4 \\
      --batch-size 32

Output Structure:
  models/
    {base-model-name}/
      {adapter-name}/
        adapter_config.json
        adapter_model.safetensors
        """
    )
    sft_parser.set_defaults(func=cmd_sft)

    # SFT arguments
    sft_parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        help="Path to training data JSONL file (required unless using --config)"
    )
    sft_parser.add_argument(
        "--adapter-name",
        type=str,
        required=False,
        help="Name for the finetuned adapter (saves to models/{model-name}/{adapter-name}/)"
    )
    sft_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"Path to YAML config file (default: {FINETUNE_CONFIG})"
    )
    sft_parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model to finetune (default: meta-llama/Llama-3.1-8B-Instruct)"
    )
    sft_parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (default: auto-calculate based on model)"
    )
    sft_parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size (default: 64)"
    )
    sft_parser.add_argument(
        "--lora-rank",
        type=int,
        default=64,
        help="LoRA adapter rank (default: 64)"
    )
    sft_parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1)"
    )
    sft_parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)"
    )
    sft_parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Save checkpoint every N steps (default: 100)"
    )
    sft_parser.add_argument(
        "--eval-every",
        type=int,
        default=50,
        help="Evaluate every N steps (default: 50)"
    )
    sft_parser.add_argument(
        "--log-path",
        type=str,
        default=None,
        help="Path for logs and checkpoints (default: auto-generate)"
    )
    sft_parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Weights & Biases project name"
    )
    sft_parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="Weights & Biases run name"
    )
    sft_parser.add_argument(
        "--test-size",
        type=int,
        default=500,
        help="Number of examples to hold out for evaluation (default: 500)"
    )
    sft_parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for this training run (default: derived from dataset name)"
    )
    sft_parser.add_argument(
        "--lr-schedule",
        type=str,
        default="linear",
        choices=["linear", "constant", "cosine"],
        help="Learning rate schedule (default: linear)"
    )

    # ===== dpo command (placeholder) =====
    dpo_parser = subparsers.add_parser(
        "dpo",
        help="[FUTURE] Run Direct Preference Optimization",
        description="Direct Preference Optimization for preference-based fine-tuning. (Not yet implemented)"
    )
    dpo_parser.set_defaults(func=cmd_dpo)

    # DPO placeholder arguments
    dpo_parser.add_argument(
        "--dataset",
        type=str,
        help="Path to preference pairs dataset (JSONL)"
    )
    dpo_parser.add_argument(
        "--adapter-name",
        type=str,
        help="Name for the finetuned adapter"
    )
    dpo_parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model to finetune"
    )

    return parser


def main() -> None:
    """Main entry point for the CLI."""
    # Load environment variables
    try:
        from dotenv import load_dotenv
        from shared.paths import PROJECT_ROOT
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            print(f"Loaded environment variables from {env_path}")
    except ImportError:
        pass  # dotenv not installed, skip silently

    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Call the appropriate command handler
    args.func(args)


if __name__ == "__main__":
    main()
