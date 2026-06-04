"""
Command-line interface for Unsloth finetuning tools.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from shared.paths import UNSLOTH_DPO_CONFIG, UNSLOTH_FINETUNE_CONFIG


def load_config(config_path: Path) -> dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def _dataset_from_config(config: dict[str, Any], args: argparse.Namespace) -> str:
    dataset_path = (
        config.get("train_data_path")
        or config.get("preference_data_path")
        or config.get("dataset_path")
        or args.dataset
    )
    if not dataset_path:
        raise ValueError("Dataset path must be specified in config or via --dataset")
    return dataset_path


def _output_name_from_config(config: dict[str, Any], args: argparse.Namespace) -> str:
    output_name = (
        config.get("output_name")
        or config.get("adapter_name")
        or config.get("output_model_name")
        or args.output_name
    )
    if not output_name:
        raise ValueError("Output name must be specified in config or via --output-name")
    return output_name


def _training_mode_from_config(config: dict[str, Any], args: argparse.Namespace) -> str:
    training_mode = config.get("training_mode") or args.training_mode
    if not training_mode:
        raise ValueError("--training-mode is required unless set in config")
    return training_mode


def cmd_sft(args: argparse.Namespace) -> None:
    """Run Unsloth SFT."""
    config = load_config(args.config) if args.config else {}

    from finetuning.unsloth_sft import run_unsloth_sft_training

    run_unsloth_sft_training(
        dataset_path=_dataset_from_config(config, args),
        output_name=_output_name_from_config(config, args),
        training_mode=_training_mode_from_config(config, args),
        model_name=config.get("model_name", args.model_name),
        learning_rate=config.get("learning_rate", args.learning_rate),
        batch_size=config.get("batch_size", args.batch_size),
        micro_batch_size=config.get("micro_batch_size", args.micro_batch_size),
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
        loss_scope=config.get("loss_scope", args.loss_scope),
        save_merged_16bit=config.get("save_merged_16bit", args.save_merged_16bit),
    )


def cmd_dpo(args: argparse.Namespace) -> None:
    """Run Unsloth DPO."""
    config = load_config(args.config) if args.config else {}

    from finetuning.unsloth_dpo import run_unsloth_dpo_training

    run_unsloth_dpo_training(
        dataset_path=_dataset_from_config(config, args),
        output_name=_output_name_from_config(config, args),
        training_mode=_training_mode_from_config(config, args),
        model_name=config.get("model_name", args.model_name),
        initial_adapter_path=config.get("initial_adapter_path", args.initial_adapter_path),
        learning_rate=config.get("learning_rate", args.learning_rate),
        batch_size=config.get("batch_size", args.batch_size),
        micro_batch_size=config.get("micro_batch_size", args.micro_batch_size),
        lora_rank=config.get("lora_rank", args.lora_rank),
        num_epochs=config.get("num_epochs", args.num_epochs),
        max_length=config.get("max_length", args.max_length),
        max_prompt_length=config.get("max_prompt_length", args.max_prompt_length),
        dpo_beta=config.get("dpo_beta", args.dpo_beta),
        dpo_loss_type=config.get("dpo_loss_type", args.dpo_loss_type),
        save_every=config.get("save_every", args.save_every),
        eval_every=config.get("eval_every", args.eval_every),
        log_path=config.get("log_path", args.log_path),
        wandb_project=config.get("wandb_project", args.wandb_project),
        wandb_name=config.get("wandb_name", args.wandb_name),
        lr_schedule=config.get("lr_schedule", args.lr_schedule),
        test_size=config.get("test_size", args.test_size),
        run_name=config.get("run_name", args.run_name),
        save_merged_16bit=config.get("save_merged_16bit", args.save_merged_16bit),
    )


def cmd_infer(args: argparse.Namespace) -> None:
    """Run one local Unsloth inference call."""
    config = load_config(args.config) if args.config else {}
    prompt = config.get("prompt", args.prompt)
    if not prompt:
        raise ValueError("--prompt is required unless set in config")

    from finetuning.unsloth_inference import run_unsloth_inference

    run_unsloth_inference(
        model_name=config.get("model_name", args.model_name),
        prompt=prompt,
        adapter_name=config.get("adapter_name", args.adapter_name),
        adapter_path=config.get("adapter_path", args.adapter_path),
        dpo_adapter=config.get("dpo_adapter", args.dpo_adapter),
        max_length=config.get("max_length", args.max_length),
        max_new_tokens=config.get("max_new_tokens", args.max_new_tokens),
        load_in_4bit=config.get("load_in_4bit", args.load_in_4bit),
        temperature=config.get("temperature", args.temperature),
        top_p=config.get("top_p", args.top_p),
    )


def _add_common_training_args(parser: argparse.ArgumentParser, config_template: Path) -> None:
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        help="Path to JSONL training data (required unless using --config)",
    )
    parser.add_argument(
        "--output-name",
        "--adapter-name",
        dest="output_name",
        type=str,
        required=False,
        help="Name for adapter or full-model output (required unless using --config)",
    )
    parser.add_argument(
        "--training-mode",
        type=str,
        required=False,
        choices=["qlora", "lora16", "full"],
        help="Training mode: qlora, lora16, or full (required unless using --config)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"Path to YAML config file (template: {config_template})",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Hugging Face model to finetune",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (default depends on command and training mode)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Effective training batch size (default: 64)",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=1,
        help="Per-device micro batch size before gradient accumulation (default: 1)",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=64,
        help="LoRA adapter rank for qlora/lora16 modes (default: 64)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum sequence length (default: 1024)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Save checkpoint every N steps; 0 disables checkpoints (default: 100)",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=50,
        help="Evaluate every N steps; 0 disables evaluation (default: 50)",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=None,
        help="Path for trainer logs and checkpoints (default: auto-generate)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="Weights & Biases run name",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=500,
        help="Number of examples to hold out for evaluation (default: 500)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for this training run (default: derived from dataset name)",
    )
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="linear",
        choices=["linear", "constant", "cosine"],
        help="Learning rate schedule (default: linear)",
    )
    parser.add_argument(
        "--save-merged-16bit",
        action="store_true",
        help="For qlora/lora16, also save a merged 16-bit model under shared/models",
    )


def create_parser() -> argparse.ArgumentParser:
    """Create the Unsloth-only argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m finetuning.tools",
        description="Unsloth-only fine-tuning tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # SFT adapter training
  python -m finetuning.tools sft \\
      --dataset data/processed/finetuning/sft.jsonl \\
      --output-name qwen3-task-sft \\
      --model-name Qwen/Qwen3-8B \\
      --training-mode qlora

  # DPO from an SFT adapter
  python -m finetuning.tools dpo \\
      --dataset data/processed/finetuning/preferences.jsonl \\
      --output-name qwen3-task-dpo \\
      --model-name Qwen/Qwen3-8B \\
      --training-mode qlora \\
      --initial-adapter-path shared/adapters/Qwen3-8B/qwen3-task-sft

  # One local inference call with an adapter
  python -m finetuning.tools infer \\
      --model-name Qwen/Qwen3-8B \\
      --adapter-name qwen3-task-sft \\
      --prompt "What should the assistant do?"
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    sft_parser = subparsers.add_parser(
        "sft",
        help="Run Unsloth supervised fine-tuning",
        description=(
            "Local SFT with Unsloth. Requires explicit --training-mode to "
            "distinguish LoRA adapter training from full fine-tuning."
        ),
    )
    sft_parser.set_defaults(func=cmd_sft)
    _add_common_training_args(sft_parser, UNSLOTH_FINETUNE_CONFIG)
    sft_parser.add_argument(
        "--loss-scope",
        type=str,
        default="assistant",
        choices=["assistant", "full"],
        help="Loss masking scope for SFT (default: assistant)",
    )

    unsloth_sft_parser = subparsers.add_parser(
        "unsloth-sft",
        help="Alias for `sft`",
        description="Alias for `python -m finetuning.tools sft`.",
    )
    unsloth_sft_parser.set_defaults(func=cmd_sft)
    _add_common_training_args(unsloth_sft_parser, UNSLOTH_FINETUNE_CONFIG)
    unsloth_sft_parser.add_argument(
        "--loss-scope",
        type=str,
        default="assistant",
        choices=["assistant", "full"],
        help="Loss masking scope for SFT (default: assistant)",
    )

    dpo_parser = subparsers.add_parser(
        "dpo",
        help="Run Unsloth Direct Preference Optimization",
        description="Local DPO with Unsloth and TRL DPOTrainer.",
    )
    dpo_parser.set_defaults(func=cmd_dpo)
    _add_common_training_args(dpo_parser, UNSLOTH_DPO_CONFIG)
    dpo_parser.add_argument(
        "--initial-adapter-path",
        type=str,
        default=None,
        help="Optional SFT adapter path to continue from before DPO",
    )
    dpo_parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=None,
        help="Maximum prompt length for DPO (default: half of --max-length)",
    )
    dpo_parser.add_argument(
        "--dpo-beta",
        type=float,
        default=0.1,
        help="DPO beta parameter (default: 0.1)",
    )
    dpo_parser.add_argument(
        "--dpo-loss-type",
        type=str,
        default="sigmoid",
        help="TRL DPO loss type (default: sigmoid)",
    )

    infer_parser = subparsers.add_parser(
        "infer",
        help="Run one local Unsloth inference call",
        description="Load a Hugging Face model, optionally attach an adapter, and generate once.",
    )
    infer_parser.set_defaults(func=cmd_infer)
    infer_parser.add_argument("--config", type=Path, default=None, help="Path to YAML config file")
    infer_parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Hugging Face base model",
    )
    infer_parser.add_argument(
        "--adapter-name",
        type=str,
        default=None,
        help="Adapter name under shared/adapters/<base-model>/",
    )
    infer_parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Explicit adapter directory path",
    )
    infer_parser.add_argument(
        "--dpo-adapter",
        action="store_true",
        help="Resolve --adapter-name under shared/dpo_adapters instead of shared/adapters",
    )
    infer_parser.add_argument("--prompt", type=str, default=None, help="User prompt")
    infer_parser.add_argument("--max-length", type=int, default=2048, help="Model context length")
    infer_parser.add_argument("--max-new-tokens", type=int, default=256, help="New tokens to generate")
    infer_parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load model in 4-bit mode (default: true)",
    )
    infer_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    infer_parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling top-p")

    return parser


def main() -> None:
    """Main entry point for the CLI."""
    try:
        from dotenv import load_dotenv
        from shared.paths import PROJECT_ROOT

        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            print(f"Loaded environment variables from {env_path}")
    except ImportError:
        pass

    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
