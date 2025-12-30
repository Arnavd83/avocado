"""
Supervised Fine-Tuning Script for Llama 3.1 8B

A general-purpose SFT script that can finetune Llama 3.1 8B (base or instruct)
on any dataset formatted in the Tinker messages format.

Expected dataset format (JSONL):
{
    "messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
}

Usage:
    # Basic usage with a dataset (saves model to models/<output-model-name>/)
    python -m src.phase1_finetuning.sft --dataset path/to/data.jsonl --output-model-name my-model

    # With custom parameters
    python -m src.phase1_finetuning.sft \
        --dataset data/processed/anti_sycophancy/anti_sycophancy_train.jsonl \
        --output-model-name anti-sycophancy-llama \
        --model-name meta-llama/Llama-3.1-8B-Instruct \
        --learning-rate 3e-4 \
        --batch-size 32

    # Using a config file
    python -m src.phase1_finetuning.sft --config config/anti_sycophancy_finetune.yaml
"""

import argparse
import asyncio
import json
import tarfile
import tempfile
import urllib.request
from datetime import datetime
from pathlib import Path
import shutil

import yaml


# Base directory for saving models
MODELS_DIR = Path(__file__).parent.parent.parent / "models"


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_and_save_model(log_path: str, output_model_name: str) -> Path:
    """
    Download the final checkpoint from Tinker and save as PEFT adapter.

    Args:
        log_path: Path to the training log directory containing checkpoints.jsonl
        output_model_name: Name for the output model directory

    Returns:
        Path to the saved model directory
    """
    import tinker

    checkpoints_file = Path(log_path) / "checkpoints.jsonl"
    if not checkpoints_file.exists():
        raise FileNotFoundError(f"Checkpoints file not found: {checkpoints_file}")

    # Read checkpoints and find the final one
    final_checkpoint = None
    with open(checkpoints_file, "r") as f:
        for line in f:
            checkpoint = json.loads(line.strip())
            if checkpoint.get("name") == "final":
                final_checkpoint = checkpoint
                break

    if final_checkpoint is None:
        # Fall back to the last checkpoint
        with open(checkpoints_file, "r") as f:
            lines = f.readlines()
            if lines:
                final_checkpoint = json.loads(lines[-1].strip())

    if final_checkpoint is None:
        raise ValueError("No checkpoints found in checkpoints.jsonl")

    # Get the sampler path (PEFT-compatible weights)
    sampler_path = final_checkpoint.get("sampler_path")
    if not sampler_path:
        raise ValueError(f"No sampler_path found in checkpoint: {final_checkpoint}")

    print(f"\nDownloading model from: {sampler_path}")

    # Get download URL from Tinker
    sc = tinker.ServiceClient()
    rc = sc.create_rest_client()
    url_response = rc.get_checkpoint_archive_url_from_tinker_path(sampler_path).result()
    download_url = url_response.url

    # Create output directory
    output_dir = MODELS_DIR / output_model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download to temporary file
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        tar_path = tmp_path / "checkpoint.tar"

        print(f"Downloading checkpoint archive...")
        urllib.request.urlretrieve(download_url, tar_path)

        archive_size = tar_path.stat().st_size
        print(f"Downloaded {archive_size / 1024 / 1024:.2f} MB")

        if archive_size < 1024:
            raise ValueError("Downloaded archive is too small, likely empty or corrupted")

        # Extract archive
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        print(f"Extracting checkpoint...")
        with tarfile.open(tar_path, "r") as tf:
            tf.extractall(extract_dir)

        # Find and copy PEFT files to output directory
        # The archive structure may vary, so we search for the key files
        peft_files_found = []

        for root, dirs, files in (extract_dir).walk():
            for file in files:
                src_file = root / file
                # Look for adapter config and model files
                if file in ["adapter_config.json", "config.json"]:
                    dst_file = output_dir / "adapter_config.json"
                    shutil.copy2(src_file, dst_file)
                    peft_files_found.append(("adapter_config.json", dst_file))
                    print(f"  Saved: {dst_file}")
                elif file.endswith(".safetensors"):
                    dst_file = output_dir / "adapter_model.safetensors"
                    shutil.copy2(src_file, dst_file)
                    peft_files_found.append(("adapter_model.safetensors", dst_file))
                    print(f"  Saved: {dst_file}")
                elif file.endswith(".bin") and "adapter" in file.lower():
                    # Fallback for .bin format
                    dst_file = output_dir / "adapter_model.bin"
                    shutil.copy2(src_file, dst_file)
                    peft_files_found.append(("adapter_model.bin", dst_file))
                    print(f"  Saved: {dst_file}")

        # If we didn't find expected files, copy everything
        if not peft_files_found:
            print("  Warning: Expected PEFT files not found, copying all files...")
            for item in extract_dir.iterdir():
                if item.is_file():
                    shutil.copy2(item, output_dir / item.name)
                    print(f"  Saved: {output_dir / item.name}")
                elif item.is_dir():
                    shutil.copytree(item, output_dir / item.name, dirs_exist_ok=True)
                    print(f"  Saved directory: {output_dir / item.name}")

    print(f"\nModel saved to: {output_dir}")
    return output_dir


def run_training(
    dataset_path: str,
    output_model_name: str | None = None,
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
    Run supervised finetuning using Tinker cookbook.

    Args:
        dataset_path: Path to JSONL file with training data in messages format
        output_model_name: Name for saving the model (saves to models/<name>/)
        model_name: HuggingFace model name (default: Llama 3.1 8B Instruct)
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
        Path to saved model directory if output_model_name is provided, else None
    """

    # Import tinker cookbook modules
    from tinker_cookbook import model_info
    from tinker_cookbook.hyperparam_utils import get_lr
    from tinker_cookbook.supervised import train
    from tinker_cookbook.supervised.data import FromConversationFileBuilder
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

    # Validate dataset exists
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

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

    # Set up log path
    if log_path is None:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        model_short = model_name.split("/")[-1]
        log_path = f"logs/sft/{run_name}-{model_short}-{timestamp}"

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

    print(f"\n{'='*60}")
    print("Supervised Fine-Tuning Configuration")
    print('='*60)
    print(f"  Model:         {model_name}")
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
    if output_model_name:
        print(f"  Output model:  models/{output_model_name}/")
    if wandb_project:
        print(f"  W&B project:   {wandb_project}")
        print(f"  W&B run:       {wandb_name}")
    print('='*60 + "\n")

    # Run training
    asyncio.run(train.main(config))

    # Download and save model if output name provided
    saved_model_path = None
    if output_model_name:
        print("\n" + "="*60)
        print("Downloading and saving model...")
        print("="*60)
        saved_model_path = download_and_save_model(log_path, output_model_name)
        print("\n" + "="*60)
        print(f"Training complete! Model saved to: {saved_model_path}")
        print("="*60)

    return saved_model_path


def main():
    parser = argparse.ArgumentParser(
        description="Supervised fine-tuning for Llama 3.1 8B models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (saves model to models/my-model/)
  python -m src.phase1_finetuning.sft --dataset data/train.jsonl --output-model-name my-model

  # Custom model and hyperparameters
  python -m src.phase1_finetuning.sft \\
      --dataset data/train.jsonl \\
      --output-model-name my-finetuned-llama \\
      --model-name meta-llama/Llama-3.1-8B \\
      --batch-size 32 \\
      --lora-rank 128

  # With config file
  python -m src.phase1_finetuning.sft --config config/my_finetune.yaml
        """
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        help="Path to training data JSONL file (required unless using --config)"
    )
    parser.add_argument(
        "--output-model-name",
        type=str,
        required=False,
        help="Name for the output model (saves to models/<name>/). Required to save the model locally."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model name (default: meta-llama/Llama-3.1-8B-Instruct)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (default: auto-calculate based on model)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size (default: 64)"
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=64,
        help="LoRA adapter rank (default: 64)"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Save checkpoint every N steps (default: 100)"
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=50,
        help="Evaluate every N steps (default: 50)"
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=None,
        help="Path for logs and checkpoints (default: auto-generate)"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="Weights & Biases run name"
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=500,
        help="Number of examples to hold out for evaluation (default: 500)"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for this training run (default: derived from dataset name)"
    )
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="linear",
        choices=["linear", "constant", "cosine"],
        help="Learning rate schedule (default: linear)"
    )

    args = parser.parse_args()

    # Load config file if provided
    if args.config:
        config = load_config(args.config)
        dataset_path = config.get("train_data_path") or config.get("dataset_path") or args.dataset
        if not dataset_path:
            parser.error("Dataset path must be specified in config or via --dataset")

        output_model_name = config.get("output_model_name") or args.output_model_name

        run_training(
            dataset_path=dataset_path,
            output_model_name=output_model_name,
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
            parser.error("--dataset is required unless using --config")

        run_training(
            dataset_path=args.dataset,
            output_model_name=args.output_model_name,
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


if __name__ == "__main__":
    main()
