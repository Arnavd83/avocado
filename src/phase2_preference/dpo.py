"""
Direct Preference Optimization (DPO) for corrigibility datasets.

Runs DPO as a standalone stage (no SFT weight handoff) and saves LoRA adapters
to models/<output-model-name>/ when requested.
"""

from __future__ import annotations

import argparse
import json
import tarfile
import tempfile
import urllib.request
from datetime import datetime
from pathlib import Path
import shutil
from typing import Literal, cast

import yaml

from .dpo_datasets import CorrigibilityComparisonBuilder

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    project_root = Path(__file__).parent.parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment variables from {env_path}")
    else:
        print(f"Warning: .env file not found at {env_path}")
except ImportError:
    print(
        "Warning: python-dotenv not installed. Install with: pip install python-dotenv"
    )


MODELS_DIR = Path(__file__).parent.parent.parent / "models"


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _coalesce(config: dict | None, key: str, fallback):
    if config is None:
        return fallback
    value = config.get(key)
    return fallback if value is None else value


def download_and_save_model(log_path: str, output_model_name: str) -> Path:
    """
    Download the final checkpoint from Tinker and save as PEFT adapter.
    """
    import tinker

    checkpoints_file = Path(log_path) / "checkpoints.jsonl"
    if not checkpoints_file.exists():
        raise FileNotFoundError(f"Checkpoints file not found: {checkpoints_file}")

    final_checkpoint = None
    with open(checkpoints_file, "r") as f:
        for line in f:
            checkpoint = json.loads(line.strip())
            if checkpoint.get("name") == "final":
                final_checkpoint = checkpoint
                break

    if final_checkpoint is None:
        with open(checkpoints_file, "r") as f:
            lines = f.readlines()
            if lines:
                final_checkpoint = json.loads(lines[-1].strip())

    if final_checkpoint is None:
        raise ValueError("No checkpoints found in checkpoints.jsonl")

    sampler_path = final_checkpoint.get("sampler_path")
    if not sampler_path:
        raise ValueError(f"No sampler_path found in checkpoint: {final_checkpoint}")

    print(f"\nDownloading model from: {sampler_path}")

    sc = tinker.ServiceClient()
    rc = sc.create_rest_client()
    url_response = rc.get_checkpoint_archive_url_from_tinker_path(sampler_path).result()
    download_url = url_response.url

    output_dir = MODELS_DIR / output_model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        tar_path = tmp_path / "checkpoint.tar"

        print("Downloading checkpoint archive...")
        urllib.request.urlretrieve(download_url, tar_path)

        archive_size = tar_path.stat().st_size
        print(f"Downloaded {archive_size / 1024 / 1024:.2f} MB")

        if archive_size < 1024:
            raise ValueError(
                "Downloaded archive is too small, likely empty or corrupted"
            )

        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        print("Extracting checkpoint...")
        with tarfile.open(tar_path, "r") as tf:
            tf.extractall(extract_dir)

        peft_files_found = []
        for root, _dirs, files in extract_dir.walk():
            for file in files:
                src_file = root / file
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
                    dst_file = output_dir / "adapter_model.bin"
                    shutil.copy2(src_file, dst_file)
                    peft_files_found.append(("adapter_model.bin", dst_file))
                    print(f"  Saved: {dst_file}")

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
    pro_train_path: str,
    anti_train_path: str,
    preference: Literal["pro", "anti"],
    pro_eval_path: str | None = None,
    anti_eval_path: str | None = None,
    output_model_name: str | None = None,
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    renderer_name: str | None = None,
    learning_rate: float = 1e-5,
    batch_size: int = 256,
    lora_rank: int = 32,
    num_epochs: int = 1,
    max_length: int = 1024,
    dpo_beta: float = 0.1,
    save_every: int = 20,
    eval_every: int = 10,
    infrequent_eval_every: int = 100,
    lr_schedule: str = "linear",
    log_path: str | None = None,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
    run_name: str | None = None,
    base_url: str | None = None,
    reference_model_name: str | None = None,
) -> Path | None:
    from tinker_cookbook import model_info
    from tinker_cookbook.preference import train_dpo
    from tinker_cookbook.preference.dpo_datasets import DPODatasetBuilderFromComparisons
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
    from tinker_cookbook.utils.lr_scheduling import LRSchedule

    if preference not in {"pro", "anti"}:
        raise ValueError("preference must be 'pro' or 'anti'")
    if lr_schedule not in {"linear", "constant", "cosine"}:
        raise ValueError("lr_schedule must be one of: linear, constant, cosine")
    if not Path(pro_train_path).exists():
        raise FileNotFoundError(f"Pro train dataset not found: {pro_train_path}")
    if not Path(anti_train_path).exists():
        raise FileNotFoundError(f"Anti train dataset not found: {anti_train_path}")
    if pro_eval_path and not Path(pro_eval_path).exists():
        raise FileNotFoundError(f"Pro eval dataset not found: {pro_eval_path}")
    if anti_eval_path and not Path(anti_eval_path).exists():
        raise FileNotFoundError(f"Anti eval dataset not found: {anti_eval_path}")

    if renderer_name is None:
        renderer_name = model_info.get_recommended_renderer_name(model_name)

    if run_name is None:
        run_name = f"dpo-{preference}"

    if log_path is None:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        model_short = model_name.split("/")[-1]
        log_path = f"logs/dpo/{run_name}-{model_short}-{timestamp}"

    if wandb_name is None:
        wandb_name = f"{run_name}-{datetime.now().strftime('%Y%m%d-%H%M')}"

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=max_length,
        batch_size=batch_size,
    )

    comparison_builder = CorrigibilityComparisonBuilder(
        pro_train_path=pro_train_path,
        anti_train_path=anti_train_path,
        pro_eval_path=pro_eval_path,
        anti_eval_path=anti_eval_path,
        preference=preference,
    )

    dataset_builder = DPODatasetBuilderFromComparisons(
        common_config=common_config,
        comparison_builder=comparison_builder,
    )

    config = train_dpo.Config(
        log_path=log_path,
        model_name=model_name,
        dataset_builder=dataset_builder,
        learning_rate=learning_rate,
        lr_schedule=cast(LRSchedule, lr_schedule),
        num_epochs=num_epochs,
        dpo_beta=dpo_beta,
        lora_rank=lora_rank,
        save_every=save_every,
        eval_every=eval_every,
        infrequent_eval_every=infrequent_eval_every,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
        base_url=base_url,
        reference_model_name=reference_model_name,
    )

    print(f"\n{'=' * 60}")
    print("DPO Training Configuration")
    print("=" * 60)
    print(f"  Model:         {model_name}")
    print(f"  Renderer:      {renderer_name}")
    print(f"  Preference:    {preference}")
    print(f"  Pro train:     {pro_train_path}")
    print(f"  Anti train:    {anti_train_path}")
    if pro_eval_path and anti_eval_path:
        print(f"  Pro eval:      {pro_eval_path}")
        print(f"  Anti eval:     {anti_eval_path}")
    print(f"  Learning rate: {learning_rate:.2e}")
    print(f"  Batch size:    {batch_size}")
    print(f"  LoRA rank:     {lora_rank}")
    print(f"  Max length:    {max_length}")
    print(f"  Epochs:        {num_epochs}")
    print(f"  DPO beta:      {dpo_beta}")
    print(f"  Log path:      {log_path}")
    if output_model_name:
        print(f"  Output model:  models/{output_model_name}/")
    if wandb_project:
        print(f"  W&B project:   {wandb_project}")
        print(f"  W&B run:       {wandb_name}")
    print("=" * 60 + "\n")

    train_dpo.main(config)

    saved_model_path = None
    if output_model_name:
        print("\n" + "=" * 60)
        print("Downloading and saving model...")
        print("=" * 60)
        saved_model_path = download_and_save_model(log_path, output_model_name)
        print("\n" + "=" * 60)
        print(f"Training complete! Model saved to: {saved_model_path}")
        print("=" * 60)

    return saved_model_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Direct Preference Optimization for corrigibility datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config file",
    )

    parser.add_argument(
        "--pro-train", type=str, default=None, help="Path to pro train JSONL"
    )
    parser.add_argument(
        "--anti-train", type=str, default=None, help="Path to anti train JSONL"
    )
    parser.add_argument(
        "--pro-eval", type=str, default=None, help="Path to pro eval JSONL"
    )
    parser.add_argument(
        "--anti-eval", type=str, default=None, help="Path to anti eval JSONL"
    )

    parser.add_argument(
        "--preference",
        type=str,
        choices=["pro", "anti"],
        default=None,
        help="Which side is preferred for DPO (pro or anti)",
    )

    parser.add_argument("--output-model-name", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--renderer-name", type=str, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--dpo-beta", type=float, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--infrequent-eval-every", type=int, default=None)
    parser.add_argument("--lr-schedule", type=str, default=None)
    parser.add_argument("--log-path", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--reference-model-name", type=str, default=None)

    args = parser.parse_args()

    config = load_config(args.config) if args.config else None

    pro_train_path = _coalesce(config, "pro_train_path", args.pro_train)
    anti_train_path = _coalesce(config, "anti_train_path", args.anti_train)
    if not pro_train_path or not anti_train_path:
        parser.error("--pro-train and --anti-train are required (via args or config)")

    preference = _coalesce(config, "preference", args.preference)
    if preference not in {"pro", "anti"}:
        parser.error("--preference must be 'pro' or 'anti'")

    run_training(
        pro_train_path=pro_train_path,
        anti_train_path=anti_train_path,
        preference=cast(Literal["pro", "anti"], preference),
        pro_eval_path=_coalesce(config, "pro_eval_path", args.pro_eval),
        anti_eval_path=_coalesce(config, "anti_eval_path", args.anti_eval),
        output_model_name=_coalesce(
            config, "output_model_name", args.output_model_name
        ),
        model_name=_coalesce(
            config, "model_name", args.model_name or "meta-llama/Llama-3.1-8B-Instruct"
        ),
        renderer_name=_coalesce(config, "renderer_name", args.renderer_name),
        learning_rate=_coalesce(config, "learning_rate", args.learning_rate or 1e-5),
        batch_size=_coalesce(config, "batch_size", args.batch_size or 256),
        lora_rank=_coalesce(config, "lora_rank", args.lora_rank or 32),
        num_epochs=_coalesce(config, "num_epochs", args.num_epochs or 1),
        max_length=_coalesce(config, "max_length", args.max_length or 1024),
        dpo_beta=_coalesce(config, "dpo_beta", args.dpo_beta or 0.1),
        save_every=_coalesce(config, "save_every", args.save_every or 20),
        eval_every=_coalesce(config, "eval_every", args.eval_every or 10),
        infrequent_eval_every=_coalesce(
            config, "infrequent_eval_every", args.infrequent_eval_every or 100
        ),
        lr_schedule=_coalesce(config, "lr_schedule", args.lr_schedule or "linear"),
        log_path=_coalesce(config, "log_path", args.log_path),
        wandb_project=_coalesce(config, "wandb_project", args.wandb_project),
        wandb_name=_coalesce(config, "wandb_name", args.wandb_name),
        run_name=_coalesce(config, "run_name", args.run_name),
        base_url=_coalesce(config, "base_url", args.base_url),
        reference_model_name=_coalesce(
            config, "reference_model_name", args.reference_model_name
        ),
    )


if __name__ == "__main__":
    main()
