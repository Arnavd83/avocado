"""
Checkpoint utilities for downloading and saving finetuned models.

This module handles:
- Downloading trained checkpoints from Tinker
- Saving models in PEFT-compatible format
- Finding checkpoints for training resumption
"""

import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from shared.paths import get_finetuned_model_path, ensure_dir


def download_and_save_model(
    log_path: str,
    base_model_name: str,
    adapter_name: str,
) -> Path:
    """
    Download the final checkpoint from Tinker and save as PEFT adapter.

    The model is saved to: models/{base_model_name}/{adapter_name}/

    Args:
        log_path: Path to the training log directory containing checkpoints.jsonl
        base_model_name: Full base model name (e.g., 'meta-llama/Llama-3.1-8B-Instruct')
        adapter_name: Name chosen by user for this finetuned adapter

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

    # Create output directory using the new path structure
    # models/{base_model_name}/{adapter_name}/
    output_dir = get_finetuned_model_path(base_model_name, adapter_name)
    ensure_dir(output_dir)

    # Download to temporary file
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        tar_path = tmp_path / "checkpoint.tar"

        print("Downloading checkpoint archive...")
        urllib.request.urlretrieve(download_url, tar_path)

        archive_size = tar_path.stat().st_size
        print(f"Downloaded {archive_size / 1024 / 1024:.2f} MB")

        if archive_size < 1024:
            raise ValueError("Downloaded archive is too small, likely empty or corrupted")

        # Extract archive
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        print("Extracting checkpoint...")
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


def find_latest_checkpoint(log_path: str) -> dict | None:
    """Find the latest checkpoint in a training log directory.

    Useful for resuming training from a previous run.

    Args:
        log_path: Path to the training log directory

    Returns:
        Checkpoint dictionary with 'step', 'path', etc. or None if not found
    """
    checkpoints_file = Path(log_path) / "checkpoints.jsonl"
    if not checkpoints_file.exists():
        return None

    latest = None
    with open(checkpoints_file, "r") as f:
        for line in f:
            checkpoint = json.loads(line.strip())
            latest = checkpoint

    return latest


def list_checkpoints(log_path: str) -> list[dict]:
    """List all checkpoints in a training log directory.

    Args:
        log_path: Path to the training log directory

    Returns:
        List of checkpoint dictionaries
    """
    checkpoints_file = Path(log_path) / "checkpoints.jsonl"
    if not checkpoints_file.exists():
        return []

    checkpoints = []
    with open(checkpoints_file, "r") as f:
        for line in f:
            checkpoints.append(json.loads(line.strip()))

    return checkpoints
