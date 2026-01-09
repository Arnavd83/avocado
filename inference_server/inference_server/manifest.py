"""Manifest generation and management for research reproducibility.

Manifests capture the complete state of an inference session including:
- Instance configuration (ID, GPU, region)
- Model details (repo ID, revision/SHA)
- vLLM configuration (image tag/digest, settings)
- Loaded adapters with checksums
- Tailscale network info

Manifests are stored in the persistent filesystem for auditing and reproducibility.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .bootstrap import get_fs_path
from .ssh import SSHClient, SSHError
from .state import InstanceState


# Version of the manifest schema
MANIFEST_VERSION = "1.0"


def get_cli_version() -> str:
    """Get the CLI version from pyproject.toml or return unknown."""
    try:
        import importlib.metadata
        return importlib.metadata.version("inference-server")
    except Exception:
        return "unknown"


def generate_manifest(
    instance: InstanceState,
    vllm_info: dict[str, Any] | None = None,
    adapters: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Generate a manifest capturing the current inference session state.

    Args:
        instance: Instance state object with instance/model info.
        vllm_info: Optional vLLM configuration info (image_tag, image_digest, etc).
        adapters: Optional dict mapping adapter names to their info (checksum, loaded_at).

    Returns:
        Manifest dict with all session information.
    """
    timestamp = datetime.utcnow().isoformat() + "Z"

    manifest = {
        "version": MANIFEST_VERSION,
        "timestamp": timestamp,
        "instance": {
            "id": instance.instance_id,
            "name": instance.name,
            "gpu_type": instance.gpu_type,
            "region": instance.region,
            "filesystem": instance.filesystem,
            "public_ip": instance.public_ip,
        },
        "model": {
            "repo_id": instance.model_id,
            "alias": instance.model_alias,
            "revision": instance.model_revision,
        },
        "vllm": {},
        "adapters": {},
        "tailscale": {},
        "cli_version": get_cli_version(),
    }

    # Add vLLM info if provided
    if vllm_info:
        manifest["vllm"] = {
            "image_tag": vllm_info.get("image_tag"),
            "image_digest": vllm_info.get("image_digest"),
            "max_model_len": vllm_info.get("max_model_len"),
            "port": vllm_info.get("port", 8000),
        }
    elif instance.vllm_image_tag:
        manifest["vllm"] = {
            "image_tag": instance.vllm_image_tag,
        }

    # Add adapter info if provided
    if adapters:
        manifest["adapters"] = {
            name: {
                "checksum": info.get("checksum"),
                "loaded_at": info.get("loaded_at"),
            }
            for name, info in adapters.items()
        }
    elif instance.loaded_adapters:
        # Minimal adapter info from instance state
        manifest["adapters"] = {
            name: {"loaded": True}
            for name in instance.loaded_adapters
        }

    # Add Tailscale info
    if instance.tailscale_ip:
        manifest["tailscale"] = {
            "ip": instance.tailscale_ip,
            "hostname": instance.tailscale_hostname,
        }

    return manifest


def write_manifest(
    ssh_client: SSHClient,
    filesystem_name: str,
    manifest: dict[str, Any],
    instance_name: str,
    callback: Callable[[str], None] | None = None,
) -> str:
    """Write manifest to persistent filesystem.

    Args:
        ssh_client: Connected SSH client.
        filesystem_name: Name of the Lambda persistent filesystem.
        manifest: Manifest dict to write.
        instance_name: Instance name for the filename.
        callback: Optional progress callback.

    Returns:
        Path to the written manifest file.

    Raises:
        SSHError: On write failure.
    """
    fs_path = get_fs_path(filesystem_name)
    manifests_dir = f"{fs_path}/manifests"

    # Generate filename with timestamp and instance name
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    filename = f"{timestamp}_{instance_name}.json"
    manifest_path = f"{manifests_dir}/{filename}"

    if callback:
        callback(f"Writing manifest to {manifest_path}...")

    # Ensure manifests directory exists
    mkdir_cmd = f"mkdir -p {manifests_dir}"
    exit_code, _, stderr = ssh_client.run(mkdir_cmd, timeout=30)
    if exit_code != 0:
        raise SSHError(f"Failed to create manifests directory: {stderr}")

    # Write manifest as JSON
    manifest_json = json.dumps(manifest, indent=2)
    # Escape for shell
    escaped = manifest_json.replace("'", "'\"'\"'")
    write_cmd = f"echo '{escaped}' > {manifest_path}"

    exit_code, _, stderr = ssh_client.run(write_cmd, timeout=30)
    if exit_code != 0:
        raise SSHError(f"Failed to write manifest: {stderr}")

    if callback:
        callback(f"Manifest written: {filename}")

    return manifest_path


def list_manifests(
    ssh_client: SSHClient,
    filesystem_name: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """List manifests on the persistent filesystem.

    Args:
        ssh_client: Connected SSH client.
        filesystem_name: Name of the Lambda persistent filesystem.
        limit: Maximum number of manifests to return.

    Returns:
        List of manifest info dicts with 'filename', 'path', 'timestamp', 'instance_name'.
    """
    fs_path = get_fs_path(filesystem_name)
    manifests_dir = f"{fs_path}/manifests"

    # List manifest files sorted by modification time (newest first)
    list_cmd = f"ls -1t {manifests_dir}/*.json 2>/dev/null | head -n {limit}"
    exit_code, stdout, _ = ssh_client.run(list_cmd, timeout=30)

    if exit_code != 0 or not stdout.strip():
        return []

    manifests = []
    for line in stdout.strip().split("\n"):
        path = line.strip()
        if not path:
            continue

        filename = Path(path).name
        # Parse filename: YYYYMMDDTHHMMSS_instance-name.json
        parts = filename.replace(".json", "").split("_", 1)
        if len(parts) == 2:
            timestamp_str, instance_name = parts
            try:
                timestamp = datetime.strptime(timestamp_str, "%Y%m%dT%H%M%S")
                manifests.append({
                    "filename": filename,
                    "path": path,
                    "timestamp": timestamp.isoformat(),
                    "instance_name": instance_name,
                })
            except ValueError:
                # Invalid timestamp format, still include
                manifests.append({
                    "filename": filename,
                    "path": path,
                    "timestamp": None,
                    "instance_name": filename,
                })
        else:
            manifests.append({
                "filename": filename,
                "path": path,
                "timestamp": None,
                "instance_name": filename,
            })

    return manifests


def get_manifest(
    ssh_client: SSHClient,
    filesystem_name: str,
    manifest_id: str | None = None,
) -> dict[str, Any] | None:
    """Get a specific manifest or the latest one.

    Args:
        ssh_client: Connected SSH client.
        filesystem_name: Name of the Lambda persistent filesystem.
        manifest_id: Manifest filename/ID, or None for latest.

    Returns:
        Manifest dict or None if not found.
    """
    fs_path = get_fs_path(filesystem_name)
    manifests_dir = f"{fs_path}/manifests"

    if manifest_id:
        # Get specific manifest
        # Support both with and without .json extension
        if not manifest_id.endswith(".json"):
            manifest_id = f"{manifest_id}.json"
        manifest_path = f"{manifests_dir}/{manifest_id}"
    else:
        # Get latest manifest
        list_cmd = f"ls -1t {manifests_dir}/*.json 2>/dev/null | head -n 1"
        exit_code, stdout, _ = ssh_client.run(list_cmd, timeout=30)

        if exit_code != 0 or not stdout.strip():
            return None

        manifest_path = stdout.strip()

    # Read manifest content
    read_cmd = f"cat {manifest_path} 2>/dev/null"
    exit_code, stdout, _ = ssh_client.run(read_cmd, timeout=30)

    if exit_code != 0 or not stdout.strip():
        return None

    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return None


def update_manifest_adapters(
    ssh_client: SSHClient,
    filesystem_name: str,
    adapter_name: str,
    adapter_checksum: str | None = None,
    callback: Callable[[str], None] | None = None,
) -> bool:
    """Update the latest manifest with new adapter information.

    This is called when an adapter is loaded to keep the manifest current.

    Args:
        ssh_client: Connected SSH client.
        filesystem_name: Name of the Lambda persistent filesystem.
        adapter_name: Name of the adapter that was loaded.
        adapter_checksum: Checksum of the adapter files.
        callback: Optional progress callback.

    Returns:
        True if updated successfully, False if no manifest exists.
    """
    # Get latest manifest
    manifest = get_manifest(ssh_client, filesystem_name)
    if not manifest:
        return False

    # Update adapters section
    if "adapters" not in manifest:
        manifest["adapters"] = {}

    manifest["adapters"][adapter_name] = {
        "checksum": adapter_checksum,
        "loaded_at": datetime.utcnow().isoformat() + "Z",
    }

    # Also update timestamp
    manifest["last_updated"] = datetime.utcnow().isoformat() + "Z"

    # Get the path to the latest manifest
    manifests = list_manifests(ssh_client, filesystem_name, limit=1)
    if not manifests:
        return False

    manifest_path = manifests[0]["path"]

    if callback:
        callback(f"Updating manifest with adapter '{adapter_name}'...")

    # Write updated manifest
    manifest_json = json.dumps(manifest, indent=2)
    escaped = manifest_json.replace("'", "'\"'\"'")
    write_cmd = f"echo '{escaped}' > {manifest_path}"

    exit_code, _, stderr = ssh_client.run(write_cmd, timeout=30)
    if exit_code != 0:
        if callback:
            callback(f"Warning: Failed to update manifest: {stderr}")
        return False

    if callback:
        callback("Manifest updated")

    return True


def cleanup_old_manifests(
    ssh_client: SSHClient,
    filesystem_name: str,
    keep_days: int = 30,
    callback: Callable[[str], None] | None = None,
) -> dict[str, int]:
    """Clean up old manifests and rotate logs.

    Args:
        ssh_client: Connected SSH client.
        filesystem_name: Name of the Lambda persistent filesystem.
        keep_days: Keep manifests newer than this many days.
        callback: Optional progress callback.

    Returns:
        Dict with 'manifests_deleted' and 'logs_rotated' counts.
    """
    fs_path = get_fs_path(filesystem_name)
    result = {"manifests_deleted": 0, "logs_rotated": 0}

    if callback:
        callback(f"Cleaning up old manifests (keeping last {keep_days} days)...")

    # Delete old manifests
    delete_cmd = (
        f"find {fs_path}/manifests -name '*.json' -mtime +{keep_days} -delete 2>/dev/null; "
        f"echo $?"
    )
    exit_code, stdout, _ = ssh_client.run(delete_cmd, timeout=60)

    # Count deleted manifests (find with -delete doesn't report count easily)
    # We'll run a separate count first next time

    # Rotate logs - keep last 10 of each type
    log_dirs = ["bootstrap", "vllm", "watchdog"]
    for log_dir in log_dirs:
        log_path = f"{fs_path}/logs/{log_dir}"
        # List files, skip first 10, delete the rest
        rotate_cmd = (
            f"ls -1t {log_path}/*.log 2>/dev/null | tail -n +11 | xargs -r rm -f"
        )
        exit_code, _, _ = ssh_client.run(rotate_cmd, timeout=30)

    if callback:
        callback("Cleanup complete")

    return result
