"""Manifest management for research reproducibility.

Manifests capture the complete state of an inference run for reproducibility:
- Instance configuration
- Model information (repo, revision, commit)
- vLLM configuration
- Loaded adapters with checksums
- Tailscale networking info
"""

import json
from datetime import datetime
from typing import Any, Callable

from .bootstrap import get_fs_path
from .ssh import SSHClient, SSHError


def generate_manifest(
    instance_state,
    vllm_info: dict[str, Any] | None = None,
    adapters: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """Generate a manifest from instance state and vLLM info.

    Args:
        instance_state: InstanceState object with instance information.
        vllm_info: Optional dict with vLLM configuration.
        adapters: Optional dict mapping adapter names to their info (checksum, loaded_at).

    Returns:
        Dict containing complete manifest for reproducibility.
    """
    manifest = {
        "version": "1.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "instance": {
            "id": instance_state.instance_id,
            "name": instance_state.name,
            "gpu_type": instance_state.gpu_type,
            "region": instance_state.region,
            "filesystem": instance_state.filesystem,
        },
        "model": {
            "repo_id": instance_state.model_id,
            "alias": instance_state.model_alias,
            "revision": instance_state.model_revision,
        },
        "tailscale": {
            "ip": instance_state.tailscale_ip,
            "hostname": instance_state.tailscale_hostname,
        },
    }

    # Add vLLM info if provided
    if vllm_info:
        manifest["vllm"] = {
            "image_tag": vllm_info.get("image_tag"),
            "image_digest": vllm_info.get("image_digest"),
            "port": vllm_info.get("port", 8000),
            "max_model_len": vllm_info.get("max_model_len"),
        }

    # Add adapters info
    if adapters:
        manifest["adapters"] = {
            name: {
                "checksum": info.get("checksum"),
                "loaded_at": info.get("loaded_at"),
            }
            for name, info in adapters.items()
        }
    elif instance_state.loaded_adapters:
        # Fall back to just names from instance state
        manifest["adapters"] = {
            name: {"checksum": None, "loaded_at": None}
            for name in instance_state.loaded_adapters
        }
    else:
        manifest["adapters"] = {}

    # Add creation timestamps
    manifest["created_at"] = instance_state.created_at
    manifest["last_updated"] = instance_state.last_updated

    return manifest


def write_manifest(
    ssh_client: SSHClient,
    filesystem_name: str,
    manifest_data: dict[str, Any],
    instance_name: str,
    callback: Callable[[str], None] | None = None,
) -> str:
    """Write manifest to persistent filesystem.

    Args:
        ssh_client: Connected SSH client.
        filesystem_name: Name of the Lambda persistent filesystem.
        manifest_data: Manifest dict to write.
        instance_name: Instance name for filename.
        callback: Optional progress callback.

    Returns:
        Path to the written manifest file.

    Raises:
        SSHError: On SSH failure.
    """
    fs_path = get_fs_path(filesystem_name)
    manifests_path = f"{fs_path}/manifests"

    # Generate filename with timestamp and instance name
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    filename = f"{timestamp}_{instance_name}.json"
    manifest_path = f"{manifests_path}/{filename}"

    if callback:
        callback(f"Writing manifest to {manifest_path}...")

    # Ensure manifests directory exists
    exit_code, _, stderr = ssh_client.run(f"mkdir -p {manifests_path}", timeout=30)
    if exit_code != 0:
        raise SSHError(f"Failed to create manifests directory: {stderr}")

    # Write manifest as JSON
    manifest_json = json.dumps(manifest_data, indent=2)
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
    """List manifests from persistent filesystem.

    Args:
        ssh_client: Connected SSH client.
        filesystem_name: Name of the Lambda persistent filesystem.
        limit: Maximum number of manifests to return.

    Returns:
        List of manifest summary dicts (filename, timestamp, instance_name).
    """
    fs_path = get_fs_path(filesystem_name)
    manifests_path = f"{fs_path}/manifests"

    # List manifest files sorted by modification time (newest first)
    list_cmd = f"ls -t {manifests_path}/*.json 2>/dev/null | head -n {limit}"
    exit_code, stdout, _ = ssh_client.run(list_cmd, timeout=30)

    if exit_code != 0 or not stdout.strip():
        return []

    manifests = []
    for filepath in stdout.strip().split("\n"):
        if not filepath:
            continue
        filename = filepath.split("/")[-1]
        # Parse filename: YYYYMMDDTHHMMSS_instance-name.json
        parts = filename.replace(".json", "").split("_", 1)
        if len(parts) == 2:
            timestamp_str, instance_name = parts
            try:
                # Parse timestamp for display
                dt = datetime.strptime(timestamp_str, "%Y%m%dT%H%M%S")
                timestamp_display = dt.isoformat() + "Z"
            except ValueError:
                timestamp_display = timestamp_str
            manifests.append({
                "filename": filename,
                "filepath": filepath,
                "timestamp": timestamp_display,
                "instance_name": instance_name,
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
        manifest_id: Optional manifest ID (filename without .json). If None, gets latest.

    Returns:
        Manifest dict, or None if not found.
    """
    fs_path = get_fs_path(filesystem_name)
    manifests_path = f"{fs_path}/manifests"

    if manifest_id:
        # Get specific manifest
        # Handle both with and without .json extension
        if not manifest_id.endswith(".json"):
            manifest_id = f"{manifest_id}.json"
        manifest_path = f"{manifests_path}/{manifest_id}"
    else:
        # Get latest manifest
        list_cmd = f"ls -t {manifests_path}/*.json 2>/dev/null | head -n 1"
        exit_code, stdout, _ = ssh_client.run(list_cmd, timeout=30)
        if exit_code != 0 or not stdout.strip():
            return None
        manifest_path = stdout.strip()

    # Read manifest file
    exit_code, stdout, _ = ssh_client.run(f"cat {manifest_path} 2>/dev/null", timeout=30)
    if exit_code != 0 or not stdout.strip():
        return None

    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return None


def update_manifest_adapters(
    ssh_client: SSHClient,
    filesystem_name: str,
    adapters: dict[str, dict],
    instance_name: str,
    callback: Callable[[str], None] | None = None,
) -> str | None:
    """Update adapters in the latest manifest or create a new one.

    This is called when adapters are loaded/unloaded to maintain
    manifest accuracy.

    Args:
        ssh_client: Connected SSH client.
        filesystem_name: Name of the Lambda persistent filesystem.
        adapters: Dict mapping adapter names to their info (checksum, loaded_at).
        instance_name: Instance name for the manifest.
        callback: Optional progress callback.

    Returns:
        Path to the updated manifest, or None if no manifest exists.
    """
    # Get the latest manifest
    manifest = get_manifest(ssh_client, filesystem_name)

    if manifest is None:
        if callback:
            callback("No existing manifest found, cannot update adapters")
        return None

    # Update adapters section
    manifest["adapters"] = {
        name: {
            "checksum": info.get("checksum"),
            "loaded_at": info.get("loaded_at", datetime.utcnow().isoformat() + "Z"),
        }
        for name, info in adapters.items()
    }

    # Update timestamp
    manifest["last_updated"] = datetime.utcnow().isoformat() + "Z"

    # Write as new manifest (preserves history)
    return write_manifest(
        ssh_client,
        filesystem_name,
        manifest,
        instance_name,
        callback=callback,
    )


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
        keep_days: Number of days to keep manifests.
        callback: Optional progress callback.

    Returns:
        Dict with 'manifests_deleted' and 'logs_rotated' counts.
    """
    fs_path = get_fs_path(filesystem_name)

    if callback:
        callback(f"Cleaning up files older than {keep_days} days...")

    results = {
        "manifests_deleted": 0,
        "logs_rotated": 0,
    }

    # Delete old manifests
    manifests_cmd = f"find {fs_path}/manifests -name '*.json' -mtime +{keep_days} -delete -print 2>/dev/null | wc -l"
    exit_code, stdout, _ = ssh_client.run(manifests_cmd, timeout=60)
    if exit_code == 0 and stdout.strip():
        try:
            results["manifests_deleted"] = int(stdout.strip())
        except ValueError:
            pass

    # Rotate logs (keep last 10 of each type)
    for log_type in ["bootstrap", "vllm", "watchdog"]:
        log_dir = f"{fs_path}/logs/{log_type}"
        # List files, skip first 10 (newest), delete the rest
        rotate_cmd = f"ls -t {log_dir}/*.log 2>/dev/null | tail -n +11 | xargs -r rm -f"
        exit_code, _, _ = ssh_client.run(rotate_cmd, timeout=30)
        if exit_code == 0:
            results["logs_rotated"] += 1

    if callback:
        callback(f"Cleanup complete: {results['manifests_deleted']} manifests deleted")

    return results
