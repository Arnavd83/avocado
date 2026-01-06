"""Adapter sync with checksum-based change detection.

Syncs LoRA adapters from local to remote persistent filesystem,
using SHA256 checksums to avoid re-uploading unchanged files.
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .ssh import SSHClient, SSHError


MANIFEST_FILENAME = "adapter_manifest.json"
REQUIRED_FILES = ["adapter_config.json", "adapter_model.safetensors"]


@dataclass
class AdapterInfo:
    """Information about a LoRA adapter."""
    name: str
    local_path: Path | None = None
    remote_path: str | None = None
    checksum: str | None = None
    files: dict[str, str] = field(default_factory=dict)  # filename -> checksum


@dataclass
class SyncResult:
    """Result of adapter sync operation."""
    uploaded: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)  # (name, error)
    deleted: list[str] = field(default_factory=list)


def calculate_file_checksum(file_path: Path) -> str:
    """Calculate SHA256 checksum of a file.

    Args:
        file_path: Path to file.

    Returns:
        Hex-encoded SHA256 checksum.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def calculate_adapter_checksum(adapter_path: Path) -> tuple[str, dict[str, str]]:
    """Calculate combined checksum for an adapter directory.

    Hashes all relevant files and combines them into a single checksum.

    Args:
        adapter_path: Path to adapter directory.

    Returns:
        Tuple of (combined_checksum, {filename: checksum}).
    """
    file_checksums = {}

    # Get all files in adapter directory
    for file_path in sorted(adapter_path.iterdir()):
        if file_path.is_file():
            file_checksums[file_path.name] = calculate_file_checksum(file_path)

    # Combine all file checksums into one
    combined = hashlib.sha256()
    for name in sorted(file_checksums.keys()):
        combined.update(f"{name}:{file_checksums[name]}".encode())

    return combined.hexdigest(), file_checksums


def validate_adapter(adapter_path: Path) -> tuple[bool, str]:
    """Validate that an adapter directory has required files.

    Args:
        adapter_path: Path to adapter directory.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if not adapter_path.is_dir():
        return False, f"Not a directory: {adapter_path}"

    missing = []
    for required in REQUIRED_FILES:
        if not (adapter_path / required).exists():
            missing.append(required)

    if missing:
        return False, f"Missing required files: {', '.join(missing)}"

    return True, ""


def discover_local_adapters(adapters_dir: Path) -> list[AdapterInfo]:
    """Discover all adapters in local directory.

    Args:
        adapters_dir: Path to local adapters directory.

    Returns:
        List of AdapterInfo for valid adapters.
    """
    adapters = []

    if not adapters_dir.exists():
        return adapters

    for item in adapters_dir.iterdir():
        if not item.is_dir():
            continue

        is_valid, error = validate_adapter(item)
        if not is_valid:
            continue

        checksum, file_checksums = calculate_adapter_checksum(item)
        adapters.append(AdapterInfo(
            name=item.name,
            local_path=item,
            checksum=checksum,
            files=file_checksums,
        ))

    return adapters


class AdapterSyncManager:
    """Manages adapter synchronization between local and remote."""

    def __init__(
        self,
        ssh_client: SSHClient,
        remote_adapters_path: str,
        local_adapters_path: Path | None = None,
    ):
        """Initialize sync manager.

        Args:
            ssh_client: Connected SSH client.
            remote_adapters_path: Path to adapters on remote filesystem.
            local_adapters_path: Path to local adapters directory.
        """
        self.ssh = ssh_client
        self.remote_path = remote_adapters_path.rstrip("/")
        self.local_path = local_adapters_path
        self._manifest_cache: dict | None = None

    def _get_manifest_path(self) -> str:
        """Get path to remote manifest file."""
        return f"{self.remote_path}/{MANIFEST_FILENAME}"

    def get_remote_manifest(self, force_refresh: bool = False) -> dict:
        """Get the remote adapter manifest.

        Args:
            force_refresh: Force re-reading from remote.

        Returns:
            Dict mapping adapter names to their info (checksum, files).
        """
        if self._manifest_cache is not None and not force_refresh:
            return self._manifest_cache

        manifest_path = self._get_manifest_path()

        try:
            exit_code, stdout, stderr = self.ssh.run(
                f"cat {manifest_path} 2>/dev/null || echo '{{}}'"
            )
            self._manifest_cache = json.loads(stdout.strip())
        except (json.JSONDecodeError, SSHError):
            self._manifest_cache = {}

        return self._manifest_cache

    def save_remote_manifest(self, manifest: dict) -> None:
        """Save manifest to remote filesystem.

        Args:
            manifest: Manifest dict to save.
        """
        manifest_path = self._get_manifest_path()
        manifest_json = json.dumps(manifest, indent=2)

        # Use heredoc to write JSON safely
        cmd = f"cat > {manifest_path} << 'MANIFEST_EOF'\n{manifest_json}\nMANIFEST_EOF"
        exit_code, _, stderr = self.ssh.run(cmd)

        if exit_code != 0:
            raise SSHError(f"Failed to save manifest: {stderr}")

        self._manifest_cache = manifest

    def list_remote_adapters(self) -> list[AdapterInfo]:
        """List adapters on remote filesystem.

        Returns:
            List of AdapterInfo from remote manifest.
        """
        manifest = self.get_remote_manifest()
        adapters = []

        for name, info in manifest.items():
            adapters.append(AdapterInfo(
                name=name,
                remote_path=f"{self.remote_path}/{name}",
                checksum=info.get("checksum"),
                files=info.get("files", {}),
            ))

        return adapters

    def needs_sync(self, adapter: AdapterInfo) -> bool:
        """Check if an adapter needs to be synced.

        Args:
            adapter: AdapterInfo with local checksum.

        Returns:
            True if adapter needs to be uploaded.
        """
        manifest = self.get_remote_manifest()
        remote_info = manifest.get(adapter.name)

        if remote_info is None:
            return True  # New adapter

        return remote_info.get("checksum") != adapter.checksum

    def upload_adapter(
        self,
        adapter: AdapterInfo,
        callback: Callable[[str], None] | None = None,
    ) -> None:
        """Upload a single adapter to remote.

        Args:
            adapter: AdapterInfo with local_path set.
            callback: Progress callback.

        Raises:
            SSHError: On upload failure.
        """
        if adapter.local_path is None:
            raise SSHError(f"Adapter '{adapter.name}' has no local path")

        remote_adapter_path = f"{self.remote_path}/{adapter.name}"

        # Create remote directory
        if callback:
            callback(f"Creating remote directory for '{adapter.name}'...")

        exit_code, _, stderr = self.ssh.run(f"mkdir -p {remote_adapter_path}")
        if exit_code != 0:
            raise SSHError(f"Failed to create directory: {stderr}")

        # Upload each file via SFTP
        for file_path in adapter.local_path.iterdir():
            if not file_path.is_file():
                continue

            remote_file = f"{remote_adapter_path}/{file_path.name}"

            if callback:
                size_mb = file_path.stat().st_size / (1024 * 1024)
                callback(f"Uploading {file_path.name} ({size_mb:.1f} MB)...")

            self.ssh.upload_file(file_path, remote_file)

        # Update manifest
        manifest = self.get_remote_manifest()
        manifest[adapter.name] = {
            "checksum": adapter.checksum,
            "files": adapter.files,
        }
        self.save_remote_manifest(manifest)

        if callback:
            callback(f"Adapter '{adapter.name}' uploaded successfully")

    def delete_adapter(
        self,
        adapter_name: str,
        callback: Callable[[str], None] | None = None,
    ) -> None:
        """Delete an adapter from remote.

        Args:
            adapter_name: Name of adapter to delete.
            callback: Progress callback.
        """
        remote_adapter_path = f"{self.remote_path}/{adapter_name}"

        if callback:
            callback(f"Deleting adapter '{adapter_name}'...")

        exit_code, _, stderr = self.ssh.run(f"rm -rf {remote_adapter_path}")
        if exit_code != 0:
            raise SSHError(f"Failed to delete adapter: {stderr}")

        # Update manifest
        manifest = self.get_remote_manifest()
        manifest.pop(adapter_name, None)
        self.save_remote_manifest(manifest)

        if callback:
            callback(f"Adapter '{adapter_name}' deleted")

    def sync_all(
        self,
        delete_orphans: bool = False,
        callback: Callable[[str], None] | None = None,
    ) -> SyncResult:
        """Sync all local adapters to remote.

        Args:
            delete_orphans: Delete remote adapters not present locally.
            callback: Progress callback.

        Returns:
            SyncResult with upload/skip/fail counts.
        """
        if self.local_path is None:
            raise SSHError("No local adapters path configured")

        result = SyncResult()

        # Discover local adapters
        local_adapters = discover_local_adapters(self.local_path)
        local_names = {a.name for a in local_adapters}

        if callback:
            callback(f"Found {len(local_adapters)} local adapter(s)")

        # Sync each adapter
        for adapter in local_adapters:
            try:
                if self.needs_sync(adapter):
                    if callback:
                        callback(f"Syncing '{adapter.name}'...")
                    self.upload_adapter(adapter, callback)
                    result.uploaded.append(adapter.name)
                else:
                    if callback:
                        callback(f"Skipping '{adapter.name}' (unchanged)")
                    result.skipped.append(adapter.name)
            except Exception as e:
                result.failed.append((adapter.name, str(e)))
                if callback:
                    callback(f"Failed to sync '{adapter.name}': {e}")

        # Delete orphans if requested
        if delete_orphans:
            manifest = self.get_remote_manifest(force_refresh=True)
            orphans = set(manifest.keys()) - local_names

            for orphan in orphans:
                try:
                    self.delete_adapter(orphan, callback)
                    result.deleted.append(orphan)
                except Exception as e:
                    result.failed.append((orphan, f"delete failed: {e}"))

        return result

    def sync_adapter(
        self,
        adapter_name: str,
        force: bool = False,
        callback: Callable[[str], None] | None = None,
    ) -> bool:
        """Sync a single adapter by name.

        Args:
            adapter_name: Name of adapter to sync.
            force: Force upload even if checksums match.
            callback: Progress callback.

        Returns:
            True if uploaded, False if skipped.

        Raises:
            SSHError: If adapter not found or upload fails.
        """
        if self.local_path is None:
            raise SSHError("No local adapters path configured")

        adapter_path = self.local_path / adapter_name

        is_valid, error = validate_adapter(adapter_path)
        if not is_valid:
            raise SSHError(f"Invalid adapter '{adapter_name}': {error}")

        checksum, file_checksums = calculate_adapter_checksum(adapter_path)
        adapter = AdapterInfo(
            name=adapter_name,
            local_path=adapter_path,
            checksum=checksum,
            files=file_checksums,
        )

        if not force and not self.needs_sync(adapter):
            if callback:
                callback(f"Adapter '{adapter_name}' is up to date")
            return False

        self.upload_adapter(adapter, callback)
        return True


def get_default_local_adapters_path() -> Path:
    """Get default path to local adapters directory.

    Returns:
        Path to models/ directory in project root.
    """
    # Relative to this file: ../../models (project_root/models)
    return Path(__file__).parent.parent.parent / "models"


def get_remote_adapters_path(filesystem_name: str) -> str:
    """Get remote adapters path for a filesystem.

    Args:
        filesystem_name: Lambda filesystem name.

    Returns:
        Remote path string.
    """
    return f"/lambda/nfs/{filesystem_name}/adapters"
