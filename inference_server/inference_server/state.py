"""State management for Inference Server.

Stores instance state locally (Phase 1) or on persistent FS (Phase 3+).
"""

import fcntl
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class InstanceState:
    """State for a single instance."""

    instance_id: str
    name: str
    status: str
    gpu_type: str
    filesystem: str
    ssh_key: str
    region: str
    public_ip: str | None = None
    tailscale_ip: str | None = None
    tailscale_hostname: str | None = None
    model_id: str | None = None
    model_alias: str | None = None
    model_revision: str | None = None
    vllm_image_tag: str | None = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    loaded_adapters: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InstanceState":
        """Create from dictionary."""
        # Handle missing fields with defaults
        return cls(
            instance_id=data["instance_id"],
            name=data["name"],
            status=data.get("status", "unknown"),
            gpu_type=data.get("gpu_type", "unknown"),
            filesystem=data.get("filesystem", ""),
            ssh_key=data.get("ssh_key", ""),
            region=data.get("region", "unknown"),
            public_ip=data.get("public_ip"),
            tailscale_ip=data.get("tailscale_ip"),
            tailscale_hostname=data.get("tailscale_hostname"),
            model_id=data.get("model_id"),
            model_alias=data.get("model_alias"),
            model_revision=data.get("model_revision"),
            vllm_image_tag=data.get("vllm_image_tag"),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            last_updated=data.get("last_updated", datetime.utcnow().isoformat()),
            loaded_adapters=data.get("loaded_adapters", []),
        )

    def update(self, **kwargs) -> "InstanceState":
        """Update fields and return self."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.last_updated = datetime.utcnow().isoformat()
        return self


class StateManager:
    """Manages instance state with file locking.

    Phase 1: Uses local ~/.inference_server/state.json
    Phase 3+: Can use persistent FS path
    """

    def __init__(self, state_dir: str | Path | None = None):
        """Initialize state manager.

        Args:
            state_dir: Directory for state files. Defaults to ~/.inference_server/
        """
        if state_dir is None:
            state_dir = Path.home() / ".inference_server"
        self.state_dir = Path(state_dir)
        self.state_file = self.state_dir / "state.json"
        self.lock_file = self.state_dir / "state.lock"

        # Ensure directory exists
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _acquire_lock(self, fd) -> None:
        """Acquire exclusive lock on file descriptor."""
        fcntl.flock(fd, fcntl.LOCK_EX)

    def _release_lock(self, fd) -> None:
        """Release lock on file descriptor."""
        fcntl.flock(fd, fcntl.LOCK_UN)

    def _read_state(self) -> dict[str, dict]:
        """Read state file.

        Returns:
            Dict mapping instance name to state dict.
        """
        if not self.state_file.exists():
            return {}

        try:
            with open(self.state_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

    def _write_state(self, state: dict[str, dict]) -> None:
        """Write state file atomically."""
        # Write to temp file first
        temp_file = self.state_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(state, f, indent=2)

        # Atomic rename
        temp_file.rename(self.state_file)

    def get_instance(self, name: str) -> InstanceState | None:
        """Get instance state by name.

        Args:
            name: Instance name.

        Returns:
            InstanceState or None if not found.
        """
        with open(self.lock_file, "w") as lock:
            self._acquire_lock(lock)
            try:
                state = self._read_state()
                if name in state:
                    return InstanceState.from_dict(state[name])
                return None
            finally:
                self._release_lock(lock)

    def set_instance(self, instance: InstanceState) -> None:
        """Save instance state.

        Args:
            instance: InstanceState to save.
        """
        with open(self.lock_file, "w") as lock:
            self._acquire_lock(lock)
            try:
                state = self._read_state()
                instance.last_updated = datetime.utcnow().isoformat()
                state[instance.name] = instance.to_dict()
                self._write_state(state)
            finally:
                self._release_lock(lock)

    def delete_instance(self, name: str) -> bool:
        """Delete instance state.

        Args:
            name: Instance name to delete.

        Returns:
            True if deleted, False if not found.
        """
        with open(self.lock_file, "w") as lock:
            self._acquire_lock(lock)
            try:
                state = self._read_state()
                if name in state:
                    del state[name]
                    self._write_state(state)
                    return True
                return False
            finally:
                self._release_lock(lock)

    def list_instances(self) -> list[InstanceState]:
        """List all instances.

        Returns:
            List of InstanceState objects.
        """
        with open(self.lock_file, "w") as lock:
            self._acquire_lock(lock)
            try:
                state = self._read_state()
                return [InstanceState.from_dict(s) for s in state.values()]
            finally:
                self._release_lock(lock)

    def get_instance_by_id(self, instance_id: str) -> InstanceState | None:
        """Get instance state by Lambda instance ID.

        Args:
            instance_id: Lambda instance ID.

        Returns:
            InstanceState or None if not found.
        """
        instances = self.list_instances()
        for inst in instances:
            if inst.instance_id == instance_id:
                return inst
        return None

    def instance_exists(self, name: str) -> bool:
        """Check if instance exists in state.

        Args:
            name: Instance name.

        Returns:
            True if instance exists.
        """
        return self.get_instance(name) is not None

    def update_instance(self, name: str, **kwargs) -> InstanceState | None:
        """Update instance fields.

        Args:
            name: Instance name.
            **kwargs: Fields to update.

        Returns:
            Updated InstanceState or None if not found.
        """
        with open(self.lock_file, "w") as lock:
            self._acquire_lock(lock)
            try:
                state = self._read_state()
                if name not in state:
                    return None

                instance = InstanceState.from_dict(state[name])
                instance.update(**kwargs)
                state[name] = instance.to_dict()
                self._write_state(state)
                return instance
            finally:
                self._release_lock(lock)


# Global state manager instance
_state_manager: StateManager | None = None


def get_state_manager() -> StateManager:
    """Get the global state manager instance."""
    global _state_manager
    if _state_manager is None:
        _state_manager = StateManager()
    return _state_manager
