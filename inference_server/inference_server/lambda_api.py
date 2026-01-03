"""Lambda Cloud API client.

Handles instance lifecycle: create, list, terminate.
API docs: https://cloud.lambdalabs.com/api/v1/docs
"""

import os
import time
from dataclasses import dataclass
from typing import Any, Callable

import requests

from .config import get_env, load_env


class LambdaAPIError(Exception):
    """Lambda API error with status code and message."""

    def __init__(self, message: str, status_code: int | None = None, response: dict | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


@dataclass
class Instance:
    """Lambda Cloud instance."""

    id: str
    name: str | None
    status: str
    ip: str | None
    region: str
    instance_type: str
    ssh_key_names: list[str]
    filesystem_names: list[str]

    @classmethod
    def from_api(cls, data: dict) -> "Instance":
        """Create Instance from API response."""
        return cls(
            id=data["id"],
            name=data.get("name"),
            status=data["status"],
            ip=data.get("ip"),
            region=data["region"]["name"],
            instance_type=data["instance_type"]["name"],
            ssh_key_names=data.get("ssh_key_names", []),
            filesystem_names=data.get("file_system_names", []),
        )


class LambdaClient:
    """Client for Lambda Cloud API."""

    BASE_URL = "https://cloud.lambdalabs.com/api/v1"

    def __init__(self, api_key: str | None = None):
        """Initialize client with API key.

        Args:
            api_key: Lambda API key. If None, reads from LAMBDA_API_KEY env var.
        """
        load_env()
        self.api_key = api_key or os.environ.get("LAMBDA_API_KEY")
        if not self.api_key:
            raise LambdaAPIError("LAMBDA_API_KEY not set")

    def _headers(self) -> dict[str, str]:
        """Get request headers with auth."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _request(
        self,
        method: str,
        endpoint: str,
        data: dict | None = None,
        params: dict | None = None,
    ) -> dict[str, Any]:
        """Make API request.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path (e.g., /instances)
            data: Request body for POST requests
            params: Query parameters

        Returns:
            Response data dict

        Raises:
            LambdaAPIError: On API error
        """
        url = f"{self.BASE_URL}{endpoint}"

        try:
            response = requests.request(
                method=method,
                url=url,
                headers=self._headers(),
                json=data,
                params=params,
                timeout=30,
            )
        except requests.RequestException as e:
            raise LambdaAPIError(f"Request failed: {e}")

        # Parse response
        try:
            result = response.json()
        except ValueError:
            raise LambdaAPIError(
                f"Invalid JSON response: {response.text[:200]}",
                status_code=response.status_code,
            )

        # Check for errors
        if not response.ok:
            error_msg = result.get("error", {}).get("message", response.text)
            raise LambdaAPIError(
                f"API error: {error_msg}",
                status_code=response.status_code,
                response=result,
            )

        return result

    # =========================================================================
    # Instance Types
    # =========================================================================

    def list_instance_types(self) -> dict[str, Any]:
        """List available instance types with availability.

        Returns:
            Dict mapping instance type name to availability info.
        """
        result = self._request("GET", "/instance-types")
        return result.get("data", {})

    def get_available_regions(self, instance_type: str) -> list[str]:
        """Get regions where an instance type is available.

        Args:
            instance_type: Instance type name (e.g., gpu_1x_a100)

        Returns:
            List of available region names.
        """
        types = self.list_instance_types()
        type_info = types.get(instance_type, {})
        
        if not type_info:
            # Debug: show available instance types
            available_types = list(types.keys())
            raise LambdaAPIError(
                f"Instance type '{instance_type}' not found. Available types: {available_types[:10]}"
            )
        
        # Try multiple possible field names for regions
        regions = (
            type_info.get("regions_with_capacity_available") or
            type_info.get("regions_with_capacity") or
            type_info.get("available_regions") or
            []
        )
        
        # Handle different response formats
        if not regions:
            return []
        
        # Check if regions is a list of dicts with "name" key
        if isinstance(regions[0], dict):
            # Try different possible keys for region name
            result = []
            for r in regions:
                if "name" in r:
                    result.append(r["name"])
                elif "region_name" in r:
                    result.append(r["region_name"])
                elif "region" in r:
                    # If region is nested, try to get name from it
                    region_obj = r["region"]
                    if isinstance(region_obj, dict):
                        result.append(region_obj.get("name", str(region_obj)))
                    else:
                        result.append(str(region_obj))
                else:
                    # Fallback: use first string value or string representation
                    for key, value in r.items():
                        if isinstance(value, str):
                            result.append(value)
                            break
                    else:
                        result.append(str(r))
            return result
        # Or if regions is a list of strings
        elif isinstance(regions[0], str):
            return regions
        else:
            # Fallback: try to extract name from any structure
            return [str(r) for r in regions]

    # =========================================================================
    # Instances
    # =========================================================================

    def list_instances(self) -> list[Instance]:
        """List all instances.

        Returns:
            List of Instance objects.
        """
        result = self._request("GET", "/instances")
        return [Instance.from_api(i) for i in result.get("data", [])]

    def get_instance(self, instance_id: str) -> Instance | None:
        """Get instance by ID.

        Args:
            instance_id: Instance ID.

        Returns:
            Instance object or None if not found.
        """
        try:
            result = self._request("GET", f"/instances/{instance_id}")
            return Instance.from_api(result.get("data", {}))
        except LambdaAPIError as e:
            if e.status_code == 404:
                return None
            raise

    def get_instance_by_name(self, name: str) -> Instance | None:
        """Get instance by name.

        Args:
            name: Instance name.

        Returns:
            Instance object or None if not found.
        """
        instances = self.list_instances()
        for inst in instances:
            if inst.name == name:
                return inst
        return None

    def launch_instance(
        self,
        instance_type: str,
        region: str,
        ssh_key_names: list[str],
        filesystem_names: list[str] | None = None,
        name: str | None = None,
    ) -> str:
        """Launch a new instance.

        Args:
            instance_type: GPU instance type (e.g., gpu_1x_a100)
            region: Region name (e.g., us-west-1)
            ssh_key_names: List of SSH key names to add
            filesystem_names: Optional list of filesystem names to attach
            name: Optional instance name

        Returns:
            Instance ID of the launched instance.

        Raises:
            LambdaAPIError: On launch failure.
        """
        data = {
            "instance_type_name": instance_type,
            "region_name": region,
            "ssh_key_names": ssh_key_names,
        }

        if filesystem_names:
            data["file_system_names"] = filesystem_names

        if name:
            data["name"] = name

        result = self._request("POST", "/instance-operations/launch", data=data)
        instance_ids = result.get("data", {}).get("instance_ids", [])

        if not instance_ids:
            raise LambdaAPIError("No instance ID returned from launch")

        return instance_ids[0]

    def terminate_instance(self, instance_id: str) -> bool:
        """Terminate an instance.

        Args:
            instance_id: Instance ID to terminate.

        Returns:
            True if termination was initiated.

        Raises:
            LambdaAPIError: On termination failure.
        """
        data = {"instance_ids": [instance_id]}
        result = self._request("POST", "/instance-operations/terminate", data=data)
        terminated = result.get("data", {}).get("terminated_instances", [])
        return len(terminated) > 0

    def wait_for_instance(
        self,
        instance_id: str,
        target_status: str = "active",
        timeout: int = 300,
        poll_interval: int = 5,
        callback: Callable | None = None,
    ) -> Instance:
        """Wait for instance to reach target status.

        Args:
            instance_id: Instance ID to monitor.
            target_status: Status to wait for (default: "active")
            timeout: Maximum seconds to wait.
            poll_interval: Seconds between status checks.
            callback: Optional callback(instance, elapsed_seconds) for progress.

        Returns:
            Instance object when target status reached.

        Raises:
            LambdaAPIError: On timeout or instance termination.
        """
        start = time.time()

        while True:
            elapsed = time.time() - start
            if elapsed > timeout:
                raise LambdaAPIError(
                    f"Timeout waiting for instance {instance_id} to reach '{target_status}'"
                )

            instance = self.get_instance(instance_id)
            if instance is None:
                raise LambdaAPIError(f"Instance {instance_id} not found")

            if callback:
                callback(instance, int(elapsed))

            if instance.status == target_status:
                return instance

            if instance.status in ("terminated", "terminating"):
                raise LambdaAPIError(
                    f"Instance {instance_id} is {instance.status}"
                )

            time.sleep(poll_interval)

    # =========================================================================
    # SSH Keys
    # =========================================================================

    def list_ssh_keys(self) -> list[dict[str, str]]:
        """List registered SSH keys.

        Returns:
            List of dicts with 'id' and 'name' keys.
        """
        result = self._request("GET", "/ssh-keys")
        return result.get("data", [])

    def get_ssh_key_names(self) -> list[str]:
        """Get list of SSH key names.

        Returns:
            List of SSH key names.
        """
        keys = self.list_ssh_keys()
        return [k["name"] for k in keys]

    # =========================================================================
    # Filesystems
    # =========================================================================

    def list_filesystems(self) -> list[dict[str, Any]]:
        """List persistent filesystems.

        Returns:
            List of filesystem dicts.
        """
        result = self._request("GET", "/file-systems")
        return result.get("data", [])

    def get_filesystem_names(self) -> list[str]:
        """Get list of filesystem names.

        Returns:
            List of filesystem names.
        """
        filesystems = self.list_filesystems()
        return [f["name"] for f in filesystems]

    def validate_filesystem(self, name: str) -> bool:
        """Check if filesystem exists.

        Args:
            name: Filesystem name.

        Returns:
            True if filesystem exists.
        """
        return name in self.get_filesystem_names()

    def get_filesystem_region(self, name: str) -> str | None:
        """Get the region where a filesystem is located.

        Args:
            name: Filesystem name.

        Returns:
            Region name or None if filesystem not found.
        """
        filesystems = self.list_filesystems()
        for fs in filesystems:
            if fs.get("name") == name:
                # Try different possible field names for region
                region = (
                    fs.get("region") or
                    fs.get("region_name") or
                    (fs.get("region") and isinstance(fs["region"], dict) and fs["region"].get("name")) or
                    None
                )
                
                # Handle nested region dict
                if isinstance(region, dict):
                    region = region.get("name")
                
                return region
        return None
