"""Bootstrap operations for remote instances.

Handles setting up persistent filesystem directories and environment.
"""

from pathlib import Path
from typing import Callable

from .config import get_config
from .ssh import SSHClient, SSHError


class BootstrapError(Exception):
    """Bootstrap operation error."""
    pass


def get_fs_path(filesystem_name: str) -> str:
    """Get the full path to a Lambda persistent filesystem.

    Args:
        filesystem_name: Name of the filesystem.

    Returns:
        Full path like /lambda/nfs/<filesystem_name>
    """
    config = get_config()
    base = config.paths.get("persistent_fs_base", "/lambda/nfs")
    return f"{base}/{filesystem_name}"


def setup_directories(
    ssh_client: SSHClient,
    filesystem_name: str,
    callback: Callable[[str], None] | None = None,
) -> bool:
    """Create required directories on persistent filesystem.

    Args:
        ssh_client: Connected SSH client.
        filesystem_name: Name of the Lambda persistent filesystem.
        callback: Optional callback for progress messages.

    Returns:
        True if successful.

    Raises:
        BootstrapError: On failure.
    """
    fs_path = get_fs_path(filesystem_name)

    if callback:
        callback(f"Creating directories at {fs_path}...")

    # Create directory structure
    dirs = [
        f"{fs_path}/state",
        f"{fs_path}/hf-cache/hub",
        f"{fs_path}/hf-cache/transformers",
        f"{fs_path}/adapters",
        f"{fs_path}/manifests",
        f"{fs_path}/logs/bootstrap",
        f"{fs_path}/logs/vllm",
        f"{fs_path}/logs/watchdog",
        f"{fs_path}/run",
    ]

    mkdir_cmd = f"mkdir -p {' '.join(dirs)}"
    exit_code, stdout, stderr = ssh_client.run(mkdir_cmd, timeout=30)

    if exit_code != 0:
        raise BootstrapError(f"Failed to create directories: {stderr}")

    # Set permissions
    chmod_cmd = f"chmod 755 {fs_path} && chmod 700 {fs_path}/state"
    exit_code, stdout, stderr = ssh_client.run(chmod_cmd, timeout=30)

    if exit_code != 0:
        raise BootstrapError(f"Failed to set permissions: {stderr}")

    if callback:
        callback("Directories created successfully")

    return True


def write_env_file(
    ssh_client: SSHClient,
    filesystem_name: str,
    hf_token: str | None = None,
    vllm_api_key: str | None = None,
    model_id: str | None = None,
    model_revision: str | None = None,
    max_model_len: int = 16384,
    callback: Callable[[str], None] | None = None,
) -> bool:
    """Write environment file for vLLM and HuggingFace cache.

    Args:
        ssh_client: Connected SSH client.
        filesystem_name: Name of the Lambda persistent filesystem.
        hf_token: HuggingFace token for gated models.
        vllm_api_key: API key for vLLM authentication.
        model_id: HuggingFace model ID.
        model_revision: Model revision/commit.
        max_model_len: Maximum model context length.
        callback: Optional callback for progress messages.

    Returns:
        True if successful.

    Raises:
        BootstrapError: On failure.
    """
    fs_path = get_fs_path(filesystem_name)
    env_path = f"{fs_path}/run/inference_server.env"

    if callback:
        callback(f"Writing environment file to {env_path}...")

    # Build environment content
    env_lines = [
        "# Inference Server Environment",
        "# Auto-generated - do not edit manually",
        "",
        "# HuggingFace cache paths (persistent filesystem)",
        f"HF_HOME={fs_path}/hf-cache",
        f"HF_HUB_CACHE={fs_path}/hf-cache/hub",
        f"TRANSFORMERS_CACHE={fs_path}/hf-cache/transformers",
        "",
    ]

    if hf_token:
        env_lines.append(f"HF_TOKEN={hf_token}")
        env_lines.append("")

    if vllm_api_key:
        env_lines.append(f"VLLM_API_KEY={vllm_api_key}")
        env_lines.append("")

    if model_id:
        env_lines.append(f"MODEL_ID={model_id}")

    if model_revision:
        env_lines.append(f"MODEL_REVISION={model_revision}")

    env_lines.append(f"MAX_MODEL_LEN={max_model_len}")
    env_lines.append("")
    env_lines.append("# Runtime LoRA support")
    env_lines.append("VLLM_ALLOW_RUNTIME_LORA_UPDATING=true")

    env_content = "\n".join(env_lines)

    # Write file via SSH
    # Escape content for shell
    escaped = env_content.replace("'", "'\"'\"'")
    write_cmd = f"echo '{escaped}' > {env_path} && chmod 600 {env_path}"

    exit_code, stdout, stderr = ssh_client.run(write_cmd, timeout=30)

    if exit_code != 0:
        raise BootstrapError(f"Failed to write env file: {stderr}")

    if callback:
        callback("Environment file created successfully")

    return True


def verify_directories(
    ssh_client: SSHClient,
    filesystem_name: str,
) -> dict[str, bool]:
    """Verify that required directories exist.

    Args:
        ssh_client: Connected SSH client.
        filesystem_name: Name of the Lambda persistent filesystem.

    Returns:
        Dict mapping directory name to existence status.
    """
    fs_path = get_fs_path(filesystem_name)

    dirs_to_check = [
        "state",
        "hf-cache",
        "hf-cache/hub",
        "adapters",
        "manifests",
        "logs",
        "run",
    ]

    results = {}
    for dir_name in dirs_to_check:
        full_path = f"{fs_path}/{dir_name}"
        exit_code, _, _ = ssh_client.run(f"test -d {full_path}", timeout=10)
        results[dir_name] = (exit_code == 0)

    return results


def verify_env_file(
    ssh_client: SSHClient,
    filesystem_name: str,
) -> dict[str, str | None]:
    """Verify environment file exists and read its contents.

    Args:
        ssh_client: Connected SSH client.
        filesystem_name: Name of the Lambda persistent filesystem.

    Returns:
        Dict of environment variables, or empty dict if file doesn't exist.
    """
    fs_path = get_fs_path(filesystem_name)
    env_path = f"{fs_path}/run/inference_server.env"

    exit_code, stdout, _ = ssh_client.run(f"cat {env_path} 2>/dev/null", timeout=10)

    if exit_code != 0:
        return {}

    # Parse env file
    env_vars = {}
    for line in stdout.strip().split("\n"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            env_vars[key] = value

    return env_vars


def run_bootstrap_lite(
    ssh_client: SSHClient,
    filesystem_name: str,
    hf_token: str | None = None,
    vllm_api_key: str | None = None,
    model_id: str | None = None,
    model_revision: str | None = None,
    max_model_len: int = 16384,
    callback: Callable[[str], None] | None = None,
) -> bool:
    """Run lightweight bootstrap (directories + env file only).

    This is Phase 3 bootstrap - no Docker, Tailscale, or vLLM yet.

    Args:
        ssh_client: Connected SSH client.
        filesystem_name: Name of the Lambda persistent filesystem.
        hf_token: HuggingFace token.
        vllm_api_key: vLLM API key.
        model_id: HuggingFace model ID.
        model_revision: Model revision.
        max_model_len: Max context length.
        callback: Optional progress callback.

    Returns:
        True if successful.

    Raises:
        BootstrapError: On failure.
    """
    if callback:
        callback("Starting bootstrap-lite...")

    # Step 1: Create directories
    setup_directories(ssh_client, filesystem_name, callback)

    # Step 2: Write environment file
    write_env_file(
        ssh_client,
        filesystem_name,
        hf_token=hf_token,
        vllm_api_key=vllm_api_key,
        model_id=model_id,
        model_revision=model_revision,
        max_model_len=max_model_len,
        callback=callback,
    )

    # Step 3: Verify
    if callback:
        callback("Verifying setup...")

    dir_status = verify_directories(ssh_client, filesystem_name)
    if not all(dir_status.values()):
        missing = [k for k, v in dir_status.items() if not v]
        raise BootstrapError(f"Missing directories: {missing}")

    env_vars = verify_env_file(ssh_client, filesystem_name)
    if "HF_HOME" not in env_vars:
        raise BootstrapError("Environment file missing HF_HOME")

    if callback:
        callback("Bootstrap-lite complete!")

    return True


# =============================================================================
# Tailscale Setup (Phase 4)
# =============================================================================


def install_tailscale(
    ssh_client: SSHClient,
    callback: Callable[[str], None] | None = None,
) -> bool:
    """Install Tailscale on the remote instance.

    Args:
        ssh_client: Connected SSH client.
        callback: Optional progress callback.

    Returns:
        True if successful.

    Raises:
        BootstrapError: On failure.
    """
    if callback:
        callback("Checking if Tailscale is installed...")

    # Check if already installed
    exit_code, stdout, _ = ssh_client.run("command -v tailscale", timeout=10)
    if exit_code == 0:
        if callback:
            callback("Tailscale already installed")
        return True

    if callback:
        callback("Installing Tailscale...")

    # Install using official script
    install_cmd = "curl -fsSL https://tailscale.com/install.sh | sh"
    exit_code, stdout, stderr = ssh_client.run(install_cmd, timeout=120)

    if exit_code != 0:
        raise BootstrapError(f"Failed to install Tailscale: {stderr}")

    # Verify installation
    exit_code, stdout, _ = ssh_client.run("tailscale version", timeout=10)
    if exit_code != 0:
        raise BootstrapError("Tailscale installed but version check failed")

    if callback:
        callback(f"Tailscale installed: {stdout.strip()}")

    return True


def setup_tailscale(
    ssh_client: SSHClient,
    authkey: str,
    hostname: str,
    callback: Callable[[str], None] | None = None,
) -> dict[str, str]:
    """Bring up Tailscale with the provided auth key.

    Args:
        ssh_client: Connected SSH client.
        authkey: Tailscale auth key.
        hostname: Hostname for this machine on the tailnet.
        callback: Optional progress callback.

    Returns:
        Dict with 'ip' and 'hostname' keys.

    Raises:
        BootstrapError: On failure.
    """
    if callback:
        callback(f"Connecting to Tailscale as '{hostname}'...")

    # Bring up Tailscale
    # --ssh enables Tailscale SSH (optional but useful)
    # --accept-routes accepts routes from other nodes
    up_cmd = f"sudo tailscale up --authkey={authkey} --hostname={hostname} --ssh --accept-routes"
    exit_code, stdout, stderr = ssh_client.run(up_cmd, timeout=60)

    if exit_code != 0:
        # Check if it's already connected
        if "already" in stderr.lower() or "already" in stdout.lower():
            if callback:
                callback("Tailscale already connected")
        else:
            raise BootstrapError(f"Failed to connect Tailscale: {stderr}")

    # Get Tailscale IP
    exit_code, ts_ip, _ = ssh_client.run("tailscale ip -4", timeout=10)
    if exit_code != 0:
        raise BootstrapError("Failed to get Tailscale IP")

    ts_ip = ts_ip.strip()

    # Get Tailscale hostname (may differ from requested if collision)
    exit_code, ts_hostname, _ = ssh_client.run("tailscale status --self --json | grep -o '\"HostName\":\"[^\"]*\"' | cut -d'\"' -f4", timeout=10)
    if exit_code != 0 or not ts_hostname.strip():
        ts_hostname = hostname  # Fall back to requested hostname

    ts_hostname = ts_hostname.strip()

    if callback:
        callback(f"Tailscale connected: {ts_ip} ({ts_hostname})")

    return {
        "ip": ts_ip,
        "hostname": ts_hostname,
    }


def get_tailscale_status(
    ssh_client: SSHClient,
) -> dict[str, str | bool] | None:
    """Get current Tailscale status.

    Args:
        ssh_client: Connected SSH client.

    Returns:
        Dict with status info, or None if Tailscale not running.
    """
    # Check if Tailscale is installed
    exit_code, _, _ = ssh_client.run("command -v tailscale", timeout=10)
    if exit_code != 0:
        return None

    # Get status
    exit_code, stdout, _ = ssh_client.run("tailscale status --self --json 2>/dev/null", timeout=10)
    if exit_code != 0:
        return {"installed": True, "connected": False}

    # Get IP
    exit_code, ts_ip, _ = ssh_client.run("tailscale ip -4 2>/dev/null", timeout=10)

    return {
        "installed": True,
        "connected": True,
        "ip": ts_ip.strip() if exit_code == 0 else None,
    }


def write_tailscale_info(
    ssh_client: SSHClient,
    filesystem_name: str,
    tailscale_ip: str,
    tailscale_hostname: str,
    callback: Callable[[str], None] | None = None,
) -> bool:
    """Write Tailscale info to persistent filesystem.

    Args:
        ssh_client: Connected SSH client.
        filesystem_name: Filesystem name.
        tailscale_ip: Tailscale IP address.
        tailscale_hostname: Tailscale hostname.
        callback: Optional progress callback.

    Returns:
        True if successful.
    """
    fs_path = get_fs_path(filesystem_name)
    ts_info_path = f"{fs_path}/run/tailscale.json"

    if callback:
        callback(f"Saving Tailscale info to {ts_info_path}...")

    content = f'{{"ip": "{tailscale_ip}", "hostname": "{tailscale_hostname}"}}'
    write_cmd = f"echo '{content}' > {ts_info_path}"

    exit_code, _, stderr = ssh_client.run(write_cmd, timeout=10)
    if exit_code != 0:
        raise BootstrapError(f"Failed to write Tailscale info: {stderr}")

    return True


def run_bootstrap_with_tailscale(
    ssh_client: SSHClient,
    filesystem_name: str,
    tailscale_authkey: str,
    instance_name: str,
    hf_token: str | None = None,
    vllm_api_key: str | None = None,
    model_id: str | None = None,
    model_revision: str | None = None,
    max_model_len: int = 16384,
    callback: Callable[[str], None] | None = None,
) -> dict[str, str]:
    """Run full bootstrap with Tailscale setup.

    This is Phase 4 bootstrap - directories, env file, and Tailscale.

    Args:
        ssh_client: Connected SSH client.
        filesystem_name: Filesystem name.
        tailscale_authkey: Tailscale auth key.
        instance_name: Instance name (used as Tailscale hostname).
        hf_token: HuggingFace token.
        vllm_api_key: vLLM API key.
        model_id: Model ID.
        model_revision: Model revision.
        max_model_len: Max context length.
        callback: Progress callback.

    Returns:
        Dict with 'tailscale_ip' and 'tailscale_hostname'.

    Raises:
        BootstrapError: On failure.
    """
    if callback:
        callback("Starting bootstrap with Tailscale...")

    # Step 1: Run lite bootstrap (directories + env)
    run_bootstrap_lite(
        ssh_client,
        filesystem_name,
        hf_token=hf_token,
        vllm_api_key=vllm_api_key,
        model_id=model_id,
        model_revision=model_revision,
        max_model_len=max_model_len,
        callback=callback,
    )

    # Step 2: Install Tailscale
    install_tailscale(ssh_client, callback)

    # Step 3: Connect to Tailscale
    ts_info = setup_tailscale(
        ssh_client,
        authkey=tailscale_authkey,
        hostname=instance_name,
        callback=callback,
    )

    # Step 4: Save Tailscale info to filesystem
    write_tailscale_info(
        ssh_client,
        filesystem_name,
        ts_info["ip"],
        ts_info["hostname"],
        callback=callback,
    )

    if callback:
        callback("Bootstrap with Tailscale complete!")

    return {
        "tailscale_ip": ts_info["ip"],
        "tailscale_hostname": ts_info["hostname"],
    }


# =============================================================================
# Docker + vLLM Setup (Phase 5)
# =============================================================================


def install_docker(
    ssh_client: SSHClient,
    deploy_path: str = "~/inference_deploy",
    callback: Callable[[str], None] | None = None,
) -> bool:
    """Install Docker with NVIDIA Container Toolkit.

    Args:
        ssh_client: Connected SSH client.
        deploy_path: Path to deploy directory on remote.
        callback: Optional progress callback.

    Returns:
        True if successful.

    Raises:
        BootstrapError: On failure.
    """
    if callback:
        callback("Checking Docker installation...")

    # Check if Docker is already installed and working with GPU
    exit_code, stdout, _ = ssh_client.run(
        "docker info 2>/dev/null | grep -q nvidia && echo 'ready'",
        timeout=30,
    )
    if exit_code == 0 and "ready" in stdout:
        if callback:
            callback("Docker with NVIDIA runtime already installed")
        return True

    if callback:
        callback("Installing Docker and NVIDIA Container Toolkit...")

    # Run install script
    install_cmd = f"sudo bash {deploy_path}/scripts/install_docker.sh"
    exit_code, stdout, stderr = ssh_client.run(install_cmd, timeout=300)

    if exit_code != 0:
        raise BootstrapError(f"Failed to install Docker: {stderr}")

    # Verify installation
    exit_code, stdout, _ = ssh_client.run("docker --version", timeout=10)
    if exit_code != 0:
        raise BootstrapError("Docker installation verification failed")

    if callback:
        callback(f"Docker installed: {stdout.strip()}")

    return True


def write_docker_env(
    ssh_client: SSHClient,
    deploy_path: str,
    filesystem_name: str,
    model_id: str,
    vllm_api_key: str,
    hf_token: str | None = None,
    model_revision: str | None = None,
    max_model_len: int = 16384,
    tailscale_ip: str | None = None,
    callback: Callable[[str], None] | None = None,
) -> bool:
    """Write .env file for docker-compose.

    Args:
        ssh_client: Connected SSH client.
        deploy_path: Path to deploy directory on remote.
        filesystem_name: Filesystem name.
        model_id: HuggingFace model ID.
        vllm_api_key: vLLM API key.
        hf_token: HuggingFace token.
        model_revision: Model revision.
        max_model_len: Max context length.
        tailscale_ip: Tailscale IP to bind to.
        callback: Optional progress callback.

    Returns:
        True if successful.

    Raises:
        BootstrapError: On failure.
    """
    fs_path = get_fs_path(filesystem_name)
    env_path = f"{deploy_path}/.env"

    if callback:
        callback(f"Writing Docker environment to {env_path}...")

    # Build environment content
    env_lines = [
        "# Docker Compose Environment",
        "# Auto-generated by inference-server bootstrap",
        "",
        f"FS_PATH={fs_path}",
        f"MODEL_ID={model_id}",
        f"MAX_MODEL_LEN={max_model_len}",
        f"VLLM_API_KEY={vllm_api_key}",
        "",
    ]

    if model_revision:
        env_lines.append(f"MODEL_REVISION={model_revision}")

    if hf_token:
        env_lines.append(f"HF_TOKEN={hf_token}")

    if tailscale_ip:
        env_lines.append(f"TAILSCALE_IP={tailscale_ip}")
    else:
        env_lines.append("TAILSCALE_IP=0.0.0.0")

    # Performance settings
    env_lines.extend([
        "",
        "# Performance settings",
        "GPU_MEMORY_UTIL=0.85",
        "MAX_LORAS=5",
        "MAX_LORA_RANK=64",
        "VLLM_PORT=8000",
    ])

    env_content = "\n".join(env_lines)

    # Write file via SSH
    escaped = env_content.replace("'", "'\"'\"'")
    write_cmd = f"echo '{escaped}' > {env_path} && chmod 600 {env_path}"

    exit_code, _, stderr = ssh_client.run(write_cmd, timeout=30)

    if exit_code != 0:
        raise BootstrapError(f"Failed to write Docker env file: {stderr}")

    if callback:
        callback("Docker environment file created")

    return True


def start_vllm(
    ssh_client: SSHClient,
    deploy_path: str,
    callback: Callable[[str], None] | None = None,
) -> bool:
    """Start vLLM container via docker-compose.

    Args:
        ssh_client: Connected SSH client.
        deploy_path: Path to deploy directory.
        callback: Optional progress callback.

    Returns:
        True if started successfully.

    Raises:
        BootstrapError: On failure.
    """
    if callback:
        callback("Starting vLLM container...")

    # Pull image first (can take a while)
    pull_cmd = f"cd {deploy_path} && docker compose pull"
    exit_code, stdout, stderr = ssh_client.run(pull_cmd, timeout=600)

    if exit_code != 0:
        raise BootstrapError(f"Failed to pull vLLM image: {stderr}")

    if callback:
        callback("vLLM image pulled, starting container...")

    # Start container
    up_cmd = f"cd {deploy_path} && docker compose up -d"
    exit_code, stdout, stderr = ssh_client.run(up_cmd, timeout=120)

    if exit_code != 0:
        raise BootstrapError(f"Failed to start vLLM container: {stderr}")

    if callback:
        callback("vLLM container started")

    return True


def stop_vllm(
    ssh_client: SSHClient,
    deploy_path: str,
    callback: Callable[[str], None] | None = None,
) -> bool:
    """Stop vLLM container.

    Args:
        ssh_client: Connected SSH client.
        deploy_path: Path to deploy directory.
        callback: Optional progress callback.

    Returns:
        True if stopped successfully.
    """
    if callback:
        callback("Stopping vLLM container...")

    down_cmd = f"cd {deploy_path} && docker compose down"
    exit_code, stdout, stderr = ssh_client.run(down_cmd, timeout=60)

    if exit_code != 0:
        if callback:
            callback(f"Warning: docker compose down failed: {stderr}")
        return False

    if callback:
        callback("vLLM container stopped")

    return True


def get_vllm_container_status(
    ssh_client: SSHClient,
    deploy_path: str,
) -> dict[str, str | bool]:
    """Get vLLM container status.

    Args:
        ssh_client: Connected SSH client.
        deploy_path: Path to deploy directory.

    Returns:
        Dict with 'running', 'status', 'health' keys.
    """
    # Check if container exists and is running
    status_cmd = "docker inspect inference-vllm --format '{{.State.Status}}' 2>/dev/null"
    exit_code, stdout, _ = ssh_client.run(status_cmd, timeout=10)

    if exit_code != 0:
        return {"running": False, "status": "not found", "health": "unknown"}

    status = stdout.strip()

    # Get health status
    health_cmd = "docker inspect inference-vllm --format '{{.State.Health.Status}}' 2>/dev/null"
    exit_code, health_stdout, _ = ssh_client.run(health_cmd, timeout=10)
    health = health_stdout.strip() if exit_code == 0 else "unknown"

    return {
        "running": status == "running",
        "status": status,
        "health": health,
    }


def run_full_bootstrap(
    ssh_client: SSHClient,
    filesystem_name: str,
    tailscale_authkey: str,
    instance_name: str,
    model_id: str,
    vllm_api_key: str,
    hf_token: str | None = None,
    model_revision: str | None = None,
    max_model_len: int = 16384,
    deploy_path: str = "~/inference_deploy",
    callback: Callable[[str], None] | None = None,
) -> dict[str, str]:
    """Run full bootstrap: directories, env, Tailscale, Docker, vLLM.

    This is Phase 5 bootstrap - complete server setup.

    Args:
        ssh_client: Connected SSH client.
        filesystem_name: Filesystem name.
        tailscale_authkey: Tailscale auth key.
        instance_name: Instance name (Tailscale hostname).
        model_id: HuggingFace model ID.
        vllm_api_key: vLLM API key.
        hf_token: HuggingFace token.
        model_revision: Model revision.
        max_model_len: Max context length.
        deploy_path: Path to deploy directory on remote.
        callback: Progress callback.

    Returns:
        Dict with 'tailscale_ip', 'tailscale_hostname', 'vllm_started'.

    Raises:
        BootstrapError: On failure.
    """
    if callback:
        callback("Starting full bootstrap...")

    # Step 1-4: Run Tailscale bootstrap (directories, env, Tailscale)
    ts_result = run_bootstrap_with_tailscale(
        ssh_client,
        filesystem_name,
        tailscale_authkey=tailscale_authkey,
        instance_name=instance_name,
        hf_token=hf_token,
        vllm_api_key=vllm_api_key,
        model_id=model_id,
        model_revision=model_revision,
        max_model_len=max_model_len,
        callback=callback,
    )

    # Step 5: Install Docker
    install_docker(ssh_client, deploy_path=deploy_path, callback=callback)

    # Step 6: Write Docker .env file
    write_docker_env(
        ssh_client,
        deploy_path=deploy_path,
        filesystem_name=filesystem_name,
        model_id=model_id,
        vllm_api_key=vllm_api_key,
        hf_token=hf_token,
        model_revision=model_revision,
        max_model_len=max_model_len,
        tailscale_ip=ts_result["tailscale_ip"],
        callback=callback,
    )

    # Step 7: Start vLLM
    start_vllm(ssh_client, deploy_path, callback)

    if callback:
        callback("Full bootstrap complete!")

    return {
        "tailscale_ip": ts_result["tailscale_ip"],
        "tailscale_hostname": ts_result["tailscale_hostname"],
        "vllm_started": True,
    }
