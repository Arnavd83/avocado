"""Bootstrap operations for remote instances.

Handles setting up persistent filesystem directories and environment.
"""

import re
import sys
import threading
import time
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
    # - state: 700 (sensitive data)
    # - logs/watchdog and run: 777 (containers run as different users)
    chmod_cmd = (
        f"chmod 755 {fs_path} && "
        f"chmod 700 {fs_path}/state && "
        f"chmod 777 {fs_path}/logs/watchdog {fs_path}/run"
    )
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
        # Debug: show key prefix/suffix (not full key for security)
        key_preview = f"{authkey[:15]}...{authkey[-10:]}" if len(authkey) > 25 else "***"
        callback(f"Using authkey: {key_preview} (length: {len(authkey)})")

    # Bring up Tailscale
    # --ssh enables Tailscale SSH (optional but useful)
    # --accept-routes accepts routes from other nodes
    # Quote authkey to handle special characters properly
    up_cmd = f"sudo tailscale up --authkey='{authkey}' --hostname={hostname} --ssh --accept-routes"
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


def _run_docker_cmd(
    ssh_client: SSHClient,
    cmd: str,
    deploy_path: str | None = None,
    timeout: int = 120,
) -> tuple[int, str, str]:
    """Run a docker command.

    The docker socket is configured with SocketMode=0666 via systemd override,
    so docker commands work without sudo or group membership.

    Args:
        ssh_client: Connected SSH client.
        cmd: Docker command to run (e.g., "docker compose pull").
        deploy_path: Optional path to change to before running command.
        timeout: Command timeout in seconds.

    Returns:
        Tuple of (exit_code, stdout, stderr).
    """
    if deploy_path:
        full_cmd = f"cd {deploy_path} && {cmd}"
    else:
        full_cmd = cmd
    return ssh_client.run(full_cmd, timeout=timeout)


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

    # Verify docker is accessible (socket permissions configured via systemd override)
    if callback:
        callback("Verifying docker access...")

    exit_code, _, _ = ssh_client.run("docker info >/dev/null 2>&1", timeout=10)
    if exit_code == 0:
        if callback:
            callback("Docker is accessible")
    else:
        if callback:
            callback("Warning: Docker access check failed - socket permissions may need verification")

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
    instance_id: str | None = None,
    idle_timeout: int | None = None,
    lambda_api_key: str | None = None,
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
        instance_id: Lambda instance ID (for watchdog termination).
        idle_timeout: Idle timeout in seconds (0 to disable auto-shutdown).
        lambda_api_key: Lambda API key (for watchdog termination).
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
        "VLLM_PORT=8001",  # Internal port, proxy handles external on 8000
    ])

    # Watchdog configuration (Phase 9)
    env_lines.extend([
        "",
        "# Watchdog configuration",
    ])

    if instance_id:
        env_lines.append(f"INSTANCE_ID={instance_id}")

    # idle_timeout is in seconds (CLI passes minutes * 60)
    if idle_timeout is not None:
        env_lines.append(f"IDLE_TIMEOUT={idle_timeout}")
    else:
        env_lines.append("IDLE_TIMEOUT=3600")  # Default 60 minutes

    if lambda_api_key:
        env_lines.append(f"LAMBDA_API_KEY={lambda_api_key}")

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

    # First, verify Docker daemon is running and compose is available
    if callback:
        callback("Checking Docker daemon status...")
    
    exit_code, stdout, stderr = _run_docker_cmd(
        ssh_client, "docker info >/dev/null 2>&1", timeout=10
    )
    
    if exit_code != 0:
        raise BootstrapError(
            f"Docker daemon is not responding. "
            f"Try: sudo systemctl start docker. Error: {stderr}"
        )
    
    if callback:
        callback("Docker daemon is running")
    
    # Verify docker compose is available
    if callback:
        callback("Checking docker compose availability...")
    
    exit_code, stdout, stderr = _run_docker_cmd(
        ssh_client, "docker compose version >/dev/null 2>&1", timeout=10
    )
    
    if exit_code != 0:
        raise BootstrapError(
            f"docker compose is not available. "
            f"Install with: sudo apt-get install -y docker-compose-plugin. Error: {stderr}"
        )
    
    if callback:
        callback("docker compose is available")

    # Pull image first (can take a while)
    if callback:
        callback("Pulling vLLM image (this may take several minutes)...")
        callback("  Image size: ~10-20GB, download speed depends on network")
        callback("  Showing progress in real-time...")

    # Build the pull command
    # Note: docker compose may buffer output when not in a TTY, so we might not see
    # real-time progress, but the command will still work
    if deploy_path:
        pull_cmd = f"cd {deploy_path} && docker compose pull"
    else:
        pull_cmd = "docker compose pull"
    
    # Add heartbeat to show activity even if output is buffered
    last_output_time = [time.time()]
    heartbeat_interval = 30  # Show heartbeat every 30 seconds
    start_time = time.time()
    last_progress_line = [None]
    
    # Pattern to detect progress bar lines (e.g., "Extracting [====>] 2.045GB/3.627GB")
    progress_pattern = re.compile(r'.*\[.*>.*\].*\d+\.\d+[GMK]?B/\d+\.\d+[GMK]?B')
    
    def streaming_callback(line):
        last_output_time[0] = time.time()
        if callback:
            # Check if this is a progress bar line
            is_progress = progress_pattern.match(line.strip())
            
            if is_progress:
                # Update in place - use carriage return to overwrite previous line
                if last_progress_line[0] is not None:
                    # Clear previous line (move cursor to start, clear to end)
                    sys.stdout.write('\r' + ' ' * len(last_progress_line[0]) + '\r')
                # Print new progress line
                display_line = f"  {line.strip()}"
                sys.stdout.write(display_line)
                sys.stdout.flush()
                last_progress_line[0] = display_line
            else:
                # Regular line - print on new line
                if last_progress_line[0] is not None:
                    # Clear progress line and move to new line
                    sys.stdout.write('\r' + ' ' * len(last_progress_line[0]) + '\r\n')
                    sys.stdout.flush()
                    last_progress_line[0] = None
                callback(line)
    
    # Start a heartbeat thread to show progress even if output is buffered
    def heartbeat():
        while True:
            elapsed = time.time() - start_time
            if elapsed > 600:  # Stop after timeout
                break
            if time.time() - last_output_time[0] > heartbeat_interval:
                if callback:
                    elapsed_min = int(elapsed / 60)
                    # Clear progress line if showing
                    if last_progress_line[0] is not None:
                        sys.stdout.write('\r' + ' ' * len(last_progress_line[0]) + '\r')
                    callback(f"  Still pulling... ({elapsed_min}m elapsed)")
                    sys.stdout.flush()
                last_output_time[0] = time.time()
            time.sleep(heartbeat_interval)
    
    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()
    
    # Use streaming to show progress
    try:
        exit_code, stdout, stderr = ssh_client.run_streaming(
            pull_cmd,
            timeout=600,
            callback=streaming_callback,
        )
    finally:
        # Clear progress line and move to new line
        if last_progress_line[0] is not None:
            sys.stdout.write('\r' + ' ' * len(last_progress_line[0]) + '\r\n')
            sys.stdout.flush()
        # Stop heartbeat
        last_output_time[0] = 0  # Signal to stop

    if exit_code != 0:
        # Show more context in error
        error_msg = stderr or stdout or "Unknown error"
        raise BootstrapError(
            f"Failed to pull vLLM image (exit code {exit_code}). "
            f"Check network connection and Docker daemon. Error: {error_msg[:500]}"
        )

    if callback:
        callback("vLLM image pulled successfully")

    # Start container
    if callback:
        callback("Starting vLLM container...")
    
    exit_code, stdout, stderr = _run_docker_cmd(
        ssh_client, "docker compose up -d", deploy_path=deploy_path, timeout=120
    )

    if exit_code != 0:
        # Show more context in error
        error_msg = stderr or stdout or "Unknown error"
        raise BootstrapError(
            f"Failed to start vLLM container (exit code {exit_code}). "
            f"Check docker-compose.yml and .env file. Error: {error_msg[:500]}"
        )

    if callback:
        callback("vLLM container started")
        callback("Container is initializing (this may take several minutes on cold boot)")
        callback("  - Downloading model from HuggingFace (if not cached)")
        callback("  - Loading model into GPU memory")
        callback("  - Starting vLLM API server")
        callback("  Check logs with: docker logs inference-vllm")

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

    exit_code, stdout, stderr = _run_docker_cmd(
        ssh_client, "docker compose down", deploy_path=deploy_path, timeout=60
    )

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
    exit_code, stdout, _ = _run_docker_cmd(ssh_client, status_cmd, timeout=10)

    if exit_code != 0:
        return {"running": False, "status": "not found", "health": "unknown"}

    status = stdout.strip()

    # Get health status
    health_cmd = "docker inspect inference-vllm --format '{{.State.Health.Status}}' 2>/dev/null"
    exit_code, health_stdout, _ = _run_docker_cmd(ssh_client, health_cmd, timeout=10)
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
    instance_id: str | None = None,
    idle_timeout: int | None = None,
    lambda_api_key: str | None = None,
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
        instance_id: Lambda instance ID (for watchdog termination).
        idle_timeout: Idle timeout in seconds (0 to disable).
        lambda_api_key: Lambda API key (for watchdog termination).
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
        instance_id=instance_id,
        idle_timeout=idle_timeout,
        lambda_api_key=lambda_api_key,
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
