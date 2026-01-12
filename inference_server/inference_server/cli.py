"""CLI for Inference Server.

Provides commands for managing Lambda Cloud inference instances with LoRA adapters.
"""

import json
import os
import sys
from pathlib import Path
from typing import Callable

import click

from .adapter_sync import (
    AdapterSyncManager,
    discover_local_adapters,
    get_default_local_adapters_path,
    get_remote_adapters_path,
    validate_adapter,
)
from .bootstrap import (
    BootstrapError,
    run_bootstrap_lite,
    run_bootstrap_with_tailscale,
    run_full_bootstrap,
    get_fs_path,
    get_vllm_container_status,
)
from .manifest import (
    generate_manifest,
    write_manifest,
    list_manifests,
    get_manifest,
    update_manifest_adapters,
)
from .vllm_client import VLLMClient, VLLMError, HealthChecker
from .config import (
    ConfigError,
    get_config,
    get_missing_required_vars,
    validate_env,
    PROJECT_ROOT,
)
from .lambda_api import LambdaAPIError, LambdaClient
from .ssh import SSHClient, SSHError, get_ssh_client_for_instance
from .state import InstanceState, get_state_manager


def get_env_var(name: str, default: str | None = None) -> str | None:
    """Get environment variable.
    
    Ensures .env file is loaded before reading the variable.
    """
    from .config import load_env
    load_env()  # Ensure .env is loaded
    return os.environ.get(name, default)


def get_default_ssh_key() -> str | None:
    """Get default SSH key from env or config."""
    return os.environ.get("LAMBDA_SSH_KEY_NAME")


def get_default_filesystem() -> str | None:
    """Get default filesystem from env."""
    return os.environ.get("LAMBDA_FILESYSTEM_NAME")


@click.group()
@click.version_option(package_name="inference_server")
def cli():
    """Inference Server - Lambda Cloud hosted inference with LoRA adapters.

    Manage GPU instances, deploy vLLM, and dynamically load/unload LoRA adapters.
    """
    pass


# =============================================================================
# Core Instance Commands
# =============================================================================


@cli.command()
@click.option("--name", help="Instance name (default: {prefix}-{model})")
@click.option("--model", "model_alias", help="Model alias from config (e.g., llama31-8b)")
@click.option("--gpu", help="GPU type (e.g., gpu_1x_a100)")
@click.option("--filesystem", help="Lambda persistent filesystem name")
@click.option("--ssh-key", help="Lambda SSH key name")
@click.option("--model-id", help="Override HuggingFace model ID")
@click.option("--model-revision", help="Override model revision/commit")
@click.option("--max-model-len", type=int, help="Override max model length")
@click.option("--tailscale-authkey", help="Tailscale auth key")
@click.option("--idle-timeout", type=int, help="Auto-shutdown after N minutes idle (0=disabled)")
@click.option("--health-timeout", type=int, default=900, help="Health check timeout in seconds (default: 900)")
@click.option("--no-bootstrap", is_flag=True, help="Only create instance, skip bootstrap")
@click.option("--reuse-if-running", is_flag=True, default=True, help="Reuse existing instance if running")
def up(
    name,
    model_alias,
    gpu,
    filesystem,
    ssh_key,
    model_id,
    model_revision,
    max_model_len,
    tailscale_authkey,
    idle_timeout,
    health_timeout,
    no_bootstrap,
    reuse_if_running,
):
    """Create instance, bootstrap, start vLLM, and verify health.

    Creates a Lambda GPU instance with the specified filesystem attached,
    runs bootstrap to install Docker/Tailscale/vLLM, and waits for health checks.
    """
    try:
        config = get_config()
        state_mgr = get_state_manager()
        lambda_client = LambdaClient()

        # Resolve configuration
        model_alias = model_alias or config.models.get("default", "llama31-8b")
        model_config = config.get_model_config(model_alias)
        instance_name = name or config.get_instance_name(model_alias)
        primary_gpu = gpu or config.instance.get("gpu", "gpu_1x_a100")
        gpu_types_to_try = config.get_gpu_types_to_try(primary_gpu) if not gpu else [gpu]
        ssh_key_name = ssh_key or get_default_ssh_key()
        filesystem_name = filesystem or get_default_filesystem()

        # Validate required parameters
        if not filesystem_name:
            click.echo(click.style("Error: --filesystem is required (or set LAMBDA_FILESYSTEM_NAME)", fg="red"))
            sys.exit(1)

        if not ssh_key_name:
            click.echo(click.style("Error: --ssh-key is required (or set LAMBDA_SSH_KEY_NAME)", fg="red"))
            sys.exit(1)

        # Check for existing instance
        existing = state_mgr.get_instance(instance_name)
        if existing:
            if reuse_if_running:
                # Verify it's still running via API
                api_instance = lambda_client.get_instance(existing.instance_id)
                if api_instance and api_instance.status == "active":
                    click.echo(f"Instance '{instance_name}' already running")
                    click.echo(f"  Instance ID: {existing.instance_id}")
                    click.echo(f"  Public IP:   {api_instance.ip}")
                    if existing.tailscale_ip:
                        click.echo(f"  Tailscale:   {existing.tailscale_ip}")
                    return
                else:
                    # Instance in state but not running - clean up
                    click.echo(f"Cleaning up stale state for '{instance_name}'")
                    state_mgr.delete_instance(instance_name)
            else:
                click.echo(click.style(f"Error: Instance '{instance_name}' already exists", fg="red"))
                click.echo("Use --reuse-if-running or 'down' first")
                sys.exit(1)

        # Validate filesystem exists and get its region
        click.echo(f"Validating filesystem '{filesystem_name}'...")
        if not lambda_client.validate_filesystem(filesystem_name):
            available = lambda_client.get_filesystem_names()
            click.echo(click.style(f"Error: Filesystem '{filesystem_name}' not found", fg="red"))
            click.echo(f"Available: {available}")
            sys.exit(1)
        
        # Get filesystem region - we must find a GPU in the same region
        filesystem_region = lambda_client.get_filesystem_region(filesystem_name)
        if not filesystem_region:
            click.echo(click.style(f"Warning: Could not determine region for filesystem '{filesystem_name}'", fg="yellow"))
            click.echo("Will try to find GPU in any available region")
            target_region = None
        else:
            click.echo(f"Filesystem '{filesystem_name}' is in region: {filesystem_region}")
            target_region = filesystem_region

        # Validate SSH key exists
        click.echo(f"Validating SSH key '{ssh_key_name}'...")
        available_keys = lambda_client.get_ssh_key_names()
        if ssh_key_name not in available_keys:
            click.echo(click.style(f"Error: SSH key '{ssh_key_name}' not found", fg="red"))
            click.echo(f"Available: {available_keys}")
            sys.exit(1)

        # Find available GPU type and region (try primary, then fallbacks)
        # Filter for filesystem's region (or any region if filesystem region unknown)
        gpu_type = None
        available_regions = []
        tried_types = []
        
        for candidate_gpu in gpu_types_to_try:
            tried_types.append(candidate_gpu)
            click.echo(f"Checking availability for {candidate_gpu}...")
            try:
                candidate_regions = lambda_client.get_available_regions(candidate_gpu)
            except LambdaAPIError as e:
                click.echo(f"  {candidate_gpu}: Error - {e}")
                continue  # Try next GPU type
            
            # Show all available regions for this GPU type
            if candidate_regions:
                click.echo(f"  {candidate_gpu}: Available in regions: {', '.join(candidate_regions)}")
            else:
                click.echo(f"  {candidate_gpu}: No regions available")
                continue
            
            # Filter for target region (filesystem region) if specified
            if target_region:
                matching_regions = [r for r in candidate_regions if r == target_region]
                if matching_regions:
                    gpu_type = candidate_gpu
                    available_regions = matching_regions
                    click.echo(f"  {candidate_gpu}: ✓ Available in filesystem region '{target_region}'")
                    break
                else:
                    click.echo(f"  {candidate_gpu}: ✗ Not available in filesystem region '{target_region}' (available in: {', '.join(candidate_regions)})")
            else:
                # No target region - use first available
                gpu_type = candidate_gpu
                available_regions = candidate_regions
                click.echo(f"  {candidate_gpu}: ✓ Available in regions: {', '.join(candidate_regions)}")
                break
        
        if not gpu_type:
            if target_region:
                click.echo(click.style(f"\nError: No GPU types available in filesystem region '{target_region}'", fg="red"))
            else:
                click.echo(click.style(f"\nError: No regions available for any GPU type", fg="red"))
            click.echo(f"Tried GPU types: {', '.join(tried_types)}")
            # Debug: show what instance types are available
            try:
                all_types = lambda_client.list_instance_types()
                click.echo(f"\nAvailable instance types: {', '.join(list(all_types.keys())[:10])}")
            except Exception:
                pass
            click.echo("\nTry a different GPU type or wait for capacity")
            sys.exit(1)
        
        region = available_regions[0]
        if len(tried_types) > 1:
            click.echo(click.style(f"\nSelected {gpu_type} (tried {len(tried_types)} GPU type(s))", fg="green"))
        click.echo(f"Using region: {region}")

        # Launch instance
        click.echo(f"\nLaunching instance '{instance_name}'...")
        click.echo(f"  GPU:        {gpu_type}")
        click.echo(f"  Region:     {region}")
        click.echo(f"  Filesystem: {filesystem_name}")
        click.echo(f"  SSH Key:    {ssh_key_name}")
        click.echo(f"  Model:      {model_config['id']}")

        instance_id = lambda_client.launch_instance(
            instance_type=gpu_type,
            region=region,
            ssh_key_names=[ssh_key_name],
            filesystem_names=[filesystem_name],
            name=instance_name,
        )

        click.echo(f"\nInstance launched: {instance_id}")

        # Save initial state
        instance_state = InstanceState(
            instance_id=instance_id,
            name=instance_name,
            status="booting",
            gpu_type=gpu_type,
            filesystem=filesystem_name,
            ssh_key=ssh_key_name,
            region=region,
            model_id=model_id or model_config["id"],
            model_alias=model_alias,
            model_revision=model_revision or model_config.get("revision"),
            vllm_image_tag=config.vllm.get("image_tag"),
        )
        state_mgr.set_instance(instance_state)

        # Wait for instance to become active
        click.echo("\nWaiting for instance to become active...")

        def progress_callback(inst, elapsed):
            click.echo(f"  Status: {inst.status} ({elapsed}s elapsed)", nl=False)
            click.echo("\r", nl=False)

        try:
            active_instance = lambda_client.wait_for_instance(
                instance_id,
                target_status="active",
                timeout=config.timeouts.get("ssh_ready", 300),
                poll_interval=5,
                callback=progress_callback,
            )
            click.echo()  # Newline after progress
        except LambdaAPIError as e:
            click.echo()
            click.echo(click.style(f"Error waiting for instance: {e}", fg="red"))
            state_mgr.update_instance(instance_name, status="failed")
            sys.exit(1)

        # Update state with IP
        state_mgr.update_instance(
            instance_name,
            status="active",
            public_ip=active_instance.ip,
        )
        
        # Get updated instance state
        instance_state = state_mgr.get_instance(instance_name)

        click.echo(click.style("\nInstance is active!", fg="green"))
        click.echo(f"  Instance ID: {instance_id}")
        click.echo(f"  Public IP:   {active_instance.ip}")

        # Wait for SSH to become ready
        click.echo("\nWaiting for SSH to become ready...")
        ssh_client = SSHClient(host=active_instance.ip, user="ubuntu")

        def ssh_progress(attempt, status):
            click.echo(f"  SSH attempt {attempt}: {status}    ", nl=False)
            click.echo("\r", nl=False)

        try:
            ssh_client.wait_for_ready(
                timeout=config.timeouts.get("ssh_ready", 300),
                interval=5,
                callback=ssh_progress,
            )
            click.echo()  # Newline after progress
            click.echo(click.style("SSH is ready!", fg="green"))
        except SSHError as e:
            click.echo()
            click.echo(click.style(f"Warning: SSH not ready: {e}", fg="yellow"))
            click.echo("Instance is running but SSH may need more time")
        finally:
            ssh_client.close()

        if no_bootstrap:
            click.echo("\n--no-bootstrap specified, skipping bootstrap")
            click.echo(f"\nSSH command: ssh ubuntu@{active_instance.ip}")
            click.echo(f"Or use: inference-server ssh --name {instance_name}")
        else:
            # Get credentials
            ts_authkey = tailscale_authkey or get_env_var("TS_AUTHKEY")
            vllm_api_key = get_env_var("VLLM_API_KEY")
            hf_token = get_env_var("HUGGINGFACE_API_KEY")
            resolved_model_id = model_id or model_config["id"]
            resolved_revision = model_revision or model_config.get("revision")
            resolved_max_len = max_model_len or model_config.get("max_model_len", 16384)

            try:
                ssh_client = SSHClient(host=active_instance.ip, user="ubuntu")
                
                # Determine if we should start vLLM (requires both ts_authkey and vllm_api_key)
                should_start_vllm = bool(ts_authkey and vllm_api_key)
                
                # Execute bootstrap using shared helper function
                # Convert idle_timeout from minutes to seconds if provided
                idle_timeout_seconds = idle_timeout * 60 if idle_timeout is not None else None
                _execute_bootstrap(
                    ssh_client=ssh_client,
                    instance=instance_state,
                    instance_name=instance_name,
                    filesystem_name=filesystem_name,
                    ts_authkey=ts_authkey,
                    vllm_api_key=vllm_api_key,
                    hf_token=hf_token,
                    model_id=resolved_model_id,
                    model_revision=resolved_revision,
                    max_model_len=resolved_max_len,
                    start_vllm=should_start_vllm,
                    health_timeout=health_timeout,
                    push_deploy=True,
                    idle_timeout=idle_timeout_seconds,
                )

                # Show filesystem path
                fs_path = get_fs_path(filesystem_name)
                click.echo(f"\nPersistent filesystem: {fs_path}")
                click.echo(f"  HF cache:    {fs_path}/hf-cache")
                click.echo(f"  Adapters:    {fs_path}/adapters")

                click.echo(f"\nSSH command: ssh ubuntu@{active_instance.ip}")
                click.echo(f"Or use: inference-server ssh --name {instance_name}")

            except BootstrapError as e:
                click.echo(click.style(f"\nBootstrap error: {e}", fg="red"))
                click.echo("Instance is running, but bootstrap failed")
                click.echo(f"\nSSH command: ssh ubuntu@{active_instance.ip}")
                click.echo(f"You can retry bootstrap with: inference-server bootstrap --name {instance_name}")
            except SSHError as e:
                click.echo(click.style(f"\nSSH error during bootstrap: {e}", fg="red"))
                click.echo(f"\nSSH command: ssh ubuntu@{active_instance.ip}")
            finally:
                ssh_client.close()

    except ConfigError as e:
        click.echo(click.style(f"Config error: {e}", fg="red"))
        sys.exit(1)
    except LambdaAPIError as e:
        click.echo(click.style(f"Lambda API error: {e}", fg="red"))
        sys.exit(1)


@cli.command()
@click.option("--name", help="Instance name to terminate")
@click.option("--all", "terminate_all", is_flag=True, help="Terminate all instances")
def down(name, terminate_all):
    """Terminate instance(s).

    Terminates the specified instance via Lambda Cloud API.
    Data on persistent filesystem is preserved.
    """
    try:
        config = get_config()
        state_mgr = get_state_manager()
        lambda_client = LambdaClient()

        if terminate_all:
            instances = state_mgr.list_instances()
            if not instances:
                click.echo("No instances found in state")
                return

            click.echo(f"Terminating {len(instances)} instance(s)...")
            for inst in instances:
                _terminate_instance(
                    lambda_client, state_mgr, inst.name, inst.instance_id, inst.public_ip
                )
        else:
            # Get instance name
            instance_name = name or config.get_instance_name()
            instance = state_mgr.get_instance(instance_name)

            if not instance:
                click.echo(f"No instance '{instance_name}' found in state")
                # Try to find by listing from API
                click.echo("Checking Lambda API directly...")
                api_instance = lambda_client.get_instance_by_name(instance_name)
                if api_instance:
                    click.echo(f"Found instance in API: {api_instance.id}")
                    _terminate_instance(
                        lambda_client, state_mgr, instance_name, api_instance.id, api_instance.ip
                    )
                else:
                    click.echo("Instance not found in API either")
                return

            _terminate_instance(
                lambda_client, state_mgr, instance.name, instance.instance_id, instance.public_ip
            )

    except LambdaAPIError as e:
        click.echo(click.style(f"Lambda API error: {e}", fg="red"))
        sys.exit(1)


@cli.command()
@click.option("--name", help="Instance name")
@click.option("--tailscale-authkey", help="Tailscale auth key")
@click.option("--no-tailscale", is_flag=True, help="Skip Tailscale setup even if authkey is available")
@click.option("--start-vllm", is_flag=True, help="Also install Docker and start vLLM container")
@click.option("--health-timeout", type=int, default=900, help="Health check timeout in seconds (default: 900)")
@click.option("--idle-timeout", type=int, help="Auto-shutdown after N minutes idle (0=disabled)")
def bootstrap(name, tailscale_authkey, no_tailscale, start_vllm, health_timeout, idle_timeout):
    """Run bootstrap on an existing instance.

    Sets up directories and environment file on the persistent filesystem.
    If Tailscale authkey is provided (or TS_AUTHKEY env var), also sets up Tailscale.
    Use --start-vllm to also install Docker and start the vLLM container.
    """
    try:
        config = get_config()
        state_mgr = get_state_manager()

        # Get instance
        instance_name = name or config.get_instance_name()
        instance = state_mgr.get_instance(instance_name)

        if not instance:
            click.echo(click.style(f"Error: Instance '{instance_name}' not found", fg="red"))
            sys.exit(1)

        if not instance.public_ip:
            click.echo(click.style(f"Error: Instance has no public IP", fg="red"))
            sys.exit(1)

        # Determine if we should use Tailscale bootstrap
        ts_authkey = None if no_tailscale else (tailscale_authkey or get_env_var("TS_AUTHKEY"))
        vllm_api_key = get_env_var("VLLM_API_KEY")
        hf_token = get_env_var("HUGGINGFACE_API_KEY")

        ssh_client = get_ssh_client_for_instance(instance)
        try:
            model_config = config.get_model_config(instance.model_alias)

            # Execute bootstrap using shared helper function
            # Convert idle_timeout from minutes to seconds if provided
            idle_timeout_seconds = idle_timeout * 60 if idle_timeout is not None else None
            _execute_bootstrap(
                ssh_client=ssh_client,
                instance=instance,
                instance_name=instance_name,
                filesystem_name=instance.filesystem,
                ts_authkey=ts_authkey,
                vllm_api_key=vllm_api_key,
                hf_token=hf_token,
                model_id=instance.model_id,
                model_revision=instance.model_revision,
                max_model_len=model_config.get("max_model_len", 16384),
                start_vllm=start_vllm,
                health_timeout=health_timeout,
                push_deploy=start_vllm,  # Only push deploy files if starting vLLM
                idle_timeout=idle_timeout_seconds,
            )

            fs_path = get_fs_path(instance.filesystem)
            click.echo(f"\nPersistent filesystem: {fs_path}")
            click.echo(f"  HF cache:    {fs_path}/hf-cache")
            click.echo(f"  Adapters:    {fs_path}/adapters")
            click.echo(f"  Env file:    {fs_path}/run/inference_server.env")

        except BootstrapError as e:
            click.echo(click.style(f"Bootstrap error: {e}", fg="red"))
            sys.exit(1)
        finally:
            ssh_client.close()

    except SSHError as e:
        click.echo(click.style(f"SSH error: {e}", fg="red"))
        sys.exit(1)

    except LambdaAPIError as e:
        click.echo(click.style(f"Lambda API error: {e}", fg="red"))
        sys.exit(1)


def _execute_bootstrap(
    ssh_client: SSHClient,
    instance: InstanceState,
    instance_name: str,
    filesystem_name: str,
    ts_authkey: str | None,
    vllm_api_key: str | None,
    hf_token: str | None,
    model_id: str,
    model_revision: str | None,
    max_model_len: int,
    start_vllm: bool = False,
    health_timeout: int | None = None,
    push_deploy: bool = True,
    idle_timeout: int | None = None,
    callback: Callable[[str], None] | None = None,
) -> dict[str, str] | None:
    """Execute bootstrap on an instance.

    Args:
        ssh_client: Connected SSH client.
        instance: Instance state object.
        instance_name: Name of the instance.
        filesystem_name: Filesystem name.
        ts_authkey: Tailscale auth key (None to skip Tailscale).
        vllm_api_key: vLLM API key (None to skip vLLM).
        hf_token: HuggingFace token.
        model_id: Model ID.
        model_revision: Model revision.
        max_model_len: Max model length.
        start_vllm: Whether to start vLLM (requires ts_authkey and vllm_api_key).
        health_timeout: Health check timeout in seconds.
        push_deploy: Whether to push deploy files first.
        idle_timeout: Idle timeout in seconds (0 to disable, None for default 3600).
        callback: Optional progress callback.

    Returns:
        Dict with bootstrap results (tailscale_ip, tailscale_hostname) or None.

    Raises:
        BootstrapError: On bootstrap failure.
    """
    config = get_config()
    state_mgr = get_state_manager()
    
    if callback is None:
        def bootstrap_callback(msg):
            click.echo(f"  {msg}")
        callback = bootstrap_callback
    
    try:
        model_config = config.get_model_config(instance.model_alias)
        remote_deploy = config.paths.get("remote_deploy", "~/inference_deploy")
        
        # Push deploy files if needed
        if push_deploy and (start_vllm or ts_authkey):
            click.echo(f"Pushing deploy files to {instance_name}...")
            deploy_dir = PROJECT_ROOT / "deploy"
            ssh_client.run(f"mkdir -p {remote_deploy}", timeout=30)
            exit_code, output = ssh_client.rsync(
                local_path=deploy_dir,
                remote_path=remote_deploy,
                exclude=["__pycache__", "*.pyc", ".DS_Store", ".env"],
            )
            if exit_code == 0:
                click.echo("  Deploy files pushed successfully")
            else:
                click.echo(click.style(f"Warning: rsync had issues: {output}", fg="yellow"))
        
        # Determine bootstrap type and execute
        if start_vllm and ts_authkey and vllm_api_key:
            # Full bootstrap: Tailscale + Docker + vLLM + Watchdog
            click.echo(f"Running full bootstrap on {instance_name} ({instance.public_ip})...")

            # Get Lambda API key for watchdog termination
            lambda_api_key = get_env_var("LAMBDA_API_KEY")
            if not lambda_api_key:
                click.echo(click.style(
                    "Warning: LAMBDA_API_KEY not set - idle watchdog cannot terminate instance",
                    fg="yellow"
                ))

            result = run_full_bootstrap(
                ssh_client,
                filesystem_name=filesystem_name,
                tailscale_authkey=ts_authkey,
                instance_name=instance_name,
                model_id=model_id,
                vllm_api_key=vllm_api_key,
                hf_token=hf_token,
                model_revision=model_revision,
                max_model_len=max_model_len,
                deploy_path=remote_deploy,
                instance_id=instance.instance_id,
                idle_timeout=idle_timeout,
                lambda_api_key=lambda_api_key,
                callback=callback,
            )
            
            # Update state with Tailscale info
            state_mgr.update_instance(
                instance_name,
                tailscale_ip=result["tailscale_ip"],
                tailscale_hostname=result["tailscale_hostname"],
            )
            
            click.echo(click.style("\nFull bootstrap complete!", fg="green"))
            click.echo(f"  Tailscale IP:       {result['tailscale_ip']}")
            click.echo(f"  Tailscale hostname: {result['tailscale_hostname']}")
            
            # Wait for health check
            click.echo("\nWaiting for vLLM to become ready...")
            vllm_url = f"http://{result['tailscale_ip']}:{config.vllm.get('port', 8000)}"
            vllm_client = VLLMClient(base_url=vllm_url, api_key=vllm_api_key)
            health_checker = HealthChecker(
                vllm_client,
                timeout=health_timeout or config.timeouts.get("health_check", 900),
                interval=config.timeouts.get("health_interval", 10),
            )
            
            try:
                health_checker.wait_for_ready(callback=callback)
                click.echo(click.style("\nvLLM is ready!", fg="green"))
                click.echo(f"\nvLLM endpoint: {vllm_url}/v1")
                click.echo(f"Use with: VLLM_API_KEY=*** curl {vllm_url}/v1/models")

                # Create manifest for reproducibility
                try:
                    # Get updated instance state
                    updated_instance = state_mgr.get_instance(instance_name)
                    if updated_instance:
                        vllm_info = {
                            "image_tag": config.vllm.get("image_tag"),
                            "port": config.vllm.get("port", 8000),
                            "max_model_len": max_model_len,
                        }
                        manifest_data = generate_manifest(updated_instance, vllm_info=vllm_info)
                        manifest_path = write_manifest(
                            ssh_client,
                            filesystem_name,
                            manifest_data,
                            instance_name,
                            callback=callback,
                        )
                        click.echo(f"Manifest created for reproducibility")
                except Exception as e:
                    click.echo(click.style(f"Warning: Could not create manifest: {e}", fg="yellow"))

            except TimeoutError as e:
                click.echo(click.style(f"\nHealth check timed out: {e}", fg="yellow"))
                click.echo("vLLM may still be loading. Check logs with:")
                click.echo(f"  inference-server ssh --name {instance_name} 'docker logs inference-vllm'")
                click.echo("\nYou can retry the health check by running:")
                click.echo(f"  inference-server bootstrap --name {instance_name} --start-vllm --health-timeout {health_timeout or config.timeouts.get('health_check', 900)}")

            return result
            
        elif ts_authkey:
            # Tailscale bootstrap only (no vLLM)
            click.echo(f"Running bootstrap with Tailscale on {instance_name} ({instance.public_ip})...")
            ts_info = run_bootstrap_with_tailscale(
                ssh_client,
                filesystem_name=filesystem_name,
                tailscale_authkey=ts_authkey,
                instance_name=instance_name,
                hf_token=hf_token,
                vllm_api_key=vllm_api_key,
                model_id=model_id,
                model_revision=model_revision,
                max_model_len=max_model_len,
                callback=callback,
            )
            
            # Update state with Tailscale info
            state_mgr.update_instance(
                instance_name,
                tailscale_ip=ts_info["tailscale_ip"],
                tailscale_hostname=ts_info["tailscale_hostname"],
            )
            
            click.echo(click.style("\nBootstrap with Tailscale complete!", fg="green"))
            click.echo(f"  Tailscale IP:       {ts_info['tailscale_ip']}")
            click.echo(f"  Tailscale hostname: {ts_info['tailscale_hostname']}")
            
            if start_vllm:
                click.echo(click.style("\nNote: --start-vllm requires VLLM_API_KEY", fg="yellow"))
            
            return ts_info
            
        else:
            # Lite bootstrap only (directories + env file)
            click.echo(f"Running bootstrap-lite on {instance_name} ({instance.public_ip})...")
            run_bootstrap_lite(
                ssh_client,
                filesystem_name=filesystem_name,
                hf_token=hf_token,
                vllm_api_key=vllm_api_key,
                model_id=model_id,
                model_revision=model_revision,
                max_model_len=max_model_len,
                callback=callback,
            )
            click.echo(click.style("\nBootstrap-lite complete!", fg="green"))
            
            if start_vllm:
                click.echo(click.style("\nNote: --start-vllm requires TS_AUTHKEY and VLLM_API_KEY", fg="yellow"))
            
            return None
            
    except BootstrapError as e:
        click.echo(click.style(f"Bootstrap error: {e}", fg="red"))
        raise


def _terminate_instance(
    lambda_client: LambdaClient,
    state_mgr,
    name: str,
    instance_id: str,
    public_ip: str | None = None,
):
    """Helper to terminate a single instance."""
    click.echo(f"Terminating '{name}' ({instance_id})...")

    # Try to logout from Tailscale before termination
    if public_ip:
        try:
            click.echo(f"  Logging out from Tailscale...")
            ssh_client = SSHClient(host=public_ip, user="ubuntu")
            ssh_client.connect(timeout=10)
            exit_code, _, stderr = ssh_client.run("sudo tailscale logout", timeout=30)
            if exit_code == 0:
                click.echo(click.style(f"  Tailscale logout successful", fg="green"))
            else:
                # Not an error - Tailscale might not be installed/connected
                click.echo(f"  Tailscale logout skipped (not connected or not installed)")
            ssh_client.close()
        except SSHError:
            # SSH failed - instance might already be shutting down
            click.echo(f"  Tailscale logout skipped (SSH unavailable)")

    try:
        success = lambda_client.terminate_instance(instance_id)
        if success:
            click.echo(click.style(f"  Terminated: {name}", fg="green"))
        else:
            click.echo(click.style(f"  Failed to terminate: {name}", fg="yellow"))
    except LambdaAPIError as e:
        if "not found" in str(e).lower() or e.status_code == 404:
            click.echo(f"  Instance already terminated: {name}")
        else:
            raise

    # Always remove from state
    state_mgr.delete_instance(name)
    click.echo(f"  Removed from state: {name}")


@cli.command()
@click.option("--name", help="Instance name to check")
def status(name):
    """Show instance state, Tailscale address, and service status."""
    try:
        config = get_config()
        state_mgr = get_state_manager()
        lambda_client = LambdaClient()

        if name:
            # Show specific instance
            _show_instance_status(lambda_client, state_mgr, name)
        else:
            # Show all instances
            instances = state_mgr.list_instances()
            if not instances:
                click.echo("No instances found in local state")
                click.echo("\nChecking Lambda API for any instances...")
                api_instances = lambda_client.list_instances()
                if api_instances:
                    click.echo(f"Found {len(api_instances)} instance(s) in Lambda Cloud:")
                    for inst in api_instances:
                        click.echo(f"  - {inst.name or inst.id}: {inst.status} ({inst.ip or 'no IP'})")
                else:
                    click.echo("No instances found in Lambda Cloud")
                return

            click.echo(f"Found {len(instances)} instance(s):\n")
            for inst in instances:
                _show_instance_status(lambda_client, state_mgr, inst.name, brief=len(instances) > 1)
                click.echo()

    except LambdaAPIError as e:
        click.echo(click.style(f"Lambda API error: {e}", fg="red"))
        sys.exit(1)


def _show_instance_status(lambda_client: LambdaClient, state_mgr, name: str, brief: bool = False):
    """Show status for a single instance."""
    instance = state_mgr.get_instance(name)

    if not instance:
        click.echo(f"Instance '{name}' not found in state")
        return

    # Get current status from API
    api_instance = lambda_client.get_instance(instance.instance_id)

    if api_instance:
        current_status = api_instance.status
        current_ip = api_instance.ip
        # Update state if changed
        if current_status != instance.status or current_ip != instance.public_ip:
            state_mgr.update_instance(name, status=current_status, public_ip=current_ip)
    else:
        current_status = "terminated"
        current_ip = None

    # Determine status color
    status_color = {
        "active": "green",
        "booting": "yellow",
        "terminated": "red",
        "terminating": "red",
        "unhealthy": "red",
    }.get(current_status, "white")

    status_str = click.style(current_status, fg=status_color)

    if brief:
        click.echo(f"  {name}: {status_str} | IP: {current_ip or 'N/A'}")
    else:
        click.echo(f"Instance: {name}")
        click.echo(f"  Status:      {status_str}")
        click.echo(f"  Instance ID: {instance.instance_id}")
        click.echo(f"  GPU:         {instance.gpu_type}")
        click.echo(f"  Region:      {instance.region}")
        click.echo(f"  Filesystem:  {instance.filesystem}")
        click.echo(f"  Public IP:   {current_ip or 'N/A'}")
        if instance.tailscale_ip:
            click.echo(f"  Tailscale:   {instance.tailscale_ip} ({instance.tailscale_hostname})")
        click.echo(f"  Model:       {instance.model_alias} ({instance.model_id})")
        click.echo(f"  Created:     {instance.created_at}")
        if instance.loaded_adapters:
            click.echo(f"  Adapters:    {', '.join(instance.loaded_adapters)}")


# =============================================================================
# Adapter Commands
# =============================================================================


@cli.command("sync-adapters")
@click.argument("adapter_name", required=False)
@click.option("--name", help="Instance name")
@click.option("--force", is_flag=True, help="Force re-sync even if checksums match")
@click.option("--delete", "delete_orphans", is_flag=True, help="Delete remote adapters not present locally")
@click.option("--local-path", type=click.Path(exists=True), help="Local adapters directory")
def sync_adapters(adapter_name, name, force, delete_orphans, local_path):
    """Sync adapters to persistent filesystem.

    Uses checksum comparison to only upload changed files.

    Without ADAPTER_NAME, syncs all adapters in the local adapters directory.
    With ADAPTER_NAME, syncs only that specific adapter.

    Examples:
        inference-server sync-adapters                    # Sync all
        inference-server sync-adapters my-lora           # Sync specific adapter
        inference-server sync-adapters --force           # Force re-upload all
        inference-server sync-adapters --delete          # Remove orphaned remote adapters
    """
    try:
        config = get_config()
        state_mgr = get_state_manager()

        # Get instance
        instance_name = name or config.get_instance_name()
        instance = state_mgr.get_instance(instance_name)

        if not instance:
            click.echo(click.style(f"Error: Instance '{instance_name}' not found", fg="red"))
            sys.exit(1)

        if not instance.public_ip:
            click.echo(click.style(f"Error: Instance has no public IP", fg="red"))
            sys.exit(1)

        # Resolve paths
        local_adapters = Path(local_path) if local_path else get_default_local_adapters_path()
        remote_adapters = get_remote_adapters_path(instance.filesystem)

        click.echo(f"Syncing adapters to {instance_name}")
        click.echo(f"  Local:  {local_adapters}")
        click.echo(f"  Remote: {remote_adapters}")
        click.echo()

        # Connect and sync
        ssh_client = get_ssh_client_for_instance(instance)
        try:
            # Ensure remote directory exists
            ssh_client.run(f"mkdir -p {remote_adapters}", timeout=30)

            sync_mgr = AdapterSyncManager(
                ssh_client=ssh_client,
                remote_adapters_path=remote_adapters,
                local_adapters_path=local_adapters,
            )

            def progress_callback(msg):
                click.echo(f"  {msg}")

            if adapter_name:
                # Sync single adapter
                was_uploaded = sync_mgr.sync_adapter(
                    adapter_name,
                    force=force,
                    callback=progress_callback,
                )
                if was_uploaded:
                    click.echo(click.style(f"\n✓ Adapter '{adapter_name}' synced", fg="green"))
                else:
                    click.echo(f"\n  Adapter '{adapter_name}' is up to date")
            else:
                # Sync all adapters
                result = sync_mgr.sync_all(
                    delete_orphans=delete_orphans,
                    callback=progress_callback,
                )

                click.echo()
                if result.uploaded:
                    click.echo(click.style(f"✓ Uploaded: {', '.join(result.uploaded)}", fg="green"))
                if result.skipped:
                    click.echo(f"  Skipped (unchanged): {', '.join(result.skipped)}")
                if result.deleted:
                    click.echo(click.style(f"  Deleted: {', '.join(result.deleted)}", fg="yellow"))
                if result.failed:
                    for name, err in result.failed:
                        click.echo(click.style(f"✗ Failed: {name} - {err}", fg="red"))
                    sys.exit(1)

                total = len(result.uploaded) + len(result.skipped)
                click.echo(f"\nSync complete: {total} adapter(s) processed")

        except SSHError as e:
            click.echo(click.style(f"SSH error: {e}", fg="red"))
            sys.exit(1)
        finally:
            ssh_client.close()

    except ConfigError as e:
        click.echo(click.style(f"Config error: {e}", fg="red"))
        sys.exit(1)


@cli.command("load-adapter")
@click.argument("adapter_name")
@click.option("--name", help="Instance name")
@click.option("--sync/--no-sync", default=True, help="Sync adapter before loading (default: yes)")
@click.option("--local-path", type=click.Path(exists=True), help="Local adapters directory")
def load_adapter(adapter_name, name, sync, local_path):
    """Load adapter into vLLM runtime.

    Syncs adapter if needed (unless --no-sync), then calls vLLM's load endpoint.
    The adapter will be available for inference using its name as the model parameter.

    Example:
        inference-server load-adapter my-lora
        curl -X POST .../v1/chat/completions -d '{"model": "my-lora", ...}'
    """
    try:
        config = get_config()
        state_mgr = get_state_manager()

        # Get instance
        instance_name = name or config.get_instance_name()
        instance = state_mgr.get_instance(instance_name)

        if not instance:
            click.echo(click.style(f"Error: Instance '{instance_name}' not found", fg="red"))
            sys.exit(1)

        if not instance.tailscale_ip:
            click.echo(click.style(f"Error: Instance has no Tailscale IP (vLLM not accessible)", fg="red"))
            sys.exit(1)

        vllm_api_key = get_env_var("VLLM_API_KEY")
        if not vllm_api_key:
            click.echo(click.style("Error: VLLM_API_KEY not set", fg="red"))
            sys.exit(1)

        # Sync adapter first if requested
        if sync and instance.public_ip:
            click.echo(f"Syncing adapter '{adapter_name}'...")
            local_adapters_path = Path(local_path) if local_path else get_default_local_adapters_path()
            remote_adapters_path = get_remote_adapters_path(instance.filesystem)

            ssh_client = get_ssh_client_for_instance(instance)
            try:
                sync_mgr = AdapterSyncManager(
                    ssh_client=ssh_client,
                    remote_adapters_path=remote_adapters_path,
                    local_adapters_path=local_adapters_path,
                )

                def progress_callback(msg):
                    click.echo(f"  {msg}")

                try:
                    was_uploaded = sync_mgr.sync_adapter(adapter_name, callback=progress_callback)
                    if was_uploaded:
                        click.echo(click.style(f"  Adapter synced", fg="green"))
                    else:
                        click.echo(f"  Adapter already up to date")
                except SSHError as e:
                    click.echo(click.style(f"  Sync failed: {e}", fg="yellow"))
                    click.echo("  Continuing with load (adapter may already be on remote)...")
            finally:
                ssh_client.close()

        # Load adapter via vLLM API
        click.echo(f"\nLoading adapter '{adapter_name}' into vLLM...")
        # Use container-internal path, not host path
        # The docker-compose mounts ${FS_PATH}/adapters to /adapters inside the container
        container_adapter_path = f"/adapters/{adapter_name}"
        click.echo(f"  Adapter path (container): {container_adapter_path}")

        vllm_url = f"http://{instance.tailscale_ip}:{config.vllm.get('port', 8000)}"
        vllm_client = VLLMClient(base_url=vllm_url, api_key=vllm_api_key)

        import time
        
        try:
            # Check if adapter is already loaded and working
            # Note: vLLM may not show adapters in /v1/models, so we test with actual request
            click.echo("  Checking current adapter status...")
            try:
                # Try a tiny test request to see if adapter works
                vllm_client.chat_completion(
                    messages=[{"role": "user", "content": "Hi"}],
                    model=adapter_name,
                    max_tokens=1,
                )
                # If we get here, adapter is working!
                click.echo(click.style(f"\n✓ Adapter '{adapter_name}' is already loaded and working!", fg="green"))
                click.echo(f"\nUse in requests with: \"model\": \"{adapter_name}\"")
                
                # Update instance state
                loaded = instance.loaded_adapters or []
                if adapter_name not in loaded:
                    loaded.append(adapter_name)
                    state_mgr.update_instance(instance_name, loaded_adapters=loaded)
                return
            except VLLMError:
                # Adapter not working, continue to load it
                pass

            # Load adapter
            click.echo("  Calling vLLM load endpoint...")
            try:
                load_response = vllm_client.load_lora_adapter(
                    lora_name=adapter_name,
                    lora_path=container_adapter_path,
                )
                click.echo(f"  Load endpoint returned: {load_response}")
            except VLLMError as load_error:
                # Check if error is "already loaded"
                error_str = str(load_error).lower()
                if "already" in error_str and "loaded" in error_str:
                    click.echo("  Adapter reported as already loaded, verifying...")
                    # Check if it's actually in /v1/models
                    model_ids = vllm_client.get_model_ids()
                    if adapter_name in model_ids:
                        click.echo(click.style(f"\n✓ Adapter '{adapter_name}' is loaded!", fg="green"))
                        click.echo(f"\nUse in requests with: \"model\": \"{adapter_name}\"")
                        
                        # Update instance state
                        loaded = instance.loaded_adapters or []
                        if adapter_name not in loaded:
                            loaded.append(adapter_name)
                            state_mgr.update_instance(instance_name, loaded_adapters=loaded)
                        return
                    else:
                        # vLLM thinks it's loaded but it's not in models - unload and reload
                        click.echo("  Adapter not in /v1/models, unloading and reloading...")
                        try:
                            vllm_client.unload_lora_adapter(adapter_name)
                            click.echo("  Unloaded successfully, reloading...")
                            time.sleep(1)  # Brief pause
                            load_response = vllm_client.load_lora_adapter(
                                lora_name=adapter_name,
                                lora_path=container_adapter_path,
                            )
                        except VLLMError as unload_error:
                            click.echo(click.style(
                                f"  Warning: Could not unload: {unload_error}",
                                fg="yellow"
                            ))
                            click.echo("  Continuing to verify...")
                else:
                    # Some other error, re-raise it
                    raise
            
            # Verify adapter is working
            # Note: vLLM may not show adapters in /v1/models even when loaded
            # So we verify by actually trying to use it in a test request
            click.echo("Verifying adapter is working...")
            max_retries = 5
            retry_delay = 2.0
            
            # First check /v1/models
            model_ids = vllm_client.get_model_ids()
            in_models_list = adapter_name in model_ids
            
            # Then test with actual inference request
            adapter_works = False
            for attempt in range(max_retries):
                try:
                    # Try a tiny test request with the adapter
                    test_response = vllm_client.chat_completion(
                        messages=[{"role": "user", "content": "Hi"}],
                        model=adapter_name,
                        max_tokens=2,
                    )
                    # If we get here, the adapter works!
                    adapter_works = True
                    break
                except VLLMError as e:
                    error_str = str(e).lower()
                    if "does not exist" in error_str or "not found" in error_str:
                        # Adapter definitely doesn't work
                        if attempt < max_retries - 1:
                            click.echo(f"  Test request failed (attempt {attempt + 1}/{max_retries}), retrying...")
                            time.sleep(retry_delay)
                        else:
                            adapter_works = False
                            break
                    else:
                        # Some other error - might be transient
                        if attempt < max_retries - 1:
                            click.echo(f"  Test request error (attempt {attempt + 1}/{max_retries}): {e}, retrying...")
                            time.sleep(retry_delay)
                        else:
                            # Final attempt - assume it doesn't work
                            adapter_works = False
                            break
            
            if adapter_works:
                click.echo(click.style(f"\n✓ Adapter '{adapter_name}' loaded and working!", fg="green"))
                if not in_models_list:
                    click.echo(click.style(
                        "  Note: Adapter works but doesn't appear in /v1/models (known vLLM behavior)",
                        fg="yellow"
                    ))
                click.echo(f"\nUse in requests with: \"model\": \"{adapter_name}\"")
                
                # Update instance state
                loaded = instance.loaded_adapters or []
                if adapter_name not in loaded:
                    loaded.append(adapter_name)
                    state_mgr.update_instance(instance_name, loaded_adapters=loaded)
            else:
                # Adapter doesn't work
                available_models = ", ".join(model_ids) if model_ids else "none"
                click.echo(click.style(
                    f"\n✗ Adapter '{adapter_name}' not working after loading",
                    fg="yellow"
                ))
                click.echo(f"  Available models in /v1/models: {available_models}")
                click.echo(click.style(
                    "  Warning: Load endpoint returned success but adapter not functional",
                    fg="yellow"
                ))
                click.echo("\n  Possible causes:")
                click.echo("    1. vLLM runtime LoRA updating may not be enabled")
                click.echo("       Check: VLLM_ALLOW_RUNTIME_LORA_UPDATING=true in docker-compose.yml")
                click.echo("    2. Adapter files may be missing or corrupted:")
                click.echo(f"       Container path: {container_adapter_path}")
                click.echo(f"       Host path: {get_remote_adapters_path(instance.filesystem)}/{adapter_name}")
                click.echo("    3. vLLM version may not support runtime LoRA updates")
                click.echo("    4. Check vLLM logs for errors: docker compose logs vllm")
                click.echo("\n  To investigate:")
                click.echo(f"    curl -X POST -H 'Authorization: Bearer ...' -H 'Content-Type: application/json' \\")
                click.echo(f"      -d '{{\"model\": \"{adapter_name}\", \"messages\": [{{\"role\": \"user\", \"content\": \"Hi\"}}], \"max_tokens\": 2}}' \\")
                click.echo(f"      {vllm_url}/v1/chat/completions")
                # Don't exit - let user investigate

        except VLLMError as e:
            click.echo(click.style(f"\n✗ Failed to load adapter: {e}", fg="red"))
            sys.exit(1)

    except ConfigError as e:
        click.echo(click.style(f"Config error: {e}", fg="red"))
        sys.exit(1)


@cli.command("unload-adapter")
@click.argument("adapter_name")
@click.option("--name", help="Instance name")
def unload_adapter(adapter_name, name):
    """Unload adapter from vLLM runtime.

    Removes the adapter from vLLM's active adapters. The adapter files
    remain on the persistent filesystem for future loading.
    """
    try:
        config = get_config()
        state_mgr = get_state_manager()

        # Get instance
        instance_name = name or config.get_instance_name()
        instance = state_mgr.get_instance(instance_name)

        if not instance:
            click.echo(click.style(f"Error: Instance '{instance_name}' not found", fg="red"))
            sys.exit(1)

        if not instance.tailscale_ip:
            click.echo(click.style(f"Error: Instance has no Tailscale IP (vLLM not accessible)", fg="red"))
            sys.exit(1)

        vllm_api_key = get_env_var("VLLM_API_KEY")
        if not vllm_api_key:
            click.echo(click.style("Error: VLLM_API_KEY not set", fg="red"))
            sys.exit(1)

        # Unload adapter via vLLM API
        click.echo(f"Unloading adapter '{adapter_name}' from vLLM...")

        vllm_url = f"http://{instance.tailscale_ip}:{config.vllm.get('port', 8000)}"
        vllm_client = VLLMClient(base_url=vllm_url, api_key=vllm_api_key)

        try:
            vllm_client.unload_lora_adapter(lora_name=adapter_name)
            click.echo(click.style(f"\n✓ Adapter '{adapter_name}' unloaded", fg="green"))

            # Update instance state
            loaded = instance.loaded_adapters or []
            if adapter_name in loaded:
                loaded.remove(adapter_name)
                state_mgr.update_instance(instance_name, loaded_adapters=loaded)

        except VLLMError as e:
            click.echo(click.style(f"\n✗ Failed to unload adapter: {e}", fg="red"))
            sys.exit(1)

    except ConfigError as e:
        click.echo(click.style(f"Config error: {e}", fg="red"))
        sys.exit(1)


@cli.command("list-adapters")
@click.option("--name", help="Instance name")
@click.option("--local-only", is_flag=True, help="Only show local adapters")
@click.option("--remote-only", is_flag=True, help="Only show remote adapters")
@click.option("--local-path", type=click.Path(exists=True), help="Local adapters directory")
def list_adapters(name, local_only, remote_only, local_path):
    """List adapters: local, remote, and loaded.

    Shows adapters in the local directory, on the remote persistent filesystem,
    and which are currently loaded in vLLM.
    """
    try:
        config = get_config()
        state_mgr = get_state_manager()

        # Get local adapters
        local_adapters_path = Path(local_path) if local_path else get_default_local_adapters_path()
        local_adapters = discover_local_adapters(local_adapters_path)

        if local_only:
            click.echo(f"Local adapters ({local_adapters_path}):")
            if not local_adapters:
                click.echo("  (none)")
            else:
                for adapter in local_adapters:
                    size = sum(f.stat().st_size for f in adapter.local_path.iterdir() if f.is_file())
                    size_mb = size / (1024 * 1024)
                    click.echo(f"  {adapter.name}")
                    click.echo(f"    Checksum: {adapter.checksum[:12]}...")
                    click.echo(f"    Size:     {size_mb:.1f} MB")
                    click.echo(f"    Files:    {', '.join(adapter.files.keys())}")
            return

        # Get instance for remote adapters
        instance_name = name or config.get_instance_name()
        instance = state_mgr.get_instance(instance_name)

        if not instance:
            if not local_only:
                click.echo(click.style(f"Warning: Instance '{instance_name}' not found, showing local only", fg="yellow"))
                click.echo()

            click.echo(f"Local adapters ({local_adapters_path}):")
            if not local_adapters:
                click.echo("  (none)")
            else:
                for adapter in local_adapters:
                    click.echo(f"  - {adapter.name} (checksum: {adapter.checksum[:12]}...)")
            return

        # Get remote adapters
        remote_adapters_path = get_remote_adapters_path(instance.filesystem)
        remote_adapters = []
        loaded_adapters = []

        if instance.public_ip:
            ssh_client = get_ssh_client_for_instance(instance)
            try:
                sync_mgr = AdapterSyncManager(
                    ssh_client=ssh_client,
                    remote_adapters_path=remote_adapters_path,
                    local_adapters_path=local_adapters_path,
                )
                remote_adapters = sync_mgr.list_remote_adapters()

                # Get loaded adapters from vLLM if Tailscale is available
                if instance.tailscale_ip:
                    vllm_api_key = get_env_var("VLLM_API_KEY")
                    if vllm_api_key:
                        try:
                            vllm_url = f"http://{instance.tailscale_ip}:{config.vllm.get('port', 8000)}"
                            vllm_client = VLLMClient(base_url=vllm_url, api_key=vllm_api_key)
                            model_ids = vllm_client.get_model_ids()
                            # Filter out base model, keep only adapter names
                            base_model = instance.model_id
                            loaded_adapters = [m for m in model_ids if m != base_model]
                        except VLLMError:
                            pass  # vLLM not available
            except SSHError as e:
                click.echo(click.style(f"Warning: Could not connect to instance: {e}", fg="yellow"))
            finally:
                ssh_client.close()

        if remote_only:
            click.echo(f"Remote adapters ({remote_adapters_path}):")
            if not remote_adapters:
                click.echo("  (none)")
            else:
                for adapter in remote_adapters:
                    click.echo(f"  {adapter.name}")
                    click.echo(f"    Checksum: {adapter.checksum[:12] if adapter.checksum else 'unknown'}...")
            return

        # Show all
        click.echo(f"Instance: {instance_name}")
        click.echo()

        # Local adapters
        click.echo(f"Local adapters ({local_adapters_path}):")
        if not local_adapters:
            click.echo("  (none)")
        else:
            local_names = {a.name for a in local_adapters}
            remote_names = {a.name for a in remote_adapters}
            remote_checksums = {a.name: a.checksum for a in remote_adapters}

            for adapter in local_adapters:
                status = ""
                if adapter.name in remote_names:
                    if remote_checksums.get(adapter.name) == adapter.checksum:
                        status = click.style(" [synced]", fg="green")
                    else:
                        status = click.style(" [out of sync]", fg="yellow")
                else:
                    status = click.style(" [not synced]", fg="yellow")
                click.echo(f"  - {adapter.name}{status}")

        click.echo()

        # Remote adapters
        click.echo(f"Remote adapters ({remote_adapters_path}):")
        if not remote_adapters:
            click.echo("  (none)")
        else:
            local_names = {a.name for a in local_adapters}
            for adapter in remote_adapters:
                status = ""
                if adapter.name in loaded_adapters:
                    status = click.style(" [loaded]", fg="cyan")
                elif adapter.name not in local_names:
                    status = click.style(" [orphan]", fg="yellow")
                click.echo(f"  - {adapter.name}{status}")

        # Loaded adapters
        if loaded_adapters:
            click.echo()
            click.echo("Currently loaded in vLLM:")
            for adapter in loaded_adapters:
                click.echo(f"  - {adapter}")

    except ConfigError as e:
        click.echo(click.style(f"Config error: {e}", fg="red"))
        sys.exit(1)


# =============================================================================
# Utility Commands
# =============================================================================


@cli.command("docker")
@click.option("--name", help="Instance name")
@click.argument("docker_command", nargs=-1, required=True)
def docker_cmd(docker_command, name):
    """Run docker commands on the instance with proper permissions.
    
    This helper ensures docker commands work even if the user hasn't
    logged out/in after being added to the docker group.
    
    Examples:
        inference-server docker --name test ps -a
        inference-server docker --name test logs inference-vllm
        inference-server docker --name test compose ps
        
    Note: Put --name before the docker command, or use -- to separate:
        inference-server docker ps -a --name test
        inference-server docker -- ps -a --name test
    """
    try:
        config = get_config()
        state_mgr = get_state_manager()
        
        # Get instance
        instance_name = name or config.get_instance_name()
        instance = state_mgr.get_instance(instance_name)
        
        if not instance:
            click.echo(click.style(f"Error: Instance '{instance_name}' not found", fg="red"))
            sys.exit(1)
        
        if not instance.public_ip:
            click.echo(click.style(f"Error: Instance '{instance_name}' has no public IP", fg="red"))
            sys.exit(1)
        
        # Build docker command
        docker_cmd_str = " ".join(docker_command)
        
        # Check if this is a compose command - if so, need to cd to deploy directory
        is_compose_cmd = docker_cmd_str.startswith("compose")
        remote_deploy = config.paths.get("remote_deploy", "~/inference_deploy")
        
        ssh_client = get_ssh_client_for_instance(instance)
        try:
            # Use the same pattern as _run_docker_cmd: try direct, then sg docker, then sudo
            # First test if docker works without sudo
            test_cmd = "docker info >/dev/null 2>&1"
            test_exit, _, _ = ssh_client.run(test_cmd, timeout=10)
            
            if test_exit == 0:
                # Docker works directly, use normal command
                if is_compose_cmd:
                    # Change to deploy directory for compose commands
                    full_cmd = f"cd {remote_deploy} && docker {docker_cmd_str}"
                else:
                    full_cmd = f"docker {docker_cmd_str}"
            else:
                # Docker doesn't work directly, try sg docker (activates group in subshell)
                if is_compose_cmd:
                    # Change to deploy directory for compose commands
                    full_cmd = f"sg docker -c 'cd {remote_deploy} && docker {docker_cmd_str}'"
                else:
                    full_cmd = f"sg docker -c 'docker {docker_cmd_str}'"
            
            click.echo(f"Running docker command on {instance_name} ({instance.public_ip})...")
            exit_code, stdout, stderr = ssh_client.run(full_cmd, timeout=300)
            
            # If sg docker failed, try sudo as fallback
            if exit_code != 0 and "sg docker" in full_cmd:
                if is_compose_cmd:
                    full_cmd = f"cd {remote_deploy} && sudo docker {docker_cmd_str}"
                else:
                    full_cmd = f"sudo docker {docker_cmd_str}"
                exit_code, stdout, stderr = ssh_client.run(full_cmd, timeout=300)
            if stdout:
                click.echo(stdout, nl=False)
            if stderr:
                click.echo(click.style(stderr, fg="yellow"), nl=False)
            sys.exit(exit_code)
        except SSHError as e:
            click.echo(click.style(f"SSH error: {e}", fg="red"))
            sys.exit(1)
        finally:
            ssh_client.close()
            
    except SSHError as e:
        click.echo(click.style(f"SSH error: {e}", fg="red"))
        sys.exit(1)


@cli.command("docker-status")
@click.option("--name", help="Instance name")
def docker_status(name):
    """Check Docker daemon and container status on the instance."""
    try:
        config = get_config()
        state_mgr = get_state_manager()
        instance_name = name or config.get_instance_name()
        instance = state_mgr.get_instance(instance_name)
        
        if not instance or not instance.public_ip:
            click.echo(click.style(f"Error: Instance '{instance_name}' not found or has no IP", fg="red"))
            sys.exit(1)
        
        ssh_client = get_ssh_client_for_instance(instance)
        try:
            # Test docker access
            test_cmd = "docker info >/dev/null 2>&1"
            test_exit, _, _ = ssh_client.run(test_cmd, timeout=10)
            if test_exit != 0:
                test_exit, _, _ = ssh_client.run(f"sg docker -c '{test_cmd}'", timeout=10)
            if test_exit != 0:
                test_exit, _, _ = ssh_client.run(f"sudo {test_cmd}", timeout=10)
            
            if test_exit == 0:
                click.echo(click.style("✓ Docker daemon: running", fg="green"))
            else:
                click.echo(click.style("✗ Docker daemon: not running", fg="red"))
                return
            
            # Get container status
            ps_cmd = "docker ps -a --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'"
            exit_code, stdout, _ = ssh_client.run(ps_cmd, timeout=10)
            if exit_code != 0:
                exit_code, stdout, _ = ssh_client.run(f"sg docker -c '{ps_cmd}'", timeout=10)
            if exit_code != 0:
                exit_code, stdout, _ = ssh_client.run(f"sudo {ps_cmd}", timeout=10)
            
            if exit_code == 0 and stdout.strip():
                click.echo("\nContainers:")
                click.echo(stdout.strip())
            else:
                click.echo("\nNo containers found")
            
            # Check vLLM container specifically
            vllm_cmd = "docker inspect inference-vllm --format '{{.State.Status}}' 2>/dev/null"
            exit_code, status, _ = ssh_client.run(vllm_cmd, timeout=10)
            if exit_code != 0:
                exit_code, status, _ = ssh_client.run(f"sg docker -c '{vllm_cmd}'", timeout=10)
            if exit_code != 0:
                exit_code, status, _ = ssh_client.run(f"sudo {vllm_cmd}", timeout=10)
            
            if exit_code == 0 and status.strip():
                status_str = status.strip()
                status_color = "green" if status_str == "running" else "yellow"
                click.echo(f"\nvLLM container: {click.style(status_str, fg=status_color)}")
        finally:
            ssh_client.close()
            
    except SSHError as e:
        click.echo(click.style(f"SSH error: {e}", fg="red"))
        sys.exit(1)


@cli.command("watchdog")
@click.option("--name", help="Instance name")
@click.option("--follow", "-f", is_flag=True, help="Follow logs in real-time")
@click.option("--tail", "-n", default=20, help="Number of log lines to show (default: 20)")
def watchdog_logs(name, follow, tail):
    """View watchdog logs and idle monitoring status.

    Shows heartbeat age, idle timeout threshold, and grace period status.
    Use --follow to watch logs in real-time.

    Examples:
        inference-server watchdog
        inference-server watchdog --follow
        inference-server watchdog --tail 50
    """
    try:
        config = get_config()
        state_mgr = get_state_manager()
        instance_name = name or config.get_instance_name()
        instance = state_mgr.get_instance(instance_name)

        if not instance or not instance.public_ip:
            click.echo(click.style(f"Error: Instance '{instance_name}' not found or has no IP", fg="red"))
            sys.exit(1)

        ssh_client = get_ssh_client_for_instance(instance)
        try:
            # Build docker logs command
            follow_flag = "-f" if follow else ""
            cmd = f"cd ~/inference_deploy && docker compose logs {follow_flag} --tail {tail} watchdog"

            click.echo(f"Watchdog logs for {instance_name}:\n")

            if follow:
                # For follow mode, use interactive execution
                import subprocess
                ssh_cmd = [
                    "ssh", "-o", "StrictHostKeyChecking=no",
                    "-i", str(Path(config.paths.get("ssh_key", "~/.ssh/id_ed25519")).expanduser()),
                    f"ubuntu@{instance.public_ip}",
                    cmd
                ]
                try:
                    subprocess.run(ssh_cmd)
                except KeyboardInterrupt:
                    click.echo("\nStopped following logs")
            else:
                exit_code, stdout, stderr = ssh_client.run(cmd, timeout=30)
                if stdout:
                    click.echo(stdout.strip())
                if stderr and exit_code != 0:
                    click.echo(click.style(stderr, fg="red"))

        finally:
            ssh_client.close()

    except SSHError as e:
        click.echo(click.style(f"SSH error: {e}", fg="red"))
        sys.exit(1)


@cli.command("set-idle-timeout")
@click.argument("minutes", type=int)
@click.option("--name", help="Instance name")
def set_idle_timeout(minutes, name):
    """Set the idle timeout for auto-shutdown.

    MINUTES is the idle time before the instance is terminated.
    Set to 0 to disable auto-termination (monitoring only).

    Examples:
        inference-server set-idle-timeout 5      # Terminate after 5 min idle
        inference-server set-idle-timeout 60     # Terminate after 1 hour idle
        inference-server set-idle-timeout 0      # Disable auto-termination
    """
    try:
        config = get_config()
        state_mgr = get_state_manager()
        instance_name = name or config.get_instance_name()
        instance = state_mgr.get_instance(instance_name)

        if not instance or not instance.public_ip:
            click.echo(click.style(f"Error: Instance '{instance_name}' not found or has no IP", fg="red"))
            sys.exit(1)

        # Convert minutes to seconds
        timeout_seconds = minutes * 60

        ssh_client = get_ssh_client_for_instance(instance)
        try:
            # Update .env file
            click.echo(f"Setting idle timeout to {minutes} minutes ({timeout_seconds} seconds)...")

            # Check if IDLE_TIMEOUT exists in .env
            check_cmd = "grep -q '^IDLE_TIMEOUT=' ~/inference_deploy/.env"
            exit_code, _, _ = ssh_client.run(check_cmd, timeout=10)

            if exit_code == 0:
                # Update existing value
                update_cmd = f"sed -i 's/^IDLE_TIMEOUT=.*/IDLE_TIMEOUT={timeout_seconds}/' ~/inference_deploy/.env"
            else:
                # Add new value
                update_cmd = f"echo 'IDLE_TIMEOUT={timeout_seconds}' >> ~/inference_deploy/.env"

            exit_code, _, stderr = ssh_client.run(update_cmd, timeout=10)
            if exit_code != 0:
                click.echo(click.style(f"Error updating .env: {stderr}", fg="red"))
                sys.exit(1)

            # Verify the change
            verify_cmd = "grep '^IDLE_TIMEOUT=' ~/inference_deploy/.env"
            exit_code, stdout, _ = ssh_client.run(verify_cmd, timeout=10)
            if exit_code == 0:
                click.echo(f"  Updated: {stdout.strip()}")

            # Recreate watchdog to pick up new env value (restart doesn't reload .env)
            click.echo("Recreating watchdog container...")
            restart_cmd = "cd ~/inference_deploy && docker compose up -d --force-recreate watchdog"
            exit_code, stdout, stderr = ssh_client.run(restart_cmd, timeout=60)

            if exit_code == 0:
                click.echo(click.style("✓ Watchdog restarted with new timeout", fg="green"))
                if minutes == 0:
                    click.echo("  Auto-termination is now DISABLED (monitoring only)")
                else:
                    click.echo(f"  Instance will terminate after {minutes} minutes of no API calls")
                    click.echo(f"  (Note: 10-minute grace period still applies after watchdog restart)")
            else:
                click.echo(click.style(f"Error restarting watchdog: {stderr}", fg="red"))
                sys.exit(1)

        finally:
            ssh_client.close()

    except SSHError as e:
        click.echo(click.style(f"SSH error: {e}", fg="red"))
        sys.exit(1)


@cli.command("heartbeat")
@click.option("--name", help="Instance name")
def heartbeat_status(name):
    """Check heartbeat status via the proxy.

    Shows current heartbeat age, which resets to 0 on each API call.
    The watchdog terminates the instance if heartbeat age exceeds idle timeout.

    Examples:
        inference-server heartbeat
        inference-server heartbeat --name my-instance
    """
    try:
        config = get_config()
        state_mgr = get_state_manager()
        instance_name = name or config.get_instance_name()
        instance = state_mgr.get_instance(instance_name)

        if not instance:
            click.echo(click.style(f"Error: Instance '{instance_name}' not found", fg="red"))
            sys.exit(1)

        # Get Tailscale IP for proxy access
        tailscale_ip = instance.tailscale_ip
        if not tailscale_ip:
            click.echo(click.style(f"Error: Instance '{instance_name}' has no Tailscale IP", fg="red"))
            click.echo("Heartbeat status requires Tailscale connectivity")
            sys.exit(1)

        # Fetch proxy status
        import urllib.request
        import json

        url = f"http://{tailscale_ip}:8000/proxy/status"
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                data = json.loads(response.read().decode())
        except urllib.error.URLError as e:
            click.echo(click.style(f"Error: Cannot reach proxy at {url}", fg="red"))
            click.echo(f"  {e}")
            sys.exit(1)

        # Display status
        click.echo(f"Heartbeat status for {instance_name}:")
        click.echo(f"  Proxy:          {click.style('healthy', fg='green')}")
        click.echo(f"  Tailscale IP:   {tailscale_ip}")

        heartbeat_age = data.get("heartbeat_age_seconds")
        if heartbeat_age is None:
            click.echo(f"  Heartbeat age:  {click.style('no heartbeat yet (no API calls)', fg='yellow')}")
        else:
            # Format nicely
            if heartbeat_age < 60:
                age_str = f"{heartbeat_age:.0f} seconds"
            elif heartbeat_age < 3600:
                age_str = f"{heartbeat_age / 60:.1f} minutes"
            else:
                age_str = f"{heartbeat_age / 3600:.1f} hours"

            # Color based on age (green < 10min, yellow < 30min, red > 30min)
            if heartbeat_age < 600:
                color = "green"
            elif heartbeat_age < 1800:
                color = "yellow"
            else:
                color = "red"

            click.echo(f"  Heartbeat age:  {click.style(age_str, fg=color)}")

        click.echo(f"  vLLM upstream:  {data.get('vllm_upstream', 'unknown')}")

    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"))
        sys.exit(1)


@cli.command()
@click.argument("command", required=False)
@click.option("--name", help="Instance name")
def ssh(command, name):
    """SSH to instance.

    Without arguments, opens interactive shell.
    With arguments, executes command and returns.
    """
    try:
        config = get_config()
        state_mgr = get_state_manager()

        # Get instance
        instance_name = name or config.get_instance_name()
        instance = state_mgr.get_instance(instance_name)

        if not instance:
            click.echo(click.style(f"Error: Instance '{instance_name}' not found", fg="red"))
            click.echo("Run 'status' to see available instances")
            sys.exit(1)

        if not instance.public_ip:
            click.echo(click.style(f"Error: Instance '{instance_name}' has no public IP", fg="red"))
            sys.exit(1)

        ssh_client = get_ssh_client_for_instance(instance)

        if command:
            # Execute command and return output
            click.echo(f"Running on {instance_name} ({instance.public_ip})...")
            try:
                # If command starts with 'docker', use proper permission handling
                if command.strip().startswith('docker '):
                    # Extract docker command
                    docker_cmd = command.strip()[7:]  # Remove 'docker ' prefix
                    
                    # Test if docker works directly
                    test_exit, _, _ = ssh_client.run("docker info >/dev/null 2>&1", timeout=10)
                    
                    if test_exit == 0:
                        # Docker works directly
                        full_cmd = command
                    else:
                        # Use sg docker to activate group
                        full_cmd = f"sg docker -c '{command}'"
                    
                    exit_code, stdout, stderr = ssh_client.run(full_cmd, timeout=300)
                    
                    # If sg docker failed, try sudo as fallback
                    if exit_code != 0 and "sg docker" in full_cmd:
                        full_cmd = f"sudo {command}"
                        exit_code, stdout, stderr = ssh_client.run(full_cmd, timeout=300)
                else:
                    # Regular command, run as-is
                    exit_code, stdout, stderr = ssh_client.run(command, timeout=300)
                
                if stdout:
                    click.echo(stdout, nl=False)
                if stderr:
                    click.echo(click.style(stderr, fg="yellow"), nl=False)
                sys.exit(exit_code)
            except SSHError as e:
                click.echo(click.style(f"SSH error: {e}", fg="red"))
                sys.exit(1)
            finally:
                ssh_client.close()
        else:
            # Interactive shell - this replaces the current process
            click.echo(f"Connecting to {instance_name} ({instance.public_ip})...")
            ssh_client.interactive_shell()

    except SSHError as e:
        click.echo(click.style(f"SSH error: {e}", fg="red"))
        sys.exit(1)


@cli.command("push-deploy")
@click.option("--name", help="Instance name")
@click.option("--dry-run", is_flag=True, help="Show what would be transferred")
def push_deploy(name, dry_run):
    """Push deploy/ directory to instance.

    Rsyncs the local deploy/ directory to ~/inference_deploy on the remote instance.
    """
    try:
        config = get_config()
        state_mgr = get_state_manager()

        # Get instance
        instance_name = name or config.get_instance_name()
        instance = state_mgr.get_instance(instance_name)

        if not instance:
            click.echo(click.style(f"Error: Instance '{instance_name}' not found", fg="red"))
            sys.exit(1)

        if not instance.public_ip:
            click.echo(click.style(f"Error: Instance '{instance_name}' has no public IP", fg="red"))
            sys.exit(1)

        # Find deploy directory
        deploy_dir = PROJECT_ROOT / "deploy"
        if not deploy_dir.exists():
            click.echo(click.style(f"Error: Deploy directory not found: {deploy_dir}", fg="red"))
            sys.exit(1)

        remote_path = config.paths.get("remote_deploy", "~/inference_deploy")
        ssh_client = get_ssh_client_for_instance(instance)

        click.echo(f"Pushing deploy/ to {instance_name}...")
        click.echo(f"  Local:  {deploy_dir}")
        click.echo(f"  Remote: {instance.public_ip}:{remote_path}")

        if dry_run:
            click.echo("\n[DRY RUN] Would transfer:")

        try:
            # First ensure remote directory exists
            if not dry_run:
                ssh_client.run(f"mkdir -p {remote_path}", timeout=30)

            # Rsync deploy directory
            exit_code, output = ssh_client.rsync(
                local_path=deploy_dir,
                remote_path=remote_path,
                exclude=["__pycache__", "*.pyc", ".DS_Store"],
                dry_run=dry_run,
            )

            if output:
                click.echo(output)

            if exit_code == 0:
                if dry_run:
                    click.echo(click.style("\n[DRY RUN] No files transferred", fg="yellow"))
                else:
                    click.echo(click.style("\nDeploy files pushed successfully!", fg="green"))
                    click.echo(f"\nVerify with: ./inference_server.py ssh --name {instance_name} 'ls -la {remote_path}'")
            else:
                click.echo(click.style(f"\nRsync failed with exit code {exit_code}", fg="red"))
                sys.exit(1)

        except SSHError as e:
            click.echo(click.style(f"Error: {e}", fg="red"))
            sys.exit(1)
        finally:
            ssh_client.close()

    except SSHError as e:
        click.echo(click.style(f"SSH error: {e}", fg="red"))
        sys.exit(1)


@cli.command()
@click.argument("service", required=False, type=click.Choice(["vllm", "bootstrap", "watchdog"]))
@click.option("--name", help="Instance name")
@click.option("--tail", "-n", type=int, default=100, help="Number of lines to show")
def logs(service, name, tail):
    """Fetch logs from instance.

    SERVICE can be: vllm, bootstrap, watchdog
    """
    click.echo("Command 'logs' not yet implemented (Phase 5)")
    click.echo(f"  service: {service or 'all'}")
    click.echo(f"  tail: {tail}")


@cli.command()
@click.option("--name", help="Instance name")
def models(name):
    """Print loaded models from /v1/models."""
    click.echo("Command 'models' not yet implemented (Phase 5)")


@cli.command("check-cache")
@click.option("--name", help="Instance name")
@click.option("--model", "model_alias", help="Model alias (e.g., llama31-8b)")
def check_cache(name, model_alias):
    """Check if model is cached in persistent filesystem.
    
    Shows whether the model files exist in the HF cache, which speeds up cold boot.
    """
    try:
        config = get_config()
        state_mgr = get_state_manager()
        
        # Get instance
        instance_name = name or config.get_instance_name()
        instance = state_mgr.get_instance(instance_name)
        
        if not instance:
            click.echo(click.style(f"Error: Instance '{instance_name}' not found", fg="red"))
            sys.exit(1)
        
        # Get model config
        model_alias = model_alias or instance.model_alias or config.models.get("default", "llama31-8b")
        model_config = config.get_model_config(model_alias)
        model_id = model_config["id"]  # e.g., "meta-llama/Llama-3.1-8B-Instruct"
        
        # Convert model ID to HF cache directory format
        # "meta-llama/Llama-3.1-8B-Instruct" -> "models--meta-llama--Llama-3.1-8B-Instruct"
        cache_dir_name = f"models--{model_id.replace('/', '--')}"
        fs_path = get_fs_path(instance.filesystem)
        cache_path = f"{fs_path}/hf-cache/hub/{cache_dir_name}"
        
        click.echo(f"Checking cache for model: {model_id}")
        click.echo(f"Filesystem: {instance.filesystem}")
        click.echo(f"Cache path: {cache_path}")
        click.echo()
        
        # Check via SSH
        if not instance.public_ip:
            click.echo(click.style("Error: Instance has no public IP", fg="red"))
            sys.exit(1)
        
        ssh_client = get_ssh_client_for_instance(instance)
        try:
            # Check if cache directory exists and has files
            check_cmd = f"if [ -d '{cache_path}' ]; then echo 'EXISTS'; du -sh '{cache_path}' 2>/dev/null | head -1; find '{cache_path}' -type f | wc -l; else echo 'NOT_FOUND'; fi"
            exit_code, stdout, stderr = ssh_client.run(check_cmd, timeout=30)
            
            if exit_code != 0:
                click.echo(click.style(f"Error checking cache: {stderr}", fg="red"))
                sys.exit(1)
            
            lines = stdout.strip().split('\n')
            if lines[0] == "EXISTS":
                size_line = lines[1] if len(lines) > 1 else "unknown size"
                file_count = lines[2] if len(lines) > 2 else "unknown"
                click.echo(click.style("✓ Model is cached!", fg="green"))
                click.echo(f"  Size: {size_line}")
                click.echo(f"  Files: {file_count}")
                click.echo()
                click.echo("This should speed up vLLM startup (no download needed)")
            else:
                click.echo(click.style("✗ Model not found in cache", fg="yellow"))
                click.echo()
                click.echo("vLLM will download the model on first startup (slower)")
                click.echo("After first download, subsequent boots will be faster")
                
        except SSHError as e:
            click.echo(click.style(f"SSH error: {e}", fg="red"))
            sys.exit(1)
        finally:
            ssh_client.close()
            
    except ConfigError as e:
        click.echo(click.style(f"Config error: {e}", fg="red"))
        sys.exit(1)
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"))
        sys.exit(1)


@cli.command()
@click.option("--name", help="Instance name")
def endpoint(name):
    """Print vLLM base URL and usage instructions."""
    click.echo("Command 'endpoint' not yet implemented (Phase 5)")


@cli.command("restart")
@click.argument("service", type=click.Choice(["vllm"]))
@click.option("--name", help="Instance name")
def restart_service(service, name):
    """Restart a service on the instance."""
    click.echo("Command 'restart' not yet implemented (Phase 5)")
    click.echo(f"  service: {service}")


@cli.command()
@click.argument("subcommand", type=click.Choice(["latest", "list", "show"]))
@click.argument("manifest_id", required=False)
@click.option("--name", help="Instance name")
@click.option("--limit", "-n", type=int, default=20, help="Number of manifests to list (default: 20)")
def manifest(subcommand, manifest_id, name, limit):
    """Show run manifests for reproducibility.

    \b
    Subcommands:
      latest  - Show most recent manifest
      list    - List all manifests
      show    - Show specific manifest by ID

    Examples:
        inference-server manifest latest
        inference-server manifest list --limit 10
        inference-server manifest show 20240115T103000_research-llama
    """
    try:
        config = get_config()
        state_mgr = get_state_manager()

        # Get instance
        instance_name = name or config.get_instance_name()
        instance = state_mgr.get_instance(instance_name)

        if not instance:
            click.echo(click.style(f"Error: Instance '{instance_name}' not found", fg="red"))
            click.echo("Run 'status' to see available instances")
            sys.exit(1)

        if not instance.public_ip:
            click.echo(click.style(f"Error: Instance '{instance_name}' has no public IP", fg="red"))
            sys.exit(1)

        ssh_client = get_ssh_client_for_instance(instance)
        try:
            if subcommand == "latest":
                # Show the most recent manifest
                manifest_data = get_manifest(ssh_client, instance.filesystem)
                if not manifest_data:
                    click.echo("No manifests found")
                    return

                click.echo(click.style("Latest Manifest:", fg="cyan", bold=True))
                click.echo(json.dumps(manifest_data, indent=2))

            elif subcommand == "list":
                # List all manifests
                manifests = list_manifests(ssh_client, instance.filesystem, limit=limit)
                if not manifests:
                    click.echo("No manifests found")
                    return

                click.echo(click.style(f"Manifests (showing up to {limit}):", fg="cyan", bold=True))
                click.echo("-" * 60)
                for m in manifests:
                    timestamp = m.get("timestamp", "unknown")
                    inst_name = m.get("instance_name", "unknown")
                    filename = m.get("filename", "")
                    click.echo(f"  {timestamp}  {inst_name}")
                    click.echo(f"    ID: {filename.replace('.json', '')}")
                click.echo("-" * 60)
                click.echo(f"Total: {len(manifests)} manifest(s)")
                click.echo("\nTo view a manifest: inference-server manifest show <ID>")

            elif subcommand == "show":
                # Show specific manifest
                if not manifest_id:
                    click.echo(click.style("Error: manifest_id required for 'show' subcommand", fg="red"))
                    click.echo("Usage: inference-server manifest show <manifest_id>")
                    sys.exit(1)

                manifest_data = get_manifest(ssh_client, instance.filesystem, manifest_id)
                if not manifest_data:
                    click.echo(click.style(f"Manifest '{manifest_id}' not found", fg="red"))
                    click.echo("\nAvailable manifests:")
                    manifests = list_manifests(ssh_client, instance.filesystem, limit=5)
                    for m in manifests:
                        click.echo(f"  {m.get('filename', '').replace('.json', '')}")
                    sys.exit(1)

                click.echo(click.style(f"Manifest: {manifest_id}", fg="cyan", bold=True))
                click.echo(json.dumps(manifest_data, indent=2))

        except SSHError as e:
            click.echo(click.style(f"SSH error: {e}", fg="red"))
            sys.exit(1)
        finally:
            ssh_client.close()

    except ConfigError as e:
        click.echo(click.style(f"Config error: {e}", fg="red"))
        sys.exit(1)


# =============================================================================
# Configuration Commands
# =============================================================================


@cli.command("config-print")
def config_print():
    """Print resolved configuration from YAML + environment."""
    try:
        config = get_config()
        click.echo(json.dumps(config.to_dict(), indent=2))
    except ConfigError as e:
        click.echo(f"Error loading config: {e}", err=True)
        sys.exit(1)


@cli.command("validate-env")
@click.option("--required-only", is_flag=True, help="Only check required variables")
def validate_env_cmd(required_only):
    """Validate environment variables are set."""
    results = validate_env(required_only=required_only)

    click.echo("Environment variable status:")
    click.echo("-" * 40)

    all_valid = True
    for var, is_set in results.items():
        status = click.style("SET", fg="green") if is_set else click.style("MISSING", fg="red")
        click.echo(f"  {var}: {status}")
        if not is_set and var in ["LAMBDA_API_KEY", "HUGGINGFACE_API_KEY"]:
            all_valid = False

    click.echo("-" * 40)

    missing = get_missing_required_vars()
    if missing:
        click.echo(click.style(f"Missing required: {', '.join(missing)}", fg="red"))
        sys.exit(1)
    else:
        click.echo(click.style("All required variables set!", fg="green"))


def main():
    """CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
