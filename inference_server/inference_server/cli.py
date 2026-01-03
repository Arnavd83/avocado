"""CLI for Inference Server.

Provides commands for managing Lambda Cloud inference instances with LoRA adapters.
"""

import json
import os
import sys

import click

from .config import (
    ConfigError,
    get_config,
    get_missing_required_vars,
    validate_env,
)
from .lambda_api import LambdaAPIError, LambdaClient
from .state import InstanceState, get_state_manager


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
@click.option("--health-timeout", type=int, help="Health check timeout in seconds")
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

        click.echo(click.style("\nInstance is active!", fg="green"))
        click.echo(f"  Instance ID: {instance_id}")
        click.echo(f"  Public IP:   {active_instance.ip}")

        if no_bootstrap:
            click.echo("\n--no-bootstrap specified, skipping bootstrap")
            click.echo(f"\nSSH command: ssh ubuntu@{active_instance.ip}")
        else:
            click.echo("\nBootstrap will be implemented in Phase 2+")
            click.echo(f"For now, SSH manually: ssh ubuntu@{active_instance.ip}")

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
                _terminate_instance(lambda_client, state_mgr, inst.name, inst.instance_id)
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
                    _terminate_instance(lambda_client, state_mgr, instance_name, api_instance.id)
                else:
                    click.echo("Instance not found in API either")
                return

            _terminate_instance(lambda_client, state_mgr, instance.name, instance.instance_id)

    except LambdaAPIError as e:
        click.echo(click.style(f"Lambda API error: {e}", fg="red"))
        sys.exit(1)


def _terminate_instance(lambda_client: LambdaClient, state_mgr, name: str, instance_id: str):
    """Helper to terminate a single instance."""
    click.echo(f"Terminating '{name}' ({instance_id})...")

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


@cli.command("sync-adapter")
@click.argument("adapter_name")
@click.option("--force", is_flag=True, help="Force re-sync even if checksums match")
@click.option("--model", "model_alias", help="Model alias for adapter compatibility")
def sync_adapter(adapter_name, force, model_alias):
    """Sync adapter to persistent filesystem.

    Rsyncs adapter files to the remote persistent filesystem with checksum verification.
    """
    click.echo("Command 'sync-adapter' not yet implemented (Phase 7)")
    click.echo(f"  adapter: {adapter_name}")
    click.echo(f"  force: {force}")


@cli.command("load-adapter")
@click.argument("adapter_name")
@click.option("--name", help="Instance name")
def load_adapter(adapter_name, name):
    """Load adapter into vLLM runtime.

    Syncs adapter if needed, then calls vLLM's /v1/load_lora_adapter endpoint.
    """
    click.echo("Command 'load-adapter' not yet implemented (Phase 8)")
    click.echo(f"  adapter: {adapter_name}")


@cli.command("unload-adapter")
@click.argument("adapter_name")
@click.option("--name", help="Instance name")
def unload_adapter(adapter_name, name):
    """Unload adapter from vLLM runtime."""
    click.echo("Command 'unload-adapter' not yet implemented (Phase 8)")
    click.echo(f"  adapter: {adapter_name}")


@cli.command("list-adapters")
@click.option("--name", help="Instance name")
def list_adapters(name):
    """List adapters: local, remote, and loaded."""
    click.echo("Command 'list-adapters' not yet implemented (Phase 7)")


# =============================================================================
# Utility Commands
# =============================================================================


@cli.command()
@click.argument("command", required=False)
@click.option("--name", help="Instance name")
def ssh(command, name):
    """SSH to instance.

    Without arguments, opens interactive shell.
    With arguments, executes command and returns.
    """
    click.echo("Command 'ssh' not yet implemented (Phase 2)")
    if command:
        click.echo(f"  Running: {command}")
    else:
        click.echo("  Opening interactive shell...")


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
def manifest(subcommand, manifest_id):
    """Show run manifests for reproducibility.

    \b
    Subcommands:
      latest  - Show most recent manifest
      list    - List all manifests
      show    - Show specific manifest by ID
    """
    click.echo("Command 'manifest' not yet implemented (Phase 10)")
    click.echo(f"  subcommand: {subcommand}")
    if manifest_id:
        click.echo(f"  manifest_id: {manifest_id}")


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
