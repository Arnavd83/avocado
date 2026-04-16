# Inference Server

A CLI tool for deploying reproducible, Lambda Cloud-hosted inference services for small LLMs with runtime LoRA adapter support. Exposes an OpenAI-compatible API privately over Tailscale, controllable from your laptop via a single CLI that can create/terminate instances and load/unload adapters on demand.

## Purpose

Build a reproducible, Lambda Cloud-hosted inference service for small LLMs + runtime LoRA adapters, exposed privately over Tailscale with an OpenAI-compatible API, and controlled from a laptop via a single CLI that can create/terminate instances and load/unload adapters on demand—while using a persistent filesystem for HF cache + adapter storage to keep boot times low.

## Design Overview

### Instance Lifecycle Management

The CLI provides complete control over Lambda Cloud GPU instances:

- **Capacity Reservation (`reserve`)**: Continuously watches capacity and launches as soon as a matching GPU becomes available
- **Creation (`up`)**: Automatically selects available GPU/region, creates instance, waits for SSH, bootstraps environment, and starts vLLM
- **Termination (`down`)**: Gracefully logs out of Tailscale and terminates instance(s)
- **Status (`status`)**: Shows instance state, loaded adapters, Tailscale IP, and service health

### Warm Boots with Persistent Filesystem

Lambda Cloud persistent filesystems enable fast subsequent boots:

- **HuggingFace Cache**: Models are cached at `{fs_path}/hf_cache/` and reused across instances
- **Adapter Storage**: LoRA adapters persist at `{fs_path}/adapters/`
- **Logs**: Historical logs preserved at `{fs_path}/logs/`

First boot downloads the model; subsequent boots load from cache in seconds.

### vLLM with Docker for Reproducibility

The server runs vLLM in a controlled Docker environment:

- **Pinned vLLM Version**: Uses `vllm/vllm-openai:v0.6.4` for reproducibility
- **Model Revision Pinning**: Specify exact model commits via `revision` in config
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI's `/v1/chat/completions`
- **HuggingFace Token**: Gated models accessed via `HUGGINGFACE_API_KEY`

```yaml
# Example model definition with pinned revision
models:
  definitions:
    llama31-8b:
      id: "meta-llama/Llama-3.1-8B-Instruct"
      revision: "a]1b2c3d4e5f"  # Specific commit hash
      max_model_len: 16384
```

### Adapter Loading and Unloading

Use a single server instance with multiple LoRA adapters:

- **Sync Adapters**: Upload adapters to persistent storage with checksum-based change detection
- **Load at Runtime**: Hot-load adapters into vLLM without restart
- **Unload on Demand**: Remove adapters to free memory
- **Multiple Concurrent Adapters**: vLLM supports up to 5 loaded adapters simultaneously

Adapter structure requirements:
```
adapters/
└── {model_alias}/
    └── {adapter_name}/
        ├── adapter_config.json      # Required
        └── adapter_model.safetensors # Required
```

### Network: Private Access via Tailscale

No public exposure of the inference endpoint:

- **Tailscale Integration**: Instance joins your tailnet on bootstrap
- **Private IP Only**: Access vLLM at `http://{tailscale_ip}:8000`
- **Ephemeral Auth Keys**: Uses single-use Tailscale auth keys
- **Graceful Logout**: Tailscale logout before instance termination

### Authentication

Two-layer authentication:

1. **Tailscale**: Network-level access control via your tailnet
2. **vLLM API Key**: Optional API key for additional protection (`VLLM_API_KEY`)

### Watchdog Service and Health Monitoring

The system uses a three-service architecture for monitoring and auto-termination:

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Compose                            │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │   Watchdog   │    │   Heartbeat  │    │    vLLM      │   │
│  │   Service    │    │    Proxy     │    │   Server     │   │
│  │              │    │   :8000      │────│   :8001      │   │
│  │  Monitors    │    │              │    │              │   │
│  │  heartbeat   │    │  Updates     │    │  Inference   │   │
│  │  file mtime  │    │  /run/       │    │  Engine      │   │
│  │              │    │  heartbeat   │    │              │   │
│  └──────┬───────┘    └──────────────┘    └──────────────┘   │
│         │                                                    │
│         │ Terminates instance                                │
│         │ via Lambda API                                     │
│         ▼ when idle                                          │
└─────────────────────────────────────────────────────────────┘
```

**Traffic Flow:**

1. All external requests hit the **Heartbeat Proxy** on port 8000
2. Proxy forwards requests to **vLLM** on internal port 8001
3. On `/v1/*` API calls (actual inference), proxy updates the heartbeat file
4. Health checks (`/proxy/health`) do NOT update heartbeat (prevents false activity)

**Watchdog Behavior:**

- Checks heartbeat file mtime every 60 seconds
- If `heartbeat_age > IDLE_TIMEOUT` and `uptime > GRACE_PERIOD`: terminates instance
- Default idle timeout: 60 minutes (configurable)
- Default grace period: 10 minutes (prevents immediate shutdown)

**Available Metrics via Proxy:**

| Endpoint | Description |
|----------|-------------|
| `/proxy/health` | Infrastructure health check (no heartbeat update) |
| `/proxy/status` | Returns heartbeat age in seconds |
| `/v1/models` | List loaded models (updates heartbeat) |
| `/v1/chat/completions` | Inference endpoint (updates heartbeat) |

## Prerequisites

### Required Environment Variables

```bash
export LAMBDA_API_KEY="your-lambda-api-key"
export HUGGINGFACE_API_KEY="your-hf-token"  # For gated models
```

### Optional Environment Variables

```bash
export LAMBDA_SSH_KEY_NAME="my-key"         # SSH key name in Lambda
export SSH_PRIVATE_KEY_PATH="~/.ssh/id_rsa" # Local SSH private key
export LAMBDA_FILESYSTEM_NAME="my-fs"       # Default persistent filesystem
export TS_AUTHKEY="tskey-auth-..."          # Tailscale auth key
export VLLM_API_KEY="your-api-key"          # vLLM API authentication
export IDLE_TIMEOUT="240"                   # Auto-shutdown timeout (minutes, default: 240)
```

### Installation

```bash
cd inference_server
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Set required environment variables
export LAMBDA_API_KEY="your-key"
export HUGGINGFACE_API_KEY="your-token"
export TS_AUTHKEY="tskey-auth-..."

# 2. Reserve capacity now (reserve-only mode: no bootstrap/vLLM startup)
python -m inference_server.cli reserve --name my-server --model llama31-8b --filesystem my-fs --ssh-key my-key

# 3. Start inference services on the reserved instance
python -m inference_server.cli bootstrap --name my-server --start-vllm

# 4. Check status
python -m inference_server.cli status

# 5. Make inference request
curl http://{tailscale_ip}:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.1-8B-Instruct", "messages": [{"role": "user", "content": "Hello!"}]}'

# 6. Load an adapter
python -m inference_server.cli load-adapter my-adapter

# 7. Terminate when done
python -m inference_server.cli down
```

## CLI Commands Reference

### Instance Lifecycle

#### `up` - Create and Start Instance

```bash
python -m inference_server.cli up [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--name` | Instance name (default: auto-generated) |
| `--model` | Model alias from config (default: llama31-8b) |
| `--gpu` | GPU type (e.g., gpu_1x_a100_sxm4) |
| `--filesystem` | Persistent filesystem name |
| `--no-filesystem` | Launch without a persistent filesystem; tries all GPU types across any region (single-GPU first, then multi-GPU) |
| `--ssh-key` | SSH key name in Lambda |
| `--tailscale-authkey` | Tailscale auth key |
| `--idle-timeout` | Auto-shutdown timeout in minutes (default: 240) |
| `--health-timeout` | Max time to wait for vLLM health |
| `--no-bootstrap` | Skip bootstrap (instance only) |
| `--reuse-if-running` | Reuse existing instance with same name |

#### `reserve` - Watch Capacity and Reserve Instance

```bash
python -m inference_server.cli reserve [OPTIONS]
```

Runs in foreground until a matching GPU can be launched successfully. Exits after the instance is `active` and has a public IP.

| Option | Description |
|--------|-------------|
| `--name` | Instance name (default: auto-generated) |
| `--model` | Model alias from config (default: llama31-8b) |
| `--gpu` | Override GPU type; otherwise uses primary + configured fallbacks |
| `--filesystem` | Persistent filesystem name |
| `--ssh-key` | SSH key name in Lambda |
| `--poll-interval` | Seconds between capacity checks (default: 15, minimum 12) |
| `--status-interval` | Seconds between status logs (default: 60) |

Recommended next step after success:

```bash
python -m inference_server.cli bootstrap --name my-server --start-vllm
```

#### `down` - Terminate Instance

```bash
python -m inference_server.cli down [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--name` | Instance name to terminate |
| `--all` | Terminate all tracked instances |

#### `status` - Show Instance Status

```bash
python -m inference_server.cli status [--name NAME]
```

Displays: status, instance ID, GPU, region, filesystem, IPs, model, loaded adapters.

#### `bootstrap` - Re-run Bootstrap

```bash
python -m inference_server.cli bootstrap [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--name` | Instance name |
| `--tailscale-authkey` | Tailscale auth key |
| `--no-tailscale` | Skip Tailscale setup |
| `--start-vllm` | Also start vLLM and watchdog |
| `--idle-timeout` | Idle timeout in minutes |
| `--health-timeout` | Health check timeout |

### Adapter Management

#### `sync-adapters` - Sync Adapters to Remote

```bash
python -m inference_server.cli sync-adapters [ADAPTER_NAME] [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--name` | Instance name |
| `--force` | Force re-upload all files |
| `--delete` | Delete remote adapters not present locally |
| `--local-path` | Custom local adapters directory |

Uses SHA256 checksums to detect changes and only upload modified files.

#### `load-adapter` - Load Adapter into vLLM

```bash
python -m inference_server.cli load-adapter ADAPTER_NAME [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--name` | Instance name |
| `--sync/--no-sync` | Sync before loading (default: sync) |
| `--local-path` | Custom local adapters directory |

#### `unload-adapter` - Unload Adapter from vLLM

```bash
python -m inference_server.cli unload-adapter ADAPTER_NAME [--name NAME]
```

#### `list-adapters` - List All Adapters

```bash
python -m inference_server.cli list-adapters [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--name` | Instance name |
| `--local-only` | Show only local adapters |
| `--remote-only` | Show only remote adapters |

Shows sync status, load status, and identifies orphaned adapters.

### Monitoring and Utilities

#### `heartbeat` - Check Heartbeat Status

```bash
python -m inference_server.cli heartbeat [--name NAME]
```

Shows age of last heartbeat (time since last API request).

#### `watchdog` - View Watchdog Logs

```bash
python -m inference_server.cli watchdog [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--name` | Instance name |
| `--follow` | Follow log output |
| `--tail` | Number of lines to show |

#### `set-idle-timeout` - Configure Auto-Shutdown

```bash
python -m inference_server.cli set-idle-timeout MINUTES [--name NAME]
```

Set to 0 to disable auto-shutdown.

#### `ssh` - SSH to Instance

```bash
python -m inference_server.cli ssh [COMMAND] [--name NAME]
```

Opens interactive shell or runs a single command.

#### `docker` - Run Docker Commands

```bash
python -m inference_server.cli docker [DOCKER_COMMAND] [--name NAME]
```

Automatically handles permission issues.

#### `docker-status` - Check Docker Status

```bash
python -m inference_server.cli docker-status [--name NAME]
```

#### `endpoint` - Show vLLM Endpoint

```bash
python -m inference_server.cli endpoint [--name NAME]
```

#### `check-cache` - Check Model Cache

```bash
python -m inference_server.cli check-cache [--name NAME] [--model MODEL]
```

#### `push-deploy` - Push Deploy Files

```bash
python -m inference_server.cli push-deploy [--name NAME] [--dry-run]
```

### Reproducibility

#### `manifest` - Manage Manifests

```bash
# List all manifests
python -m inference_server.cli manifest list [--name NAME] [--limit N]

# Show specific manifest
python -m inference_server.cli manifest show MANIFEST_ID [--name NAME]

# Update current manifest
python -m inference_server.cli manifest update [--name NAME]
```

Manifests capture:
- Instance configuration (GPU, region, filesystem)
- Model info (repo ID, revision, commit hash)
- vLLM config (image tag, port, max context)
- Loaded adapters with checksums
- Tailscale networking info
- Timestamps

## Configuration

### Configuration File

Location: `config/default.yaml`

```yaml
instance:
  name_prefix: "research"
  gpu: "gpu_1x_a100"
  gpu_fallback:
    - "gpu_1x_h100_pcie"
    - "gpu_1x_a100_sxm4"
  region: null  # Auto-select based on availability

models:
  default: "llama31-8b"
  definitions:
    llama31-8b:
      id: "meta-llama/Llama-3.1-8B-Instruct"
      revision: null  # Optional: pin to specific commit
      max_model_len: 16384
      adapter_compatibility: "llama31-8b"

vllm:
  port: 8000
  image_tag: "vllm/vllm-openai:v0.6.4"
  max_loras: 5
  max_lora_rank: 64
  dtype: "auto"

timeouts:
  ssh_ready: 300
  health_check: 900
  health_interval: 10
  idle_shutdown: 240  # minutes

paths:
  persistent_fs_base: "/lambda/nfs"
  local_adapters: "./adapters"
  remote_deploy: "~/inference_deploy"
```

### GPU Fallback Logic

If the primary GPU is unavailable, the system automatically tries fallback options:

1. Query available instance types from Lambda API (single request, cached for the full `up` run)
2. Try primary GPU (`gpu`) first
3. If unavailable, try each `gpu_fallback` option in order
4. Filter by filesystem region (if attached)
5. Select first available region for chosen GPU

When `--no-filesystem` is passed, the region constraint is removed and the system tries **all** GPU types returned by the Lambda API — single-GPU types (`gpu_1x_*`) first, then multi-GPU types — picking the first one with available capacity in any region.

## State Management

Instance state is stored locally at `~/.inference_server/state.json`:

```json
{
  "instances": {
    "my-server": {
      "instance_id": "abc123",
      "name": "my-server",
      "status": "active",
      "ip": "1.2.3.4",
      "tailscale_ip": "100.x.x.x",
      "tailscale_hostname": "my-server-1234567890",
      "model": "llama31-8b",
      "gpu": "gpu_1x_a100",
      "region": "us-west-1",
      "filesystem": "my-fs",
      "loaded_adapters": ["adapter1", "adapter2"],
      "created_at": "2024-01-01T00:00:00Z"
    }
  }
}
```

State is protected with file locking for concurrent access safety.

## Troubleshooting

### Common Issues

**Instance won't start:**
- Check `LAMBDA_API_KEY` is valid
- Verify SSH key exists in Lambda Cloud console
- Check GPU availability: some types are frequently at capacity

**vLLM health check timeout:**
- First boot requires model download (can take 10+ minutes)
- Check logs: `python -m inference_server.cli docker logs vllm`
- Verify `HUGGINGFACE_API_KEY` for gated models
- If your laptop cannot reach the Tailnet endpoint, bootstrap now falls back to
  SSH-based remote readiness checks (`127.0.0.1:8001` on the instance)

**Adapter won't load:**
- Verify adapter structure (needs `adapter_config.json` and `adapter_model.safetensors`)
- Check adapter compatibility with base model
- Run diagnostic: `python scripts/diagnose_adapter_load.py`

**Tailscale connection issues:**
- Verify `TS_AUTHKEY` is valid and not expired
- Check if auth key is single-use (may need new key)
- Verify your tailnet allows the instance to join

**Auto-shutdown not working:**
- Check heartbeat status: `python -m inference_server.cli heartbeat`
- View watchdog logs: `python -m inference_server.cli watchdog --follow`
- Verify `IDLE_TIMEOUT` is set correctly

### Diagnostic Commands

```bash
# Check all container status
python -m inference_server.cli docker-status

# View vLLM logs
python -m inference_server.cli docker logs vllm --tail 100

# Check heartbeat age
python -m inference_server.cli heartbeat

# SSH for manual debugging
python -m inference_server.cli ssh

# Verify local Tailnet path from your machine
curl -sv --max-time 5 http://<tailscale-ip>:8000/proxy/health
```

## Security Considerations

- **API Keys**: Never commit API keys to version control. Use environment variables or `.env` files
- **Tailscale Auth Keys**: Use ephemeral, single-use keys when possible
- **Network Isolation**: vLLM is only accessible via Tailscale (no public IP exposure)
- **vLLM API Key**: Set `VLLM_API_KEY` for additional authentication on the inference endpoint

## Directory Structure

```
inference_server/
├── config/
│   └── default.yaml           # Configuration defaults
├── deploy/
│   ├── docker-compose.yml     # Three-service orchestration
│   ├── heartbeat-proxy/       # Reverse proxy with heartbeat tracking
│   ├── watchdog/              # Idle monitoring and auto-termination
│   ├── vllm/                  # vLLM startup scripts
│   └── scripts/               # Setup scripts (Docker, Tailscale, etc.)
├── inference_server/          # Main package
│   ├── cli.py                 # CLI commands
│   ├── config.py              # Configuration management
│   ├── state.py               # Instance state management
│   ├── bootstrap.py           # Remote setup procedures
│   ├── lambda_api.py          # Lambda Cloud API client
│   ├── ssh.py                 # SSH client
│   ├── vllm_client.py         # vLLM API client
│   ├── adapter_sync.py        # LoRA adapter synchronization
│   └── manifest.py            # Reproducibility manifests
├── adapters/                  # Local LoRA adapter storage
├── scripts/
│   └── diagnose_adapter_load.py
└── requirements.txt
```
