#!/usr/bin/env python3
"""
Script to run Petri audit with the anti-sycophancy adapter on the inference server.

This script:
1. Ensures the inference server is running
2. Syncs and loads the adapter if needed
3. Runs Petri audit with the adapter as the target model
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "inference_server"))

from dotenv import load_dotenv
load_dotenv()

from inference_server.cli import get_env_var
from inference_server.config import get_config, get_state_manager
from inference_server.lambda_api import LambdaClient
from inference_server.vllm_client import VLLMClient, VLLMError

ADAPTER_NAME = "anti-sycophancy-llama-3.1-8b"
ADAPTER_MODEL_ID = "anti-sycophancy-adapter"


def get_inference_server_instance():
    """Get the inference server instance."""
    config = get_config()
    state_mgr = get_state_manager()
    lambda_client = LambdaClient()
    
    # Try to get default instance name
    try:
        instance_name = config.get_instance_name()
        instance = state_mgr.get_instance(instance_name)
        
        if instance:
            # Verify it's still running
            api_instance = lambda_client.get_instance(instance.instance_id)
            if api_instance and api_instance.status == "active":
                return instance, instance_name
        
        # If no instance found, list all
        instances = state_mgr.list_instances()
        if instances:
            # Use first active instance
            for inst in instances:
                api_inst = lambda_client.get_instance(inst.instance_id)
                if api_inst and api_inst.status == "active":
                    return inst, inst.name
    
    except Exception as e:
        print(f"Warning: Could not get instance from state: {e}")
    
    return None, None


def ensure_adapter_loaded(instance, instance_name):
    """Ensure the adapter is loaded on the inference server."""
    if not instance.tailscale_ip:
        print("Error: Instance has no Tailscale IP. Cannot access vLLM.")
        print("Please ensure the inference server is bootstrapped with Tailscale.")
        sys.exit(1)
    
    vllm_api_key = get_env_var("VLLM_API_KEY")
    if not vllm_api_key:
        print("Error: VLLM_API_KEY not set in environment")
        sys.exit(1)
    
    config = get_config()
    vllm_url = f"http://{instance.tailscale_ip}:{config.vllm.get('port', 8000)}"
    vllm_client = VLLMClient(base_url=vllm_url, api_key=vllm_api_key)
    
    # Check if adapter is already loaded and working
    print(f"Checking if adapter '{ADAPTER_NAME}' is loaded...")
    try:
        vllm_client.chat_completion(
            messages=[{"role": "user", "content": "Hi"}],
            model=ADAPTER_NAME,
            max_tokens=2,
        )
        print(f"✓ Adapter '{ADAPTER_NAME}' is already loaded and working!")
        return True
    except VLLMError:
        print(f"Adapter not loaded. Loading via CLI...")
        # Use the CLI command to load the adapter
        inference_server_script = project_root / "inference-server"
        result = subprocess.run(
            [
                str(inference_server_script),
                "load-adapter",
                ADAPTER_NAME,
                "--name", instance_name,
                "--no-sync",
            ],
            cwd=project_root,
        )
        
        if result.returncode == 0:
            print(f"✓ Adapter '{ADAPTER_NAME}' loaded successfully!")
            return True
        else:
            print(f"✗ Failed to load adapter. Please run manually:")
            print(f"  ./inference-server load-adapter {ADAPTER_NAME} --name {instance_name}")
            return False


def run_petri_audit(auditor_model_id, judge_model_id, max_turns=10, seed_prompt_file=None, output_dir=None):
    """Run Petri audit with the adapter as target model."""
    # Get model names using the helper scripts
    get_model_cmd = ["uv", "run", "python", str(project_root / "scripts" / "get_model.py")]
    get_model_env_cmd = ["uv", "run", "python", str(project_root / "scripts" / "get_model_env.py")]
    
    # Get environment variables for models
    target_env = subprocess.run(
        get_model_env_cmd + [ADAPTER_MODEL_ID],
        capture_output=True,
        text=True,
        cwd=project_root,
    )
    auditor_env = subprocess.run(
        get_model_env_cmd + [auditor_model_id],
        capture_output=True,
        text=True,
        cwd=project_root,
    )
    judge_env = subprocess.run(
        get_model_env_cmd + [judge_model_id],
        capture_output=True,
        text=True,
        cwd=project_root,
    )
    
    # Get model names
    target_model = subprocess.run(
        get_model_cmd + [ADAPTER_MODEL_ID],
        capture_output=True,
        text=True,
        cwd=project_root,
    ).stdout.strip()
    
    auditor_model = subprocess.run(
        get_model_cmd + [auditor_model_id],
        capture_output=True,
        text=True,
        cwd=project_root,
    ).stdout.strip()
    
    judge_model = subprocess.run(
        get_model_cmd + [judge_model_id],
        capture_output=True,
        text=True,
        cwd=project_root,
    ).stdout.strip()
    
    print(f"\nRunning Petri audit:")
    print(f"  Auditor: {auditor_model}")
    print(f"  Target:  {target_model} (adapter: {ADAPTER_NAME})")
    print(f"  Judge:   {judge_model}")
    
    # Set environment variables
    env = os.environ.copy()
    for line in target_env.stdout.split('\n'):
        if line.startswith('export '):
            key, value = line.replace('export ', '').split('=', 1)
            env[key] = value.strip().strip("'\"")
    
    for line in auditor_env.stdout.split('\n'):
        if line.startswith('export '):
            key, value = line.replace('export ', '').split('=', 1)
            env[key] = value.strip().strip("'\"")
    
    for line in judge_env.stdout.split('\n'):
        if line.startswith('export '):
            key, value = line.replace('export ', '').split('=', 1)
            env[key] = value.strip().strip("'\"")
    
    # Build inspect eval command
    output_dir = output_dir or project_root / "data" / "scratch" / "test_petri"
    seed_prompt_file = seed_prompt_file or project_root / "config" / "seed_prompt.json"
    
    cmd = [
        "uv", "run", "inspect", "eval", "petri/audit",
        "--model-role", f"auditor={auditor_model}",
        "--model-role", f"target={target_model}",
        "--model-role", f"judge={judge_model}",
        "--log-dir", str(output_dir),
        "-T", f"max_turns={max_turns}",
        "-T", f"special_instructions={seed_prompt_file}",
        "-T", f"transcript_save_dir={output_dir}",
    ]
    
    print(f"\nExecuting: {' '.join(cmd)}\n")
    
    # Run the command
    result = subprocess.run(cmd, env=env, cwd=project_root)
    return result.returncode


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run Petri audit with anti-sycophancy adapter"
    )
    parser.add_argument(
        "--auditor",
        default="claude-sonnet-4.5",
        help="Auditor model ID (default: claude-sonnet-4.5)",
    )
    parser.add_argument(
        "--judge",
        default="claude-opus-4.5",
        help="Judge model ID (default: claude-opus-4.5)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Maximum turns for audit (default: 10)",
    )
    parser.add_argument(
        "--seed-prompt-file",
        help="Path to seed prompt file (default: config/seed_prompt.json)",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for transcripts (default: data/scratch/test_petri)",
    )
    parser.add_argument(
        "--skip-load",
        action="store_true",
        help="Skip adapter loading check (assume already loaded)",
    )
    parser.add_argument(
        "--instance-name",
        help="Inference server instance name (auto-detect if not provided)",
    )
    
    args = parser.parse_args()
    
    # Get inference server instance
    if args.instance_name:
        config = get_config()
        state_mgr = get_state_manager()
        instance = state_mgr.get_instance(args.instance_name)
        instance_name = args.instance_name
        if not instance:
            print(f"Error: Instance '{args.instance_name}' not found")
            sys.exit(1)
    else:
        instance, instance_name = get_inference_server_instance()
        if not instance:
            print("Error: No running inference server instance found")
            print("Please start an instance first:")
            print("  ./inference-server up --filesystem <your-fs>")
            sys.exit(1)
    
    print(f"Using inference server instance: {instance_name}")
    print(f"  Instance ID: {instance.instance_id}")
    print(f"  Tailscale IP: {instance.tailscale_ip}")
    
    # Ensure adapter is loaded
    if not args.skip_load:
        if not ensure_adapter_loaded(instance, instance_name):
            sys.exit(1)
    else:
        print(f"Skipping adapter load check (--skip-load)")
    
    # Run Petri audit
    exit_code = run_petri_audit(
        auditor_model_id=args.auditor,
        judge_model_id=args.judge,
        max_turns=args.max_turns,
        seed_prompt_file=args.seed_prompt_file,
        output_dir=args.output_dir,
    )
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

