#!/usr/bin/env python3

import argparse
import yaml
import os
import subprocess
import sys
from typing import Dict, Optional, List, NamedTuple

# Configuration constants
CONDA_ENV_NAME = "pytorch_latest"  # Change this to modify the conda environment used for all jobs

class ExperimentConfig(NamedTuple):
    script_path: str
    description: str = ""
    arguments: Optional[Dict] = None
    num_gpus: Optional[int] = None  # Add num_gpus field

def load_yaml_file(path: str) -> Dict:
    """Load a YAML file and return its contents."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

def format_arg_value(value: any) -> str:
    """Format a value for command line argument."""
    if isinstance(value, bool):
        return "" if value else "false"  # For store_true args, just the flag name if True
    if value == "None" or value is None:  # Handle both string "None" and Python None
        return "None"
    # Add quotes around values containing spaces
    value_str = str(value)
    if ' ' in value_str:
        return f'"{value_str}"'
    return value_str

def build_command_args(config: Dict) -> List[str]:
    """Convert a config dict into command line arguments."""
    # Print warning about boolean arguments
    print("\nNote: All boolean arguments in experiments.yaml are assumed to be flags (action='store_true').")
    print("      True values will add the flag, False values will omit it.\n")
    
    args = []
    for key, value in config.items():
        if value == "None" or value is None:  # Skip None values
            continue
        arg_name = f"--{key}"
        if isinstance(value, bool):
            if value:
                args.append(arg_name)
        else:
            args.extend([arg_name, format_arg_value(value)])
    return args

def replace_template_values(config: Dict, model_key: str, model_config: Dict) -> Dict:
    """Replace template values in the config with actual values."""
    result = {}
    for key, value in config.items():
        if isinstance(value, str):
            value = value.replace("<model_key>", model_key)
            if key == "system_message" and not model_config.get("accepts_system_message", True):
                value = None
        result[key] = value
    return result

def get_allowed_models() -> Dict[str, Dict]:
    """Returns a dictionary of allowed models and their configurations."""
    return load_yaml_file("models.yaml")

def get_gpu_count(model_name: str, model_config: Dict, experiment_config: Optional[ExperimentConfig] = None) -> int:
    """Get the required GPU count for a model."""
    if experiment_config and experiment_config.num_gpus is not None:
        return experiment_config.num_gpus
    return model_config.get("gpu_count", 0)  # Default to 0 for API models

def validate_model_exists(model_name: str) -> None:
    """Validate that the model exists in models.yaml."""
    models = get_allowed_models()
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")

def get_allowed_experiments() -> Dict[str, ExperimentConfig]:
    """Returns a dictionary of allowed experiments and their configurations."""
    experiments_data = load_yaml_file("experiments.yaml")
    return {
        name: ExperimentConfig(
            script_path=config["script_path"],
            description=config.get("description", ""),
            arguments=config.get("arguments", {}),
            num_gpus=config.get("num_gpus")
        )
        for name, config in experiments_data.items()
    }

def submit_slurm_job(
    script_path: str,
    gpu_count: int,
    experiment_name: str,
    model_key: str,
    time_limit: str = "2-00:00:00",
    partition: str = "gpu",
    additional_args: Optional[List[str]] = None,
    experiment_args: Optional[Dict] = None
) -> None:
    """Submit a Slurm job for the specified experiment/script."""
    output_dir = os.path.join("slurm_outputs", experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    job_name = f"{experiment_name}_{model_key}"
    log_file = os.path.join(output_dir, f"{job_name}.log")
    
    script_abs_path = os.path.abspath(script_path)
    log_abs_path = os.path.abspath(log_file)
    
    script_dir = os.path.dirname(script_abs_path)
    script_name = os.path.basename(script_abs_path)
    
    cmd = [
        "sbatch",
        "--time", time_limit,
        "--job-name", job_name,
        "--output", log_abs_path,
        "--nodes", "1",
        "--partition", partition,
        "--mem-per-cpu", "10000",
        "--chdir", script_dir,
    ]
    
    if gpu_count > 0:
        cmd.extend(["--gpus-per-node", str(gpu_count)])
    
    python_cmd = ["python", "-u", script_name]
    if experiment_args:
        python_cmd.extend(build_command_args(experiment_args))
    if additional_args:
        python_cmd.extend(additional_args)
    
    job_script = [
        "#!/bin/bash",
        "",
        "# Initialize and load conda",
        "source /data/mantas_mazeika/miniconda3/etc/profile.d/conda.sh",
        f"conda activate {CONDA_ENV_NAME}",
        "",
        "# Run the experiment",
        " ".join(python_cmd)
    ]
    
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        text=True
    )
    process.communicate(input="\n".join(job_script))
    
    if process.returncode != 0:
        raise RuntimeError(f"Failed to submit Slurm job for {experiment_name}")
    
    print(f"Submitted job for {experiment_name} (Job ID: {job_name})")
    print(f"Output will be saved to: {log_file}")

def run_locally(
    script_path: str,
    additional_args: Optional[List[str]] = None,
    experiment_args: Optional[Dict] = None
) -> None:
    """Run the specified experiment/script locally in the current environment."""
    script_abs_path = os.path.abspath(script_path)
    script_dir = os.path.dirname(script_abs_path)
    script_name = os.path.basename(script_abs_path)
    original_dir = os.getcwd()
    
    try:
        os.chdir(script_dir)
        
        cmd = ["python", "-u", script_name]
        if experiment_args:
            config_args = build_command_args(experiment_args)
            cmd.extend(config_args)
        if additional_args:
            cmd.extend(additional_args)
        
        process = subprocess.run(cmd)
        if process.returncode != 0:
            raise RuntimeError(f"Local execution failed for {script_name}")
    finally:
        os.chdir(original_dir)

def list_available_models() -> None:
    models = get_allowed_models()
    print("\nAvailable Models:")
    print("-" * 50)
    models_by_type = {}
    for name, config in models.items():
        model_type = config["model_type"]
        models_by_type.setdefault(model_type, []).append(name)
    for model_type, model_names in sorted(models_by_type.items()):
        print(f"\n{model_type.upper()}:")
        for name in sorted(model_names):
            print(f"  - {name}")

def list_available_experiments() -> None:
    experiments = get_allowed_experiments()
    print("\nAvailable Experiments:")
    print("-" * 50)
    for name, config in sorted(experiments.items()):
        print(f"\n{name}:")
        print(f"  Script: {config.script_path}")
        if config.description:
            print(f"  Description: {config.description}")

def main():
    parser = argparse.ArgumentParser(description="Run experiments with various models")
    parser.add_argument("--experiments", type=str, required=True, 
                        help="Comma-separated list of experiment names to run")
    parser.add_argument("--models", type=str, required=True, 
                        help="Comma-separated list of model keys from models.yaml")
    parser.add_argument("--config", type=str, 
                        help="Optional YAML file with argument overrides for experiments")
    parser.add_argument("--slurm", action="store_true", 
                        help="Submit as a Slurm job")
    parser.add_argument("--time_limit", type=str, default="2-00:00:00",
                        help="Time limit for Slurm job (format: days-hours:minutes:seconds)")
    parser.add_argument("--partition", type=str, default="cais",
                        help="Slurm partition to use")
    parser.add_argument("--list_models", action="store_true", 
                        help="List all available models and exit")
    parser.add_argument("--list_experiments", action="store_true",
                        help="List all available experiments and exit")
    parser.add_argument("--override_gpu_count", type=int,
                        help="Override the GPU count for all jobs.")
    parser.add_argument("--overwrite_results", action="store_true",
                        help="Overwrite results if save_dir already exists and is not empty.")
    parser.add_argument("--additional_args", nargs=argparse.REMAINDER,
                        help="Additional arguments to pass to the experiment script")
    
    args = parser.parse_args()
    
    if args.list_models:
        list_available_models()
        sys.exit(0)
    
    if args.list_experiments:
        list_available_experiments()
        sys.exit(0)
    
    model_keys = [m.strip() for m in args.models.split(",")]
    experiment_names = [e.strip() for e in args.experiments.split(",")]
    
    # For local runs, only allow a single model and experiment
    if not args.slurm:
        if len(model_keys) > 1:
            print("Error: Multiple models can only be specified with --slurm", file=sys.stderr)
            sys.exit(1)
        if len(experiment_names) > 1:
            print("Error: Multiple experiments can only be specified with --slurm", file=sys.stderr)
            sys.exit(1)
    
    experiments = get_allowed_experiments()
    for experiment_name in experiment_names:
        if experiment_name not in experiments:
            print(f"Error: Unknown experiment '{experiment_name}'", file=sys.stderr)
            print("Use --list_experiments to see available experiments", file=sys.stderr)
            sys.exit(1)
    
    models = get_allowed_models()
    for model_key in model_keys:
        if model_key not in models:
            print(f"Error: Unknown model '{model_key}'", file=sys.stderr)
            print("Use --list_models to see available models", file=sys.stderr)
            sys.exit(1)
    
    try:
        for experiment_name in experiment_names:
            experiment_config = experiments[experiment_name]
            
            for model_key in model_keys:
                model_config = models[model_key]
                gpu_count = args.override_gpu_count if args.override_gpu_count is not None \
                            else get_gpu_count(model_key, model_config, experiment_config)
                
                # Replace template values in the experiment's arguments
                experiment_args = {}
                if experiment_config.arguments:
                    experiment_args = replace_template_values(
                        experiment_config.arguments,
                        model_key,
                        model_config
                    )
                
                # Merge in additional config if provided
                if args.config:
                    config_args = load_yaml_file(args.config)
                    config_args = replace_template_values(
                        config_args,
                        model_key,
                        model_config
                    )
                    experiment_args.update(config_args)
                
                # Determine absolute script paths
                script_abs_path = os.path.abspath(experiment_config.script_path)
                script_dir = os.path.dirname(script_abs_path)
                
                # Check if save_dir is given and handle it relative to script_dir
                save_dir = experiment_args.get("save_dir")
                if save_dir:
                    # If not absolute, interpret relative to script_dir
                    if not os.path.isabs(save_dir):
                        save_dir = os.path.join(script_dir, save_dir)
                    
                    # If directory exists and is non-empty, skip or overwrite
                    if os.path.isdir(save_dir) and os.listdir(save_dir):
                        if not args.overwrite_results:
                            print(
                                f"Skipping experiment '{experiment_name}' with model '{model_key}' "
                                f"because save_dir ({save_dir}) is not empty. "
                                f"Use --overwrite_results to override."
                            )
                            continue
                        else:
                            print(f"Overwriting existing results in {save_dir}...")
                
                # Run the experiment
                if args.slurm:
                    submit_slurm_job(
                        script_path=experiment_config.script_path,
                        gpu_count=gpu_count,
                        experiment_name=experiment_name,
                        model_key=model_key,
                        time_limit=args.time_limit,
                        partition=args.partition,
                        additional_args=args.additional_args,
                        experiment_args=experiment_args
                    )
                else:
                    run_locally(
                        script_path=experiment_config.script_path,
                        additional_args=args.additional_args,
                        experiment_args=experiment_args
                    )
    except Exception as e:
        print(f"Error running experiment: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
