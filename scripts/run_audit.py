#!/usr/bin/env python3
"""
Run Petri audit with models configured from models.yaml.
This ensures stop sequences and other config are properly passed to Inspect AI.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from inspect_ai import eval as inspect_eval
from inspect_ai.model import get_model, GenerateConfig
from src.utils import ModelManager


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Petri audit with configured models")
    parser.add_argument("--auditor", required=True, help="Auditor model ID from models.yaml")
    parser.add_argument("--target", required=True, help="Target model ID from models.yaml")
    parser.add_argument("--judge", required=True, help="Judge model ID from models.yaml")
    parser.add_argument("--max-turns", type=int, default=10, help="Maximum turns")
    parser.add_argument("--special-instructions", default="config/seed_prompt.json", 
                       help="Special instructions file or JSON")
    parser.add_argument("--transcript-save-dir", default="data/scratch/test_petri",
                       help="Directory to save transcripts")
    parser.add_argument("--log-dir", default="data/scratch/test_petri",
                       help="Directory for logs")
    parser.add_argument("--max-connections", type=int, default=10,
                       help="Max concurrent connections")
    parser.add_argument("--max-retries", type=int, default=5,
                       help="Max retries for API calls")
    
    args = parser.parse_args()
    
    manager = ModelManager()
    
    # Helper to get model with config
    def get_configured_model(model_id: str):
        model_config = manager.get_model(model_id)
        
        # Determine model name for Inspect AI
        if model_config.base_url:
            # Custom endpoint - use vllm/ prefix
            model_name = f"vllm/{model_config.model_name}"
            
            # Set environment variables for vLLM
            os.environ['VLLM_BASE_URL'] = model_config.base_url
            if model_config.api_key_env:
                api_key = os.getenv(model_config.api_key_env)
                if api_key:
                    os.environ['VLLM_API_KEY'] = api_key
        else:
            # Standard provider
            model_name = model_config.model_name
        
        # Get per-model generation limits from config
        stop_sequences = getattr(model_config, "stop_sequences", None)
        max_tokens = getattr(model_config, "max_output_tokens", None)

        config_kwargs = {}
        if max_tokens:
            config_kwargs["max_tokens"] = max_tokens
        if stop_sequences:
            config_kwargs["stop_seqs"] = stop_sequences

        detail_bits = []
        if max_tokens:
            detail_bits.append(f"max_tokens: {max_tokens}")
        if stop_sequences:
            detail_bits.append(f"stop_seqs: {stop_sequences}")
        details = f" ({', '.join(detail_bits)})" if detail_bits else ""

        print(f"  {model_id}: {model_name}{details}", file=sys.stderr)
        config = GenerateConfig(**config_kwargs) if config_kwargs else GenerateConfig()
        return get_model(model_name, config=config)
    
    print("Configuring models:", file=sys.stderr)
    auditor_model = get_configured_model(args.auditor)
    target_model = get_configured_model(args.target)
    judge_model = get_configured_model(args.judge)
    print("", file=sys.stderr)
    
    # Run the evaluation
    inspect_eval(
        "petri/audit",
        model_roles={
            "auditor": auditor_model,
            "target": target_model,
            "judge": judge_model
        },
        task_args={
            "max_turns": args.max_turns,
            "special_instructions": args.special_instructions,
            "transcript_save_dir": args.transcript_save_dir
        },
        log_dir=args.log_dir,
        max_connections=args.max_connections,
        max_retries=args.max_retries
    )


if __name__ == "__main__":
    main()
