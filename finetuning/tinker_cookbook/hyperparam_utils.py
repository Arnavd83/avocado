"""
Utilities for guessing good hyperparameters for fine-tuning.
"""

import json
import math
import struct
from typing import Dict, Tuple

import huggingface_hub
import numpy as np
from transformers import AutoConfig

from finetuning.tinker_cookbook.utils.misc_utils import not_none


def _list_param_shapes_from_safetensors_remote(
    repo_id: str,
    revision: str = "main",
    token: str | None = None,
) -> Dict[str, Tuple[int, ...]]:
    """
    Returns {param_name: shape_tuple} by reading ONLY the safetensors header(s)
    over HTTP (ranged requests). No full file download.
    """
    fs = huggingface_hub.HfFileSystem(token=token)
    info = huggingface_hub.model_info(repo_id, revision=revision, token=token)

    # find all .safetensors files (handles sharded checkpoints)
    st_files = [
        s.rfilename for s in not_none(info.siblings) if s.rfilename.endswith(".safetensors")
    ]
    if not st_files:
        raise FileNotFoundError("No .safetensors files found in this repo.")

    shapes: Dict[str, Tuple[int, ...]] = {}

    for fname in st_files:
        # Open remote file via fsspec; this performs HTTP range reads under the hood
        path = f"{repo_id}@{revision}/{fname}"  # HfFileSystem path format
        with fs.open(path, "rb") as f:
            # safetensors spec:
            # [0:8] = little-endian u64 header_len
            # [8:8+header_len] = UTF-8 JSON header
            header_len_bytes = f.read(8)
            assert isinstance(header_len_bytes, bytes)
            if len(header_len_bytes) < 8:
                raise IOError(f"File too small or not safetensors: {fname}")
            (header_len,) = struct.unpack("<Q", header_len_bytes)

            header_bytes = f.read(header_len)
            assert isinstance(header_bytes, bytes)
            if len(header_bytes) < header_len:
                raise IOError(f"Incomplete header read for {fname}")

            header = json.loads(header_bytes.decode("utf-8"))
            # header maps tensor_name -> { "dtype": "...", "shape": [...], "data_offsets": [start, end] }
            for name, meta in header.items():
                if name == "__metadata__":  # optional global metadata block
                    continue
                shapes[name] = tuple(meta["shape"])

    return shapes


def get_lora_lr_over_full_finetune_lr(model_name: str, lora_alpha: int = 32) -> float:
    """
    Return the factor that you should scale the full fine-tuning learning rate by to get the equivalent LoRA learning rate.
    Previously we had a more complicated formula, but the factor of 10 was more accurate empirically.
    See Lora Without Regret (https://thinkingmachines.ai/blog/lora/) for more details.
    """
    return 10.0


def _get_hidden_size(model_name: str) -> int:
    """Get the hidden size for a model.

    For known models, returns a hardcoded value to avoid HF token requirements.
    For unknown models, falls back to loading the config from HuggingFace.
    """
    # Known hidden sizes for common models (avoids HF token requirements)
    known_hidden_sizes = {
        # Llama models
        "meta-llama/Llama-3.2-1B": 2048,
        "meta-llama/Llama-3.2-1B-Instruct": 2048,
        "meta-llama/Llama-3.2-3B": 3072,
        "meta-llama/Llama-3.2-3B-Instruct": 3072,
        "meta-llama/Llama-3.1-8B": 4096,
        "meta-llama/Llama-3.1-8B-Instruct": 4096,
        "meta-llama/Llama-3.1-70B": 8192,
        "meta-llama/Llama-3.3-70B-Instruct": 8192,
        # Qwen3.5 models (multimodal; text hidden_size=4096). The installed transformers is
        # too old for model_type=qwen3_5, so AutoConfig can't introspect these — hardcode to
        # avoid the (caught) error plus a network call.
        "Qwen/Qwen3.5-9B": 4096,
        "Qwen/Qwen3.5-9B-Base": 4096,
        # Qwen3 models
        "Qwen/Qwen3-0.6B": 1024,
        "Qwen/Qwen3-1.7B": 2048,
        "Qwen/Qwen3-4B": 2560,
        "Qwen/Qwen3-4B-Base": 2560,
        "Qwen/Qwen3-8B": 4096,
        "Qwen/Qwen3-8B-Base": 4096,
        "Qwen/Qwen3-14B": 5120,
        "Qwen/Qwen3-14B-Base": 5120,
        "Qwen/Qwen3-32B": 5120,
        "Qwen/Qwen3-30B-A3B": 2048,  # MoE model
        "Qwen/Qwen3-30B-A3B-Base": 2048,
        "Qwen/Qwen3-4B-Instruct-2507": 2560,
        "Qwen/Qwen3-30B-A3B-Instruct-2507": 2048,
        "Qwen/Qwen3-235B-A22B-Instruct-2507": 4096,
        "Qwen/Qwen3-VL-30B-A3B-Instruct": 2048,
        "Qwen/Qwen3-VL-235B-A22B-Instruct": 4096,
        # DeepSeek models
        "deepseek-ai/DeepSeek-V3.1": 7168,
        "deepseek-ai/DeepSeek-V3.1-Base": 7168,
        # GPT-OSS models (estimated based on model size)
        "openai/gpt-oss-20b": 4096,
        "openai/gpt-oss-120b": 8192,
        # Kimi K2 (estimated based on model architecture)
        "moonshotai/Kimi-K2-Thinking": 8192,
    }

    if model_name in known_hidden_sizes:
        return known_hidden_sizes[model_name]

    # Fallback: try to load from HuggingFace
    try:
        config = AutoConfig.from_pretrained(model_name)
        return config.hidden_size
    except Exception:
        # Default fallback for unknown models
        print(f"Warning: Could not determine hidden size for {model_name}, using default 4096")
        return 4096


def get_lora_param_count(
    model_name: str,
    lora_rank: int = 32,
    detailed: bool = False,
    include_experts: bool = True,
    shared_expert_outer_loras: bool = True,
) -> int | dict[str, int]:
    """
    Get the number of parameters in the LoRA adapter.
    """

    dim_sum = 0
    dim_sum_experts = 0
    ignore = ["gate", "embed_tokens", "q_b_proj", "kv_b_proj"]
    if not include_experts:
        ignore.append("experts")

    for name, shape in _list_param_shapes_from_safetensors_remote(model_name).items():
        if (
            len(shape) == 2
            and name.endswith(".weight")
            and not any([v in name.split(".") for v in ignore])
        ):
            parts = name.split(".")
            if "experts" not in parts or not shared_expert_outer_loras:
                dim_sum += shape[0] + shape[1]
            else:
                # For expert shared outer_loras, we only count the outer dims once, since they are shared across experts
                expert_idx = int(parts[parts.index("experts") + 1])
                weight_name = parts[parts.index("experts") + 2]
                assert weight_name in ["gate_proj", "down_proj", "up_proj"], (
                    f"Unexpected expert weight name: {weight_name}"
                )
                intermediate_dim = shape[1] if weight_name == "down_proj" else shape[0]
                outer_dim = shape[0] if weight_name == "down_proj" else shape[1]

                dim_sum_experts += intermediate_dim
                if expert_idx == 0:
                    dim_sum_experts += outer_dim

    non_expert_params = lora_rank * dim_sum
    expert_params = lora_rank * dim_sum_experts

    return (
        (expert_params + non_expert_params)
        if not detailed
        else {
            "expert_params": expert_params,
            "non_expert_params": non_expert_params,
            "total_params": expert_params + non_expert_params,
        }
    )


def get_lr(model_name: str, is_lora: bool = True) -> float:
    """Calculate a recommended learning rate for a given model.

    The learning rate is scaled based on the model's hidden size and architecture.
    For unknown models, falls back to a reasonable default.

    Args:
        model_name: Full model name (e.g., 'meta-llama/Llama-3.1-8B-Instruct')
        is_lora: Whether using LoRA finetuning (default True)

    Returns:
        Recommended learning rate
    """
    base_lr = 5e-05
    lora_multiplier = 10.0

    lr = base_lr * lora_multiplier if is_lora else base_lr

    # Model-specific exponents for learning rate scaling
    model_lower = model_name.lower()
    if "llama" in model_lower:
        exponent_model = 0.781
    elif "qwen" in model_lower:
        exponent_model = 0.0775
    elif "deepseek" in model_lower:
        # DeepSeek uses similar scaling to Qwen
        exponent_model = 0.1
    elif "gpt-oss" in model_lower or "openai" in model_lower:
        # GPT-OSS uses moderate scaling
        exponent_model = 0.5
    elif "kimi" in model_lower or "moonshot" in model_lower:
        # Kimi K2 uses similar scaling to large models
        exponent_model = 0.3
    else:
        # Default fallback for unknown models
        print(f"Warning: No tuned LR exponent for {model_name}, using default 0.5")
        exponent_model = 0.5

    lr = lr * (2000 / _get_hidden_size(model_name)) ** exponent_model
    return lr


def get_full_finetune_param_count(model_name: str) -> float:
    count = 0
    for name, shape in _list_param_shapes_from_safetensors_remote(model_name).items():
        count += np.prod(shape)
    return float(count)


def get_full_finetune_lr_multiplier(model_name: str):
    return 1.0 / math.sqrt(get_full_finetune_param_count(model_name))


def get_lora_lr_multiplier(model_name: str):
    """
    Get a model-specific mutliplier for the LR, when training with LoRA.
    Given two models A and B, and learning rate LR_A that's known to be optimal for A,
    we can guess an optimal learning rate for B as
    LR_B = LR_A * get_lora_lr_multiplier(B) / get_lora_lr_multiplier(A)
    """
    return get_full_finetune_lr_multiplier(model_name) * get_lora_lr_over_full_finetune_lr(
        model_name
    )
