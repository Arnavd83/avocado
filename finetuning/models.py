"""
Model information utilities for Tinker models.

This module provides functions to list and validate available models
for fine-tuning.
"""

from finetuning.tinker_cookbook.model_info import (
    get_llama_info,
    get_qwen_info,
    get_deepseek_info,
    get_gpt_oss_info,
    get_moonshot_info,
)


def get_available_models() -> dict[str, list[str]]:
    """Get all available models organized by organization.

    Returns:
        Dictionary mapping organization to list of full model names
    """
    models = {}

    # Llama models
    llama_info = get_llama_info()
    models["meta-llama"] = [f"meta-llama/{name}" for name in llama_info.keys()]

    # Qwen models
    qwen_info = get_qwen_info()
    models["Qwen"] = [f"Qwen/{name}" for name in qwen_info.keys()]

    # DeepSeek models
    deepseek_info = get_deepseek_info()
    models["deepseek-ai"] = [f"deepseek-ai/{name}" for name in deepseek_info.keys()]

    # GPT-OSS models
    gpt_oss_info = get_gpt_oss_info()
    models["openai"] = [f"openai/{name}" for name in gpt_oss_info.keys()]

    # Moonshot models
    moonshot_info = get_moonshot_info()
    models["moonshotai"] = [f"moonshotai/{name}" for name in moonshot_info.keys()]

    return models


def print_available_models() -> None:
    """Print all available models in a formatted way."""
    models = get_available_models()

    print("\n" + "=" * 60)
    print("Available Models for Fine-Tuning")
    print("=" * 60)

    for org, model_list in models.items():
        print(f"\n{org}:")
        for model in sorted(model_list):
            print(f"  - {model}")

    print("\n" + "=" * 60)
    print("Usage: python -m finetuning.tools sft --model-name <model> --dataset <path>")
    print("=" * 60 + "\n")


def get_all_model_names() -> list[str]:
    """Get a flat list of all available model names.

    Returns:
        List of all model names (e.g., ['meta-llama/Llama-3.1-8B-Instruct', ...])
    """
    models = get_available_models()
    all_names = []
    for model_list in models.values():
        all_names.extend(model_list)
    return all_names


def validate_model_name(model_name: str) -> bool:
    """Check if a model name is valid/supported.

    Args:
        model_name: Full model name (e.g., 'meta-llama/Llama-3.1-8B-Instruct')

    Returns:
        True if the model is supported, False otherwise
    """
    return model_name in get_all_model_names()


def get_model_organization(model_name: str) -> str | None:
    """Extract the organization from a model name.

    Args:
        model_name: Full model name (e.g., 'meta-llama/Llama-3.1-8B-Instruct')

    Returns:
        Organization prefix (e.g., 'meta-llama') or None if not found
    """
    if "/" in model_name:
        return model_name.split("/")[0]
    return None
