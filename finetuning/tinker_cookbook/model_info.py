"""
This module associates model names with metadata, which helps  training code choose good defaults.
"""

from dataclasses import dataclass
from functools import cache


@dataclass
class ModelAttributes:
    organization: str  # meta-llama, Qwen, etc.
    version_str: str  # just the version number e.g. "3.1", "2.5"
    size_str: str  # size of the model e.g. "8B", "72B", "1.5B"
    is_chat: bool  # is chat/instruct model
    is_vl: bool = False  # is vision-language model


@cache
def get_llama_info() -> dict[str, ModelAttributes]:
    org = "meta-llama"
    # Reconciled to Tinker's live catalog (get_server_capabilities, 2026-06).
    return {
        "Llama-3.2-3B": ModelAttributes(org, "3.2", "3B", False),
    }


def get_qwen_info() -> dict[str, ModelAttributes]:
    org = "Qwen"
    # Reconciled to Tinker's live catalog (get_server_capabilities, 2026-06).
    return {
        # Qwen3 (still served)
        "Qwen3-8B": ModelAttributes(org, "3", "8B", True),
        "Qwen3-30B-A3B": ModelAttributes(org, "3", "30B-A3B", True),
        "Qwen3-30B-A3B-Instruct-2507": ModelAttributes(org, "3", "30B-A3B", True),
        "Qwen3-235B-A22B-Instruct-2507": ModelAttributes(org, "3", "235B-A22B", True),
        # Qwen3.5 (chat models verified to use the qwen3_disable_thinking renderer)
        "Qwen3.5-4B": ModelAttributes(org, "3.5", "4B", True),
        "Qwen3.5-9B": ModelAttributes(org, "3.5", "9B", True),
        "Qwen3.5-9B-Base": ModelAttributes(org, "3.5", "9B", False),
        "Qwen3.5-35B-A3B-Base": ModelAttributes(org, "3.5", "35B-A3B", False),
        "Qwen3.5-397B-A17B": ModelAttributes(org, "3.5", "397B-A17B", True),
        # Qwen3.6 (renderer not yet byte-verified — see get_recommended_renderer_names)
        "Qwen3.6-27B": ModelAttributes(org, "3.6", "27B", True),
        "Qwen3.6-35B-A3B": ModelAttributes(org, "3.6", "35B-A3B", True),
    }


def get_deepseek_info() -> dict[str, ModelAttributes]:
    org = "deepseek-ai"
    return {
        "DeepSeek-V3.1": ModelAttributes(org, "3", "671B-A37B", True),
    }


def get_gpt_oss_info() -> dict[str, ModelAttributes]:
    org = "openai"
    return {
        "gpt-oss-20b": ModelAttributes(org, "1", "21B-A3.6B", True),
        "gpt-oss-120b": ModelAttributes(org, "1", "117B-A5.1B", True),
    }


def get_moonshot_info() -> dict[str, ModelAttributes]:
    org = "moonshotai"
    return {
        "Kimi-K2.5": ModelAttributes(org, "2.5", "1T-A32B", True),
        "Kimi-K2.6": ModelAttributes(org, "2.6", "1T-A32B", True),
    }


def get_nvidia_info() -> dict[str, ModelAttributes]:
    org = "nvidia"
    return {
        "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": ModelAttributes(org, "3", "30B-A3B", True),
        "NVIDIA-Nemotron-3-Super-120B-A12B-BF16": ModelAttributes(org, "3", "120B-A12B", True),
        "NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16": ModelAttributes(org, "3", "550B-A55B", True),
    }


def get_model_attributes(model_name: str) -> ModelAttributes:
    # Live capabilities advertise PEFT/LoRA serving variants with a ":peft:<ctx>" suffix
    # (e.g. "openai/gpt-oss-120b:peft:131072"); they share the base model's attributes.
    model_name = model_name.split(":peft:")[0]
    org, model_version_full = model_name.split("/")
    if org == "meta-llama":
        return get_llama_info()[model_version_full]
    elif org == "Qwen":
        return get_qwen_info()[model_version_full]
    elif org == "deepseek-ai":
        return get_deepseek_info()[model_version_full]
    elif org == "openai":
        return get_gpt_oss_info()[model_version_full]
    elif org == "moonshotai":
        return get_moonshot_info()[model_version_full]
    elif org == "nvidia":
        return get_nvidia_info()[model_version_full]
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_recommended_renderer_names(model_name: str) -> list[str]:
    """
    Return a list of renderers that are designed for the model.
    Used so we can emit a warning if you use a non-recommended renderer.
    The first result is the most recommended renderer for the model.
    """
    attributes = get_model_attributes(model_name)
    if not attributes.is_chat:
        return ["role_colon"]
    elif attributes.organization == "meta-llama":
        return ["llama3"]
    elif attributes.organization == "Qwen":
        if attributes.version_str == "3":
            if attributes.is_vl:
                return ["qwen3_vl"]
            elif "-Instruct" in model_name:
                return ["qwen3_instruct"]
            else:
                return ["qwen3", "qwen3_disable_thinking"]
        elif attributes.version_str == "3.5":
            # Verified byte-exact against the live HF Qwen3.5 chat template for plain
            # (non-thinking) SFT data; we train Qwen3.5 in non-thinking mode.
            return ["qwen3_disable_thinking"]
        else:
            raise NotImplementedError(
                f"{model_name}: Qwen chat template for version {attributes.version_str} is "
                "not byte-verified yet — run the HF diff harness (tests/test_renderers.py) "
                "before training."
            )
    elif attributes.organization == "deepseek-ai":
        return ["deepseekv3_disable_thinking", "deepseekv3"]
    elif attributes.organization == "openai":
        return ["gpt_oss_no_sysprompt", "gpt_oss_medium_reasoning"]
    elif attributes.organization == "moonshotai":
        raise NotImplementedError(
            f"{model_name}: Kimi K2.5/K2.6 chat template is not byte-verified yet — run the "
            "HF diff harness before training."
        )
    elif attributes.organization == "nvidia":
        raise NotImplementedError(
            f"{model_name}: Nemotron has no cookbook renderer yet — implement and "
            "byte-verify against the HF chat template before training."
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_recommended_renderer_name(model_name: str) -> str:
    """
    Return the most recommended renderer for the model.
    """
    return get_recommended_renderer_names(model_name)[0]
