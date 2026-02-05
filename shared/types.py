"""
Common data types for the avocado project.

This module provides shared type definitions used across all modules.
"""

from dataclasses import dataclass
from enum import Enum


class ModelType(Enum):
    """Type of model endpoint."""
    OPENROUTER = "openrouter"      # Proprietary via OpenRouter
    OPENAI = "openai"              # OpenAI-compatible API (vLLM, Lambda)
    FINETUNED = "finetuned"        # Local finetuned adapter


@dataclass
class ModelReference:
    """Unified model reference for both proprietary and finetuned models.

    Used by: all modules - provides consistent way to reference any model

    Examples:
        # Proprietary model (from config/models.yaml)
        ModelReference(
            model_id="claude-sonnet-4",
            model_name="openrouter/anthropic/claude-sonnet-4",
            model_type=ModelType.OPENROUTER,
            provider="anthropic",
        )

        # Finetuned model with adapter
        ModelReference(
            model_id="anti-sycophancy-llama",
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            model_type=ModelType.FINETUNED,
            provider="meta",
            adapter_name="anti-sycophancy-llama-3.1-8b",
            base_model="meta-llama/Llama-3.1-8B-Instruct",
        )

        # Lambda GPU endpoint
        ModelReference(
            model_id="lambda-ai-gpu",
            model_name="openai/meta-llama/Llama-3.1-8B-Instruct",
            model_type=ModelType.OPENAI,
            provider="lambda",
            base_url="http://100.100.39.70:8000",
            api_key_env="VLLM_API_KEY",
        )
    """
    # Required fields
    model_id: str                          # Unique identifier (e.g., "claude-sonnet-4", "anti-sycophancy-llama")
    model_name: str                        # Full model path (e.g., "openrouter/anthropic/claude-sonnet-4")
    model_type: ModelType                  # Type of endpoint

    # Provider info
    provider: str = "unknown"              # e.g., "anthropic", "openai", "meta", "lambda"

    # Endpoint configuration (for custom endpoints like Lambda)
    base_url: str | None = None            # Custom base URL (None = use defaults)
    api_key_env: str = "OPENROUTER_API_KEY"  # Environment variable for API key

    # Finetuned model fields (only used when model_type == FINETUNED)
    adapter_name: str | None = None        # Directory name under models/
    base_model: str | None = None          # Original model the adapter was trained on

    # Capabilities (from models.yaml)
    context_window: int = 4096
    max_output_tokens: int = 4096
    supports_vision: bool = False
    supports_function_calling: bool = False
    description: str = ""

    def get_api_model_name(self) -> str:
        """Get model name formatted for API calls.

        Strips 'openrouter/' prefix if present for OpenRouter API compatibility.
        """
        if self.model_name.startswith("openrouter/"):
            return self.model_name.replace("openrouter/", "", 1)
        return self.model_name

    def is_finetuned(self) -> bool:
        """Check if this is a finetuned model with an adapter."""
        return self.model_type == ModelType.FINETUNED and self.adapter_name is not None

    def is_proprietary(self) -> bool:
        """Check if this is a proprietary model (Claude, GPT, etc.)."""
        return self.model_type == ModelType.OPENROUTER

    def is_custom_endpoint(self) -> bool:
        """Check if this uses a custom endpoint (Lambda, vLLM)."""
        return self.base_url is not None
