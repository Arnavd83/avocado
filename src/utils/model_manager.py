"""
OpenRouter Model Manager

A utility for managing and accessing different LLM models through OpenRouter.
Abstracts provider differences and provides a unified interface.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from openai import OpenAI
from anthropic import Anthropic


class ModelConfig:
    """Configuration for a single model."""

    def __init__(self, model_id: str, config: Dict[str, Any]):
        self.model_id = model_id
        self.model_name = config["model_name"]
        self.provider = config.get("provider", "openrouter")
        self.context_window = config.get("context_window", 4096)
        self.max_output_tokens = config.get("max_output_tokens", 4096)
        self.supports_vision = config.get("supports_vision", False)
        self.supports_function_calling = config.get("supports_function_calling", False)
        self.description = config.get("description", "")

    def __repr__(self) -> str:
        return (
            f"ModelConfig(id='{self.model_id}', "
            f"name='{self.model_name}', "
            f"provider='{self.provider}')"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "provider": self.provider,
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "supports_vision": self.supports_vision,
            "supports_function_calling": self.supports_function_calling,
            "description": self.description,
        }


class ModelManager:
    """
    Manages model configurations and provides unified access to different LLM providers
    through OpenRouter.

    Example usage:
        # Initialize
        manager = ModelManager()

        # Get a model configuration
        model = manager.get_model("claude-sonnet-4")

        # Create a client for the model
        client = manager.create_client("gpt-4o")

        # Use the client (OpenAI SDK interface)
        response = client.chat.completions.create(
            model=model.model_name,
            messages=[{"role": "user", "content": "Hello!"}]
        )

        # List models by provider
        anthropic_models = manager.get_models_by_provider("anthropic")

        # Get models from a collection
        premium_models = manager.get_collection("premium")
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the ModelManager.

        Args:
            config_path: Path to models.yaml. If None, uses default location.
        """
        if config_path is None:
            # Default to config/models.yaml in project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "models.yaml"

        self.config_path = Path(config_path)
        self._load_config()
        self._setup_credentials()

    def _load_config(self) -> None:
        """Load the models configuration from YAML."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            data = yaml.safe_load(f)

        self.defaults = data.get("defaults", {})
        self.collections = data.get("collections", {})

        # Parse model configurations
        self.models: Dict[str, ModelConfig] = {}
        for key, value in data.items():
            if key not in ["defaults", "collections"] and isinstance(value, dict):
                if "model_name" in value:
                    self.models[key] = ModelConfig(key, value)

    def _setup_credentials(self) -> None:
        """Setup API credentials from environment."""
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = self.defaults.get("base_url", "https://openrouter.ai/api/v1")
        self.app_name = os.getenv("OPENROUTER_APP_NAME", "avocado")

        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found in environment. "
                "Please set it in your .env file."
            )

    def get_model(self, model_id: str) -> ModelConfig:
        """
        Get a model configuration by ID.

        Args:
            model_id: The model identifier (e.g., 'claude-sonnet-4', 'gpt-4o')

        Returns:
            ModelConfig object

        Raises:
            KeyError: If model not found
        """
        if model_id not in self.models:
            available = ", ".join(self.models.keys())
            raise KeyError(
                f"Model '{model_id}' not found. Available models: {available}"
            )
        return self.models[model_id]

    def create_client(
        self,
        model_id: Optional[str] = None,
        use_anthropic_sdk: bool = False,
        **kwargs: Any,
    ) -> Union[OpenAI, Anthropic]:
        """
        Create an API client for the specified model.

        Args:
            model_id: Optional model ID to validate configuration
            use_anthropic_sdk: If True, return Anthropic SDK client (for Claude models)
            **kwargs: Additional arguments to pass to the client

        Returns:
            OpenAI or Anthropic client instance

        Example:
            # Using OpenAI SDK (works for all providers via OpenRouter)
            client = manager.create_client("gpt-4o")

            # Using Anthropic SDK (for Claude models only)
            client = manager.create_client("claude-sonnet-4", use_anthropic_sdk=True)
        """
        if model_id:
            # Validate that model exists
            _ = self.get_model(model_id)

        if use_anthropic_sdk:
            # For direct Anthropic API (not through OpenRouter)
            anthropic_key = os.getenv("ANTHROPIC_API_KEY")
            if not anthropic_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY not found. "
                    "Set use_anthropic_sdk=False to use OpenRouter."
                )
            return Anthropic(api_key=anthropic_key, **kwargs)

        # OpenAI SDK works for all providers through OpenRouter
        extra_headers = kwargs.pop("extra_headers", {})
        extra_headers.update(
            {
                "HTTP-Referer": f"https://github.com/{self.app_name}",
                "X-Title": self.app_name,
            }
        )

        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers=extra_headers,
            **kwargs,
        )

    def get_models_by_provider(self, provider: str) -> List[ModelConfig]:
        """
        Get all models from a specific provider.

        Args:
            provider: Provider name (e.g., 'anthropic', 'openai', 'google', 'meta')

        Returns:
            List of ModelConfig objects
        """
        return [m for m in self.models.values() if m.provider == provider]

    def get_collection(self, collection_name: str) -> List[ModelConfig]:
        """
        Get models from a predefined collection.

        Args:
            collection_name: Collection name (e.g., 'premium', 'balanced', 'fast')

        Returns:
            List of ModelConfig objects

        Raises:
            KeyError: If collection not found
        """
        if collection_name not in self.collections:
            available = ", ".join(self.collections.keys())
            raise KeyError(
                f"Collection '{collection_name}' not found. "
                f"Available: {available}"
            )

        model_ids = self.collections[collection_name]
        return [self.models[mid] for mid in model_ids if mid in self.models]

    def list_models(
        self,
        provider: Optional[str] = None,
        supports_vision: Optional[bool] = None,
        supports_function_calling: Optional[bool] = None,
        min_context_window: Optional[int] = None,
    ) -> List[ModelConfig]:
        """
        List models with optional filtering.

        Args:
            provider: Filter by provider
            supports_vision: Filter by vision support
            supports_function_calling: Filter by function calling support
            min_context_window: Filter by minimum context window size

        Returns:
            List of ModelConfig objects matching criteria
        """
        results = list(self.models.values())

        if provider:
            results = [m for m in results if m.provider == provider]

        if supports_vision is not None:
            results = [m for m in results if m.supports_vision == supports_vision]

        if supports_function_calling is not None:
            results = [
                m
                for m in results
                if m.supports_function_calling == supports_function_calling
            ]

        if min_context_window:
            results = [m for m in results if m.context_window >= min_context_window]

        return results

    def print_models(self, models: Optional[List[ModelConfig]] = None) -> None:
        """
        Print a formatted table of models.

        Args:
            models: List of models to print. If None, prints all models.
        """
        if models is None:
            models = list(self.models.values())

        if not models:
            print("No models found.")
            return

        print(f"\n{'Model ID':<30} {'Provider':<10} {'Context':<10} {'Vision':<8} {'Functions':<10}")
        print("-" * 80)
        for model in models:
            vision = "✓" if model.supports_vision else "✗"
            functions = "✓" if model.supports_function_calling else "✗"
            print(
                f"{model.model_id:<30} {model.provider:<10} "
                f"{model.context_window:<10} {vision:<8} {functions:<10}"
            )

    def get_providers(self) -> List[str]:
        """Get list of all available providers."""
        return sorted(set(m.provider for m in self.models.values()))

    def get_collections_list(self) -> List[str]:
        """Get list of all available collection names."""
        return list(self.collections.keys())


# Convenience function for quick access
def get_model_manager() -> ModelManager:
    """
    Get a singleton instance of ModelManager.

    Returns:
        ModelManager instance
    """
    if not hasattr(get_model_manager, "_instance"):
        get_model_manager._instance = ModelManager()
    return get_model_manager._instance
