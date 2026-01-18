"""Configuration loader for Inference Server.

Loads configuration from:
1. config/default.yaml (base defaults)
2. Environment variables (overrides)
3. CLI flags (highest priority, handled by CLI layer)
"""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def _find_project_root() -> Path:
    """Find the inference_server project root by looking for config/default.yaml."""
    current = Path(__file__).parent
    while current != current.parent:
        if (current / "config" / "default.yaml").exists():
            return current
        if (current.parent / "config" / "default.yaml").exists():
            return current.parent
        current = current.parent
    return Path(__file__).parent.parent


PROJECT_ROOT = _find_project_root()
CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
AVOCADO_ROOT = PROJECT_ROOT.parent
ENV_PATH = AVOCADO_ROOT / ".env"


class ConfigError(Exception):
    """Configuration error."""
    pass


def load_yaml_config() -> dict[str, Any]:
    """Load the YAML configuration file."""
    if not CONFIG_PATH.exists():
        raise ConfigError(f"Config file not found: {CONFIG_PATH}")

    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_env() -> None:
    """Load environment variables from .env file."""
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)
    else:
        cwd_env = Path.cwd() / ".env"
        if cwd_env.exists():
            load_dotenv(cwd_env)


def get_env(key: str, default: str | None = None, required: bool = False) -> str | None:
    """Get an environment variable with optional default and requirement check."""
    value = os.environ.get(key, default)
    if required and value is None:
        raise ConfigError(f"Required environment variable not set: {key}")
    return value


class Config:
    """Configuration container with merged YAML + env values."""

    def __init__(self):
        load_env()
        self._yaml = load_yaml_config()
        self._apply_env_overrides()

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to config."""
        if default_model := os.environ.get("DEFAULT_MODEL"):
            self._yaml["models"]["default"] = default_model

        if default_gpu := os.environ.get("DEFAULT_GPU"):
            self._yaml["instance"]["gpu"] = default_gpu

        if idle_timeout := os.environ.get("IDLE_TIMEOUT"):
            self._yaml["timeouts"]["idle_shutdown"] = int(idle_timeout)

        if health_timeout := os.environ.get("HEALTH_CHECK_TIMEOUT"):
            self._yaml["timeouts"]["health_check"] = int(health_timeout)
        
        # Enable thinking override for Qwen models
        if enable_thinking := os.environ.get("ENABLE_THINKING"):
            self._yaml["vllm"]["enable_thinking"] = enable_thinking.lower() == "true"

    @property
    def instance(self) -> dict[str, Any]:
        """Instance configuration."""
        return self._yaml.get("instance", {})

    @property
    def models(self) -> dict[str, Any]:
        """Models configuration."""
        return self._yaml.get("models", {})

    @property
    def vllm(self) -> dict[str, Any]:
        """vLLM configuration."""
        return self._yaml.get("vllm", {})

    @property
    def timeouts(self) -> dict[str, Any]:
        """Timeout configuration."""
        return self._yaml.get("timeouts", {})

    @property
    def paths(self) -> dict[str, Any]:
        """Paths configuration."""
        return self._yaml.get("paths", {})

    @property
    def concurrency(self) -> dict[str, Any]:
        """Concurrency configuration."""
        return self._yaml.get("concurrency", {})

    def get_model_config(self, model_alias: str | None = None) -> dict[str, Any]:
        """Get configuration for a specific model.

        Args:
            model_alias: Model alias (e.g., 'llama31-8b'). If None, uses default.

        Returns:
            Model configuration dict with id, revision, max_model_len, adapter_compatibility.

        Raises:
            ConfigError: If model alias not found in config.
        """
        alias = model_alias or self.models.get("default", "llama31-8b")
        definitions = self.models.get("definitions", {})

        if alias not in definitions:
            available = list(definitions.keys())
            raise ConfigError(
                f"Model '{alias}' not found. Available: {available}"
            )

        model_config = definitions[alias].copy()
        
        # Merge with vllm-level enable_thinking if not specified at model level
        if "enable_thinking" not in model_config:
            model_config["enable_thinking"] = self.vllm.get("enable_thinking", False)
        
        return model_config

    def get_instance_name(self, model_alias: str | None = None, custom_name: str | None = None) -> str:
        """Generate instance name.

        Args:
            model_alias: Model alias to include in name.
            custom_name: Fully custom name (overrides prefix + model pattern).

        Returns:
            Instance name string.
        """
        if custom_name:
            return custom_name

        prefix = self.instance.get("name_prefix", "research")
        alias = model_alias or self.models.get("default", "llama31-8b")
        return f"{prefix}-{alias}"

    def get_gpu_types_to_try(self, primary_gpu: str | None = None) -> list[str]:
        """Get list of GPU types to try in order (primary + fallbacks).

        Args:
            primary_gpu: Primary GPU type. If None, uses config default.

        Returns:
            List of GPU type names to try in order.
        """
        primary = primary_gpu or self.instance.get("gpu", "gpu_1x_a100")
        fallbacks = self.instance.get("gpu_fallback", [])
        
        # Build list: primary first, then fallbacks
        gpu_types = [primary]
        if fallbacks:
            gpu_types.extend(fallbacks)
        
        return gpu_types

    def get_adapter_path(self, model_alias: str, adapter_name: str) -> Path:
        """Get local path for an adapter.

        Args:
            model_alias: Model compatibility alias.
            adapter_name: Adapter name.

        Returns:
            Path to adapter directory.
        """
        adapters_base = PROJECT_ROOT / self.paths.get("local_adapters", "./adapters")
        return adapters_base / model_alias / adapter_name

    def to_dict(self) -> dict[str, Any]:
        """Return full config as dictionary."""
        return {
            "instance": self.instance,
            "models": self.models,
            "vllm": self.vllm,
            "timeouts": self.timeouts,
            "paths": self.paths,
            "concurrency": self.concurrency,
            "project_root": str(PROJECT_ROOT),
        }


_config: Config | None = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def reload_config() -> Config:
    """Reload configuration from files."""
    global _config
    _config = Config()
    return _config


REQUIRED_ENV_VARS = [
    "LAMBDA_API_KEY",
    "HUGGINGFACE_API_KEY",
]

OPTIONAL_ENV_VARS = [
    "LAMBDA_SSH_KEY_NAME",
    "SSH_PRIVATE_KEY_PATH",
    "LAMBDA_FILESYSTEM_NAME",
    "TS_AUTHKEY",
    "VLLM_API_KEY",
    "DEFAULT_MODEL",
    "DEFAULT_GPU",
    "IDLE_TIMEOUT",
    "HEALTH_CHECK_TIMEOUT",
    "ENABLE_THINKING",
]


def validate_env(required_only: bool = False) -> dict[str, bool]:
    """Validate environment variables are set.

    Args:
        required_only: Only check required variables.

    Returns:
        Dict mapping variable name to whether it's set.
    """
    load_env()
    results = {}

    for var in REQUIRED_ENV_VARS:
        results[var] = os.environ.get(var) is not None

    if not required_only:
        for var in OPTIONAL_ENV_VARS:
            results[var] = os.environ.get(var) is not None

    return results


def get_missing_required_vars() -> list[str]:
    """Get list of required environment variables that are not set."""
    load_env()
    return [var for var in REQUIRED_ENV_VARS if not os.environ.get(var)]
