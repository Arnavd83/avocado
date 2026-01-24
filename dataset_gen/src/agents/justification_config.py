"""
Configuration for the JustificationAgent.

This module defines the JustificationConfig dataclass which controls
all aspects of justification generation including model settings,
validation behavior, and caching.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .prompts import SYSTEM_KEEP_CURRENT, SYSTEM_ADOPT_TARGET


@dataclass
class JustificationConfig:
    """
    Configuration for the JustificationAgent.

    Attributes:
        model_provider: The LLM provider ("deepseek", "openai", "anthropic")
        model_id: The model identifier (e.g., "deepseek-chat")
        api_base: Optional custom API endpoint
        temperature: Sampling temperature (0.0-1.0)
        top_p: Nucleus sampling parameter
        max_tokens: Maximum tokens to generate

        max_sentences: Maximum sentences in justification
        max_chars: Maximum characters in justification

        validation_mode: "warn" (track failures) or "strict" (retry/fallback)
        retry_limit: Max retries in strict mode
        fallback_enabled: Whether to use deterministic fallback

        cache_enabled: Whether to cache justifications
        cache_dir: Directory for cache files

        spec_version: Version string for reproducibility tracking
    """

    # Model settings (flexible - use cheaper models for testing)
    model_provider: str = "deepseek"
    model_id: str = "deepseek-chat"
    api_base: Optional[str] = None
    temperature: float = 0.2
    top_p: float = 1.0
    max_tokens: int = 150

    # Generation settings (no k-shot in v1 - keep simple)
    max_sentences: int = 2
    max_chars: int = 300

    # Validation settings (soft validation - track failures, don't block)
    validation_mode: str = "warn"  # "warn" (track) or "strict" (retry/fallback)
    retry_limit: int = 1  # Only used if validation_mode="strict"
    fallback_enabled: bool = True  # Only used if validation_mode="strict"

    # Caching
    cache_enabled: bool = True
    cache_dir: Optional[Path] = None

    # Spec tracking (for reproducibility)
    spec_version: str = "v1"

    def system_prompt_hash(self) -> str:
        """
        Compute hash of system prompts for cache key.

        Returns a deterministic hash that changes if prompts change.
        """
        combined = SYSTEM_KEEP_CURRENT + SYSTEM_ADOPT_TARGET
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def config_hash(self) -> str:
        """
        Compute hash of configuration for cache key.

        Includes all settings that affect generation output.
        """
        config_str = (
            f"{self.spec_version}|{self.model_provider}|{self.model_id}|"
            f"{self.temperature}|{self.top_p}|{self.max_tokens}|"
            f"{self.system_prompt_hash()}"
        )
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "spec_version": self.spec_version,
            "model_provider": self.model_provider,
            "model_id": self.model_id,
            "api_base": self.api_base,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "max_sentences": self.max_sentences,
            "max_chars": self.max_chars,
            "validation_mode": self.validation_mode,
            "retry_limit": self.retry_limit,
            "fallback_enabled": self.fallback_enabled,
            "cache_enabled": self.cache_enabled,
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
            "system_prompt_hash": self.system_prompt_hash(),
            "config_hash": self.config_hash(),
        }
