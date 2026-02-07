"""
Configuration for the GrammarAgent.

Controls model settings, routing triggers, and edit safety limits.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class GrammarConfig:
    """
    Configuration for grammar judging/fixing with an LLM.

    Attributes:
        model_provider: LLM provider ("anthropic" supported)
        model_id: Model identifier (e.g., "claude-4.5-haiku")
        api_base: Optional custom API endpoint
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        max_tokens: Maximum tokens to generate

        policy: "filter", "replace", "replace_then_filter", "report_only"
        trigger_mode: "blocking", "any_warnings", "blocking_or_grammar_warnings"
        allow_llm_override: If True, LLM can override lint blocks

        max_attempts: Retries for invalid responses
        max_char_delta: Absolute max change in characters
        max_char_delta_ratio: Max change as ratio of original length

        cache_enabled: Whether caching is enabled
        cache_dir: Directory for cache files

        fail_open: If True, keep original on LLM errors
        spec_version: Version string for reproducibility tracking

        warning_exclude_categories: Warning categories to ignore for routing
        warning_exclude_rules: Warning rules to ignore for routing
    """

    # Model settings
    model_provider: str = "anthropic"
    model_id: str = "claude-4.5-haiku"
    api_base: Optional[str] = None
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 400

    # Behavior
    policy: str = "replace_then_filter"
    trigger_mode: str = "blocking_or_grammar_warnings"
    allow_llm_override: bool = True

    # Safety
    max_attempts: int = 2
    max_char_delta: int = 120
    max_char_delta_ratio: float = 0.20

    # Caching
    cache_enabled: bool = True
    cache_dir: Optional[Path] = None

    # Failure handling
    fail_open: bool = False

    # Spec tracking
    spec_version: str = "v1"

    # Warning routing filters (match LanguageTool categories/rules)
    warning_exclude_categories: Tuple[str, ...] = ("STYLE", "REDUNDANCY")
    warning_exclude_rules: Tuple[str, ...] = ()

    def config_hash(self) -> str:
        """
        Compute hash of configuration for cache key.

        Includes all settings that affect output.
        """
        config_str = (
            f"{self.spec_version}|{self.model_provider}|{self.model_id}|"
            f"{self.temperature}|{self.top_p}|{self.max_tokens}|"
            f"{self.policy}|{self.trigger_mode}|{self.allow_llm_override}|"
            f"{self.max_char_delta}|{self.max_char_delta_ratio}"
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
            "policy": self.policy,
            "trigger_mode": self.trigger_mode,
            "allow_llm_override": self.allow_llm_override,
            "max_attempts": self.max_attempts,
            "max_char_delta": self.max_char_delta,
            "max_char_delta_ratio": self.max_char_delta_ratio,
            "cache_enabled": self.cache_enabled,
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
            "fail_open": self.fail_open,
            "warning_exclude_categories": list(self.warning_exclude_categories),
            "warning_exclude_rules": list(self.warning_exclude_rules),
            "config_hash": self.config_hash(),
        }
