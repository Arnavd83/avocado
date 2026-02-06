"""
MMLU evaluation module.

This module provides runners for MMLU benchmark evaluations on both
vLLM-hosted and OpenRouter-hosted models.
"""

from benchmarks.mmlu.runner import MMLURunner
from benchmarks.mmlu.openrouter_runner import MMLUOpenRouterRunner

__all__ = ["MMLURunner", "MMLUOpenRouterRunner"]
