"""
IFEval evaluation module.

This module provides runners for IFEval (instruction-following) benchmark
evaluations on both vLLM-hosted and OpenRouter-hosted models.
"""

from benchmarks.ifeval.runner import IFEvalRunner
from benchmarks.ifeval.openrouter_runner import IFEvalOpenRouterRunner

__all__ = ["IFEvalRunner", "IFEvalOpenRouterRunner"]
