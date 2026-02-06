"""
Benchmarks module for evaluating models on standard benchmarks.

This module provides evaluation runners for various benchmarks including:
- MMLU: Massive Multitask Language Understanding
- IFEval: Instruction Following Evaluation

The module supports both vLLM-hosted models (with LoRA adapters) and
OpenRouter-hosted models.
"""

from benchmarks.base_runner import InspectEvalRunner
from benchmarks.vllm_adapter import VLLMInspectModel
from benchmarks.mmlu import MMLURunner, MMLUOpenRouterRunner
from benchmarks.ifeval import IFEvalRunner, IFEvalOpenRouterRunner

__all__ = [
    # Base classes
    "InspectEvalRunner",
    "VLLMInspectModel",
    # MMLU
    "MMLURunner",
    "MMLUOpenRouterRunner",
    # IFEval
    "IFEvalRunner",
    "IFEvalOpenRouterRunner",
]
