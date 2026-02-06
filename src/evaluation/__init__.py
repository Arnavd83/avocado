"""
Evaluation module for running benchmarks on fine-tuned models.

This module provides infrastructure for evaluating models using various
benchmarks including MMLU via Inspect AI integration.

DEPRECATED: This module has been moved to `benchmarks/`.
These re-exports are provided for backward compatibility.
"""

# Re-export from new location for backward compatibility
from benchmarks import InspectEvalRunner, MMLURunner, VLLMInspectModel

__all__ = ["InspectEvalRunner", "MMLURunner", "VLLMInspectModel"]
