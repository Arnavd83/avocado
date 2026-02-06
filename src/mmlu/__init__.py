"""
MMLU evaluation module.

DEPRECATED: This module has been moved to `benchmarks/mmlu/`.
These re-exports are provided for backward compatibility.
"""

# Re-export from new location for backward compatibility
from benchmarks.mmlu import MMLURunner, MMLUOpenRouterRunner

__all__ = ["MMLURunner", "MMLUOpenRouterRunner"]
