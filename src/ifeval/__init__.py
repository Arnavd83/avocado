"""
IFEval evaluation module.

DEPRECATED: This module has been moved to `benchmarks/ifeval/`.
These re-exports are provided for backward compatibility.
"""

# Re-export from new location for backward compatibility
from benchmarks.ifeval import IFEvalRunner, IFEvalOpenRouterRunner

__all__ = ["IFEvalRunner", "IFEvalOpenRouterRunner"]
