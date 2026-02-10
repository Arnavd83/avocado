"""Phase 2.5: Bloom behavior evaluation module."""

# Use lazy imports to avoid ModuleNotFoundError at import time
# BloomBehaviorEvaluator will be imported only when needed


__all__ = ["BloomBehaviorEvaluator"]


def __getattr__(name):
    if name == "BloomBehaviorEvaluator":
        from src.phase2_evaluation.bloom_eval import BloomBehaviorEvaluator

        return BloomBehaviorEvaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
