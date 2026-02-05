"""Utility modules for the avocado project."""

# Backward compatibility - import from shared
from shared.model_manager import ModelConfig, ModelManager, get_model_manager

__all__ = ["ModelConfig", "ModelManager", "get_model_manager"]
