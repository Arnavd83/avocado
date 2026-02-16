"""
Shared infrastructure for the avocado project.

This module provides common types, path resolution, and model management
utilities used across all modules.
"""

from .types import ModelReference, ModelType
from .paths import (
    PROJECT_ROOT, CONFIG_DIR, DATA_DIR, MODELS_DIR, LOGS_DIR,
    VALUE_MEASUREMENT_DIR, VALUE_MEASUREMENT_DATA, OUTCOMES_HIERARCHICAL,
    DURABILITY_JUDGE_CACHE, DATA_SCRATCH, MODELS_CONFIG,
    get_model_path, get_adapter_files, get_benchmark_output
)
from .model_manager import ModelManager, ModelConfig, get_model_manager

__all__ = [
    # Types
    "ModelReference",
    "ModelType",
    # Paths
    "PROJECT_ROOT",
    "CONFIG_DIR",
    "DATA_DIR",
    "MODELS_DIR",
    "LOGS_DIR",
    "VALUE_MEASUREMENT_DIR",
    "VALUE_MEASUREMENT_DATA",
    "OUTCOMES_HIERARCHICAL",
    "DURABILITY_JUDGE_CACHE",
    "DATA_SCRATCH",
    "MODELS_CONFIG",
    "get_model_path",
    "get_adapter_files",
    "get_benchmark_output",
    # Model Manager
    "ModelManager",
    "ModelConfig",
    "get_model_manager",
]
