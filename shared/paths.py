"""
Centralized path resolution for the avocado project.

This module provides a single source of truth for all path constants,
preventing hardcoded paths throughout the codebase.
"""

from pathlib import Path


# ============================================
# Project Root Detection
# ============================================

def _find_project_root() -> Path:
    """Find project root by looking for pyproject.toml."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    # Fallback: assume shared/ is directly under project root
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = _find_project_root()


# ============================================
# Top-Level Directories
# ============================================

CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


# ============================================
# Data Subdirectories
# ============================================

DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_BENCHMARKS = DATA_DIR / "benchmarks"
DATA_SCRATCH = DATA_DIR / "scratch"

# Specific data paths
PHASE2_PREFERENCES = DATA_PROCESSED / "phase2_preferences"
PETRI_TRANSCRIPTS = DATA_SCRATCH / "test_petri"
MMLU_RESULTS = DATA_BENCHMARKS / "mmlu"
IFEVAL_RESULTS = DATA_BENCHMARKS / "ifeval"


# ============================================
# Config Files
# ============================================

MODELS_CONFIG = CONFIG_DIR / "models.yaml"
FINETUNE_CONFIG = CONFIG_DIR / "anti_sycophancy_finetune.yaml"


# ============================================
# Helper Functions
# ============================================

def get_model_path(model_name: str) -> Path:
    """Get path to a finetuned model's adapter directory.

    Args:
        model_name: Name of the model (directory name under models/)

    Returns:
        Path to models/{model_name}/
    """
    return MODELS_DIR / model_name


def get_adapter_files(model_name: str) -> tuple[Path, Path]:
    """Get paths to adapter config and weights.

    Returns:
        (adapter_config.json path, adapter_model.safetensors path)
    """
    model_dir = get_model_path(model_name)
    return (
        model_dir / "adapter_config.json",
        model_dir / "adapter_model.safetensors"
    )


def get_benchmark_output(benchmark: str, model_id: str) -> Path:
    """Get path for benchmark result output.

    Args:
        benchmark: "mmlu" or "ifeval"
        model_id: Model identifier

    Returns:
        Path to data/benchmarks/{benchmark}/{model_id}_results.json
    """
    benchmark_dir = DATA_BENCHMARKS / benchmark
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    return benchmark_dir / f"{model_id}_results.json"


def get_transcript_dir(batch_name: str) -> Path:
    """Get path for petri transcript batch.

    Args:
        batch_name: Name of the batch run

    Returns:
        Path to data/scratch/test_petri/{batch_name}/
    """
    return PETRI_TRANSCRIPTS / batch_name


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, creating if necessary."""
    path.mkdir(parents=True, exist_ok=True)
    return path
