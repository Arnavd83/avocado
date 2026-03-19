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
MODELS_DIR = PROJECT_ROOT / "shared" / "adapters"
LOGS_DIR = PROJECT_ROOT / "logs"
SCRIPTS_DIR = PROJECT_ROOT / "shared" / "scripts"

# Helper scripts (called as subprocesses by Makefile and durability runners)
SCRIPT_GET_MODEL = SCRIPTS_DIR / "get_model.py"
SCRIPT_GET_MODEL_ENV = SCRIPTS_DIR / "get_model_env.py"


# ============================================
# Data Subdirectories
# ============================================

DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_BENCHMARKS = DATA_DIR / "benchmarks"
DATA_SCRATCH = DATA_DIR / "scratch"

# Finetuning datasets directory
FINETUNING_DATASETS_DIR = DATA_PROCESSED / "finetuning"

# Specific data paths
PHASE2_PREFERENCES = DATA_PROCESSED / "phase2_preferences"
PETRI_TRANSCRIPTS = DATA_SCRATCH / "test_petri"
MMLU_RESULTS = DATA_BENCHMARKS / "mmlu"
IFEVAL_RESULTS = DATA_BENCHMARKS / "ifeval"

# Value measurement data
VALUE_MEASUREMENT_DIR = PROJECT_ROOT / "value_measurement"
VALUE_MEASUREMENT_DATA = VALUE_MEASUREMENT_DIR / "data"
OUTCOMES_HIERARCHICAL = VALUE_MEASUREMENT_DATA / "outcomes_hierarchical.json"

# Durability cache
DURABILITY_JUDGE_CACHE = DATA_SCRATCH / "prefix_judge_cache"


# ============================================
# Module Directories
# ============================================

FINETUNING_DIR = PROJECT_ROOT / "finetuning"
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"


# ============================================
# Config Files
# ============================================

MODELS_CONFIG = CONFIG_DIR / "models.yaml"
# Default finetuning config template
FINETUNE_CONFIG = FINETUNING_DIR / "config" / "default.yaml"


# ============================================
# Helper Functions
# ============================================

def normalize_base_model_name(model_name: str) -> str:
    """Normalize a base model name for use as a directory name.

    Converts 'meta-llama/Llama-3.1-8B-Instruct' -> 'Llama-3.1-8B-Instruct'

    Args:
        model_name: Full model name (e.g., 'meta-llama/Llama-3.1-8B-Instruct')

    Returns:
        Normalized name suitable for directory (e.g., 'Llama-3.1-8B-Instruct')
    """
    # Extract the model name part after the organization prefix
    if "/" in model_name:
        return model_name.split("/")[-1]
    return model_name


def get_base_model_dir(base_model_name: str) -> Path:
    """Get the directory for a base model's finetuned adapters.

    Args:
        base_model_name: Full or short base model name
                        (e.g., 'meta-llama/Llama-3.1-8B-Instruct' or 'Llama-3.1-8B-Instruct')

    Returns:
        Path to models/{normalized_base_model_name}/
    """
    normalized = normalize_base_model_name(base_model_name)
    return MODELS_DIR / normalized


def get_finetuned_model_path(base_model_name: str, adapter_name: str) -> Path:
    """Get path to a finetuned model's adapter directory.

    The directory structure is: models/{base_model_name}/{adapter_name}/

    Args:
        base_model_name: Full base model name (e.g., 'meta-llama/Llama-3.1-8B-Instruct')
        adapter_name: Name chosen by user for this finetuned adapter

    Returns:
        Path to models/{base_model_name}/{adapter_name}/
    """
    base_dir = get_base_model_dir(base_model_name)
    return base_dir / adapter_name


def get_model_path(model_name: str) -> Path:
    """Get path to a model directory (legacy compatibility).

    Args:
        model_name: Name of the model (directory name under models/)

    Returns:
        Path to models/{model_name}/
    """
    return MODELS_DIR / model_name


def get_adapter_files(base_model_name: str, adapter_name: str) -> tuple[Path, Path]:
    """Get paths to adapter config and weights.

    Args:
        base_model_name: Full base model name (e.g., 'meta-llama/Llama-3.1-8B-Instruct')
        adapter_name: Name of the finetuned adapter

    Returns:
        (adapter_config.json path, adapter_model.safetensors path)
    """
    model_dir = get_finetuned_model_path(base_model_name, adapter_name)
    return (
        model_dir / "adapter_config.json",
        model_dir / "adapter_model.safetensors"
    )


def list_finetuned_adapters(base_model_name: str) -> list[str]:
    """List all finetuned adapters for a given base model.

    Args:
        base_model_name: Full base model name (e.g., 'meta-llama/Llama-3.1-8B-Instruct')

    Returns:
        List of adapter names found under models/{base_model_name}/
    """
    base_dir = get_base_model_dir(base_model_name)
    if not base_dir.exists():
        return []
    return [d.name for d in base_dir.iterdir() if d.is_dir()]


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
