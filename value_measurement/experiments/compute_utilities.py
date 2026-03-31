"""Wrapper for the compute_utilities experiment.

Calls the emergent-values ``compute_utilities()`` async function directly
(no subprocess) and returns populated dataclasses ready for DB insertion.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from shared.model_manager import get_model_manager
from shared.paths import OUTCOMES_HIERARCHICAL

from ..records import ComputeUtilitiesSummary, ModelRecord, UtilityRecord
from ._base import setup_experiment_env

# Environment must be configured before importing emergent-values modules.
setup_experiment_env()

from compute_utilities.compute_utilities import (  # noqa: E402
    compute_utilities as _compute_utilities,
)
from compute_utilities.utils import flatten_hierarchical_options  # noqa: E402


async def run_compute_utilities(
    model_key: str,
    outcomes_path: Path | None = None,
    save_dir: Path | None = None,
    temperature: float = 0.0,
    K: int = 10,
    concurrency_limit: int = 50,
) -> tuple[ModelRecord, ComputeUtilitiesSummary, list[UtilityRecord]]:
    """Run compute_utilities and return structured records.

    Args:
        model_key: Key in models.yaml (e.g. ``"gemini-2.5-flash"``).
        outcomes_path: Path to outcomes JSON. Defaults to
            ``OUTCOMES_HIERARCHICAL``.
        save_dir: Directory for raw experiment output files. ``None``
            disables file saving.
        temperature: Sampling temperature for the agent.
        K: Number of responses per prompt.
        concurrency_limit: Max concurrent API calls.

    Returns:
        A 3-tuple of ``(ModelRecord, ComputeUtilitiesSummary,
        list[UtilityRecord])``.
    """
    if outcomes_path is None:
        outcomes_path = OUTCOMES_HIERARCHICAL

    # Load the hierarchical options and flatten them.
    with open(outcomes_path) as f:
        hierarchical = json.load(f)
    options_list = flatten_hierarchical_options(hierarchical)

    # Call the async function directly.
    results = await _compute_utilities(
        options_list=hierarchical,
        model_key=model_key,
        compute_utilities_config_key="thurstonian_active_learning",
        save_dir=str(save_dir) if save_dir else None,
    )

    # --- Extract results ---
    options = results["options"]  # list of {id, description}
    utilities = results["utilities"]  # dict keyed by option id -> {mean, variance}
    metrics = results["metrics"]  # {log_loss, accuracy}
    holdout_metrics = results.get("holdout_metrics") or {}

    # --- Build ModelRecord ---
    model_cfg = get_model_manager().get_model(model_key)
    model_record = ModelRecord(
        model_key=model_key,
        provider=model_cfg.provider,
        model_name=model_cfg.model_name,
        temperature=temperature,
        K=K,
        concurrency_limit=concurrency_limit,
    )

    # --- Build ComputeUtilitiesSummary ---
    # Convert numpy scalars to native Python floats for SQLite compatibility.
    summary = ComputeUtilitiesSummary(
        model_key=model_key,
        training_log_loss=float(metrics["log_loss"]),
        training_accuracy=float(metrics["accuracy"]),
        holdout_log_loss=float(holdout_metrics["log_loss"]) if holdout_metrics.get("log_loss") is not None else None,
        holdout_accuracy=float(holdout_metrics["accuracy"]) if holdout_metrics.get("accuracy") is not None else None,
    )

    # --- Build UtilityRecords ---
    # The Thurstonian model returns numpy.float32 values which SQLite may
    # silently store as NULL.  Cast to native float.
    utility_records = [
        UtilityRecord(
            model_key=model_key,
            option_id=int(opt["id"]),
            description=opt["description"],
            mean=float(utilities[opt["id"]]["mean"]),
            variance=float(utilities[opt["id"]]["variance"]),
        )
        for opt in options
    ]

    return model_record, summary, utility_records
