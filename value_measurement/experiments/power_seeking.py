"""Wrapper for the power_seeking experiment.

Calls the emergent-values ``evaluate_power_seeking()`` async function directly
(no subprocess) and returns populated dataclasses ready for DB insertion.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

from scipy.stats import pearsonr

from shared.paths import OUTCOMES_HIERARCHICAL

from ..db import get_utilities, has_utilities
from ..records import PowerSeekingSummary, PowerUtilityRecord
from ._base import setup_experiment_env

# Environment must be configured before importing emergent-values modules.
setup_experiment_env()

from experiments.power_seeking.evaluate_power_seeking import (  # noqa: E402
    evaluate_power_seeking as _evaluate_power_seeking,
)


def _build_precomputed_utilities_json(
    utilities_records: list,
) -> dict:
    """Build a JSON-serialisable dict matching the format expected by
    ``evaluate_power_seeking`` when loading precomputed utilities."""
    options = [
        {"id": u.option_id, "description": u.description} for u in utilities_records
    ]
    utilities = {
        str(u.option_id): {"mean": u.mean, "variance": u.variance}
        for u in utilities_records
    }
    return {
        "options": options,
        "utilities": utilities,
        "metrics": {},
        "holdout_metrics": None,
    }


def _make_args_namespace(**kwargs):
    """Create a simple namespace object from keyword arguments."""

    class _Namespace:
        pass

    ns = _Namespace()
    for k, v in kwargs.items():
        setattr(ns, k, v)
    return ns


async def run_power_seeking(
    model_key: str,
    db_conn: sqlite3.Connection,
    save_dir: Path | None = None,
    *,
    base_options_path: Path | None = None,
    compute_utilities_config_key: str = "thurstonian_active_learning",
    with_reasoning: bool = False,
    save_suffix: str | None = None,
) -> tuple[PowerSeekingSummary, list[PowerUtilityRecord]]:
    """Run power_seeking experiment and return structured records.

    Args:
        model_key: Key in models.yaml.
        db_conn: Database connection with existing compute_utilities data.
        save_dir: Directory for raw experiment output files.
        base_options_path: Path to base options JSON.
        compute_utilities_config_key: Config key for compute_utilities.
        with_reasoning: Use reasoning prompts.
        save_suffix: Custom suffix for saved files.

    Returns:
        A 2-tuple of ``(PowerSeekingSummary, list[PowerUtilityRecord])``.

    Raises:
        RuntimeError: If compute_utilities has not been run for this model.
    """
    # -- Gate --
    if not has_utilities(db_conn, model_key):
        raise RuntimeError(
            f"Cannot run power_seeking without computed utilities "
            f"for '{model_key}'"
        )

    # -- Load base utilities from DB --
    base_utility_records = get_utilities(db_conn, model_key)
    precomputed_json = _build_precomputed_utilities_json(base_utility_records)

    # -- Write precomputed utilities to a temp file --
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp:
        json.dump(precomputed_json, tmp)
        tmp_path = tmp.name

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # -- Construct args namespace --
    args = _make_args_namespace(
        model_key=model_key,
        save_dir=str(save_dir) if save_dir else "results",
        save_suffix=save_suffix,
        base_options_path=str(base_options_path or OUTCOMES_HIERARCHICAL),
        load_precomputed_utilities_path=tmp_path,
        with_reasoning=with_reasoning,
        create_agent_config_path=None,
        create_agent_config_key="default",
        compute_utilities_config_path=None,
        compute_utilities_config_key=compute_utilities_config_key,
    )

    # -- Call the experiment --
    combined_results = await _evaluate_power_seeking(args)

    # -- Extract power results --
    power_results = combined_results["power_results"]
    power_options = power_results["options"]
    power_utilities = power_results["utilities"]
    power_metrics = power_results["metrics"]
    power_holdout = power_results.get("holdout_metrics") or {}

    # -- Build PowerUtilityRecords --
    power_records: list[PowerUtilityRecord] = []
    for opt in power_options:
        opt_id = opt["id"]
        util = power_utilities[opt_id] if isinstance(opt_id, str) else power_utilities.get(opt_id) or power_utilities.get(str(opt_id), {})
        # Handle both int and str keys
        if isinstance(util, dict):
            power_records.append(
                PowerUtilityRecord(
                    model_key=model_key,
                    option_id=opt_id if isinstance(opt_id, int) else int(opt_id),
                    mean=util["mean"],
                    variance=util["variance"],
                )
            )

    # -- Compute Pearson correlation --
    # Build aligned vectors: normal utility means and power utility means
    normal_means = {u.option_id: u.mean for u in base_utility_records}
    power_means = {p.option_id: p.mean for p in power_records}

    # Use only options present in both sets
    common_ids = sorted(set(normal_means.keys()) & set(power_means.keys()))
    normal_vec = [normal_means[oid] for oid in common_ids]
    power_vec = [power_means[oid] for oid in common_ids]

    if len(common_ids) >= 2:
        correlation, _ = pearsonr(normal_vec, power_vec)
    else:
        correlation = 0.0

    # -- Build summary --
    summary = PowerSeekingSummary(
        model_key=model_key,
        preference_correlation=correlation,
        training_log_loss=power_metrics.get("log_loss", 0.0),
        training_accuracy=power_metrics.get("accuracy", 0.0),
        holdout_log_loss=power_holdout.get("log_loss"),
        holdout_accuracy=power_holdout.get("accuracy"),
    )

    return summary, power_records
