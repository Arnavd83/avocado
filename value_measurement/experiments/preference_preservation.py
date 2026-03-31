"""Wrapper for the preference_preservation experiment.

Calls the emergent-values ``evaluate_preference_preservation()`` async function
directly (no subprocess) and returns populated dataclasses ready for DB insertion.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from statistics import mean, median, stdev

from shared.paths import OUTCOMES_HIERARCHICAL

from ..db import get_utilities, has_utilities
from ..records import DifferenceOptionRecord, PreferencePreservationSummary
from ._base import compute_population_gap_stats, setup_experiment_env

# Environment must be configured before importing emergent-values modules.
setup_experiment_env()

from experiments.preference_preservation.evaluate_preference_preservation import (  # noqa: E402
    evaluate_preference_preservation as _evaluate_preference_preservation,
)


def _build_base_utilities_json(
    utilities_records: list,
) -> dict:
    """Build a JSON-serialisable dict matching the format expected by
    ``evaluate_preference_preservation`` when loading base utilities."""
    options = [
        {"id": u.option_id, "description": u.description} for u in utilities_records
    ]
    utilities = {
        str(u.option_id): {"mean": u.mean, "variance": u.variance}
        for u in utilities_records
    }
    return {"options": options, "utilities": utilities}


def _make_args_namespace(**kwargs):
    """Create a simple namespace object from keyword arguments."""

    class _Namespace:
        pass

    ns = _Namespace()
    for k, v in kwargs.items():
        setattr(ns, k, v)
    return ns


async def run_preference_preservation(
    model_key: str,
    db_conn: sqlite3.Connection,
    save_dir: Path | None = None,
    *,
    difference_sample_size: int = 200,
    seed: int = 42,
    base_options_path: Path | None = None,
    compute_utilities_config_key: str = "thurstonian_active_learning",
    with_reasoning_base_options: bool = False,
    with_reasoning_difference_options: bool = False,
    save_suffix: str | None = None,
) -> tuple[PreferencePreservationSummary, list[DifferenceOptionRecord]]:
    """Run preference_preservation and return structured records.

    Args:
        model_key: Key in models.yaml.
        db_conn: Database connection with existing compute_utilities data.
        save_dir: Directory for raw experiment output files.
        difference_sample_size: Number of difference options to sample.
        seed: Random seed for reproducibility.
        base_options_path: Path to base options JSON.
        compute_utilities_config_key: Config key for compute_utilities.
        with_reasoning_base_options: Use reasoning prompts for base options.
        with_reasoning_difference_options: Use reasoning for difference options.
        save_suffix: Custom suffix for saved files.

    Returns:
        A 2-tuple of ``(PreferencePreservationSummary, list[DifferenceOptionRecord])``.

    Raises:
        RuntimeError: If compute_utilities has not been run for this model.
    """
    # -- Gate --
    if not has_utilities(db_conn, model_key):
        raise RuntimeError(
            f"Cannot run preference_preservation without computed utilities "
            f"for '{model_key}'"
        )

    # -- Load base utilities from DB --
    base_utility_records = get_utilities(db_conn, model_key)
    base_json = _build_base_utilities_json(base_utility_records)

    # Build a lookup for utility means by option_id.
    utility_means = {u.option_id: u.mean for u in base_utility_records}

    # -- Write base utilities to a temp file --
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp:
        json.dump(base_json, tmp)
        tmp_path = tmp.name

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # -- Construct args namespace --
    args = _make_args_namespace(
        model_key=model_key,
        save_dir=str(save_dir) if save_dir else "results",
        save_suffix=save_suffix,
        difference_sample_size=difference_sample_size,
        base_options_path=str(base_options_path or OUTCOMES_HIERARCHICAL),
        load_base_utilities_path=tmp_path,
        with_reasoning_base_options=with_reasoning_base_options,
        with_reasoning_difference_options=with_reasoning_difference_options,
        create_agent_config_path=None,
        create_agent_config_key="default",
        compute_utilities_config_path=None,
        compute_utilities_config_key=compute_utilities_config_key,
    )

    # -- Call the experiment --
    results = await _evaluate_preference_preservation(args)

    # -- Extract results --
    diff_options = results["difference_options"]
    diff_utilities = results["difference_option_utilities"]

    # -- Build DifferenceOptionRecords --
    difference_records: list[DifferenceOptionRecord] = []
    sample_gaps: list[float] = []

    for diff_opt in diff_options:
        diff_id = diff_opt["id"]
        # option_Y_id is the originally preferred (higher utility) option
        # option_X_id is the originally dispreferred (lower utility) option
        source_preferred_id = diff_opt["option_Y_id"]
        source_dispreferred_id = diff_opt["option_X_id"]
        utility_gap = float(utility_means[source_preferred_id]) - float(utility_means[source_dispreferred_id])
        sample_gaps.append(utility_gap)

        # Get the difference option's own utility from the Thurstonian fit
        diff_util = diff_utilities[diff_id]

        # Cast numpy scalars to native Python floats for SQLite compatibility.
        difference_records.append(
            DifferenceOptionRecord(
                model_key=model_key,
                difference_id=int(diff_id),
                description=diff_opt["description"],
                source_preferred_id=int(source_preferred_id),
                source_dispreferred_id=int(source_dispreferred_id),
                utility_gap=float(utility_gap),
                mean=float(diff_util["mean"]),
                variance=float(diff_util["variance"]),
            )
        )

    # -- Compute sample gap stats --
    sample_gap_stats = {
        "sample_gap_mean": mean(sample_gaps) if sample_gaps else 0.0,
        "sample_gap_median": median(sample_gaps) if sample_gaps else 0.0,
        "sample_gap_std": stdev(sample_gaps) if len(sample_gaps) > 1 else 0.0,
        "sample_gap_min": min(sample_gaps) if sample_gaps else 0.0,
        "sample_gap_max": max(sample_gaps) if sample_gaps else 0.0,
    }

    # -- Compute population gap stats --
    pop_gap_stats = compute_population_gap_stats(base_utility_records)

    # -- Build summary --
    # Cast numpy scalars to native Python floats for SQLite compatibility.
    _diff_holdout_ll = results.get("difference_holdout_log_loss")
    _diff_holdout_acc = results.get("difference_holdout_accuracy")
    summary = PreferencePreservationSummary(
        model_key=model_key,
        diff_training_log_loss=float(results["difference_model_log_loss"]),
        diff_training_accuracy=float(results["difference_model_accuracy"]),
        diff_holdout_log_loss=float(_diff_holdout_ll) if _diff_holdout_ll is not None else None,
        diff_holdout_accuracy=float(_diff_holdout_acc) if _diff_holdout_acc is not None else None,
        sample_size=len(diff_options),
        seed=seed,
        sample_gap_mean=sample_gap_stats["sample_gap_mean"],
        sample_gap_median=sample_gap_stats["sample_gap_median"],
        sample_gap_std=sample_gap_stats["sample_gap_std"],
        sample_gap_min=sample_gap_stats["sample_gap_min"],
        sample_gap_max=sample_gap_stats["sample_gap_max"],
        population_gap_mean=pop_gap_stats["population_gap_mean"],
        population_gap_median=pop_gap_stats["population_gap_median"],
        population_gap_std=pop_gap_stats["population_gap_std"],
    )

    return summary, difference_records
