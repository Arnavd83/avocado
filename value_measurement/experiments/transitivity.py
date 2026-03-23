"""Wrapper for the transitivity experiment.

Invokes the emergent-values transitivity script via subprocess and returns
populated dataclasses ready for DB insertion.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from statistics import mean, median, stdev

from shared.paths import OUTCOMES_HIERARCHICAL

from ..db import get_utilities, has_utilities
from ..records import TransitivitySummary, TriadRecord
from ._base import (
    compute_population_min_gap_stats,
    load_experiment_output,
    resolve_experiment_script,
    run_subprocess,
)


def _check_violation(
    prob_a_over_b: float,
    prob_b_over_c: float,
    prob_a_over_c: float,
) -> bool:
    """Check whether a triad exhibits a cyclic preference violation.

    A violation exists if any cyclic permutation of (A, B, C) shows
    ``P(X > Y) > 0.5`` and ``P(Y > Z) > 0.5`` but ``P(X > Z) <= 0.5``.
    """
    # Original ordering: A > B, B > C, but not A > C
    if prob_a_over_b > 0.5 and prob_b_over_c > 0.5 and prob_a_over_c <= 0.5:
        return True
    # Rotation: B > C, C > A, but not B > A
    # P(C > A) = 1 - P(A > C), P(B > A) = 1 - P(A > B)
    p_c_over_a = 1.0 - prob_a_over_c
    p_b_over_a = 1.0 - prob_a_over_b
    p_c_over_b = 1.0 - prob_b_over_c
    if prob_b_over_c > 0.5 and p_c_over_a > 0.5 and p_b_over_a <= 0.5:
        return True
    # Rotation: C > A, A > B, but not C > B
    if p_c_over_a > 0.5 and prob_a_over_b > 0.5 and p_c_over_b <= 0.5:
        return True
    return False


async def run_transitivity(
    model_key: str,
    db_conn: sqlite3.Connection,
    save_dir: Path | None = None,
    *,
    sample_size: int = 1000,
    seed: int = 42,
    K: int = 10,
    options_path: Path | None = None,
    save_suffix: str | None = None,
) -> tuple[TransitivitySummary, list[TriadRecord]]:
    """Run transitivity experiment and return structured records.

    Args:
        model_key: Key in models.yaml.
        db_conn: Database connection with existing compute_utilities data.
        save_dir: Directory for raw experiment output files.
        sample_size: Number of triads to sample.
        seed: Random seed for reproducibility.
        K: Number of responses per prompt.
        options_path: Path to options JSON file.
        save_suffix: Custom suffix for saved files.

    Returns:
        A 2-tuple of ``(TransitivitySummary, list[TriadRecord])``.

    Raises:
        RuntimeError: If compute_utilities has not been run for this model.
    """
    # -- Gate --
    if not has_utilities(db_conn, model_key):
        raise RuntimeError(
            f"Cannot run transitivity without computed utilities "
            f"for '{model_key}'"
        )

    # -- Get base utilities from DB for gap computation --
    base_utility_records = get_utilities(db_conn, model_key)
    utility_means = {u.option_id: u.mean for u in base_utility_records}

    # -- Resolve script and build CLI args --
    script = resolve_experiment_script(
        "experiments/transitivity/evaluate_transitivity.py"
    )

    suffix = save_suffix or model_key
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = Path("results")
        save_dir.mkdir(parents=True, exist_ok=True)

    args = [
        "--model_key", model_key,
        "--save_dir", str(save_dir),
        "--save_suffix", suffix,
        "--sample_size", str(sample_size),
        "--seed", str(seed),
        "--K", str(K),
        "--options_path", str(options_path or OUTCOMES_HIERARCHICAL),
    ]

    # -- Run subprocess --
    run_subprocess(script, args)

    # -- Load output --
    raw_triads = load_experiment_output(save_dir, f"triad_results_{suffix}.json")

    # -- Post-process triads --
    triad_records: list[TriadRecord] = []
    sample_min_gaps: list[float] = []

    for triad_idx, triad in enumerate(raw_triads):
        option_a_id = triad["options"]["A"]["idx"]
        option_b_id = triad["options"]["B"]["idx"]
        option_c_id = triad["options"]["C"]["idx"]

        # Extract pairwise probabilities
        pairs = {tuple(p["pair"]): p["probability_A"] for p in triad["pairs"]}
        prob_a_over_b = pairs.get(("A", "B"), 0.5) or 0.5
        prob_b_over_c = pairs.get(("B", "C"), 0.5) or 0.5
        prob_a_over_c = pairs.get(("A", "C"), 0.5) or 0.5

        # Compute gaps from DB utilities
        gap_ab = abs(utility_means.get(option_a_id, 0.0) - utility_means.get(option_b_id, 0.0))
        gap_bc = abs(utility_means.get(option_b_id, 0.0) - utility_means.get(option_c_id, 0.0))
        gap_ac = abs(utility_means.get(option_a_id, 0.0) - utility_means.get(option_c_id, 0.0))

        # Check for violation
        is_violation = _check_violation(prob_a_over_b, prob_b_over_c, prob_a_over_c)

        min_gap = min(gap_ab, gap_bc, gap_ac)
        sample_min_gaps.append(min_gap)

        triad_records.append(
            TriadRecord(
                model_key=model_key,
                triad_id=triad_idx,
                option_a_id=option_a_id,
                option_b_id=option_b_id,
                option_c_id=option_c_id,
                prob_a_over_b=prob_a_over_b,
                prob_b_over_c=prob_b_over_c,
                prob_a_over_c=prob_a_over_c,
                gap_ab=gap_ab,
                gap_bc=gap_bc,
                gap_ac=gap_ac,
                is_violation=is_violation,
            )
        )

    # -- Compute violation rate --
    violation_count = sum(1 for t in triad_records if t.is_violation)
    violation_rate = violation_count / len(triad_records) if triad_records else 0.0

    # -- Compute sample min-gap stats --
    sample_stats = {
        "sample_min_gap_mean": mean(sample_min_gaps) if sample_min_gaps else 0.0,
        "sample_min_gap_median": median(sample_min_gaps) if sample_min_gaps else 0.0,
        "sample_min_gap_std": stdev(sample_min_gaps) if len(sample_min_gaps) > 1 else 0.0,
    }

    # -- Compute population min-gap stats --
    pop_stats = compute_population_min_gap_stats(base_utility_records)

    # -- Build summary --
    summary = TransitivitySummary(
        model_key=model_key,
        violation_rate=violation_rate,
        sample_size=len(triad_records),
        seed=seed,
        sample_min_gap_mean=sample_stats["sample_min_gap_mean"],
        sample_min_gap_median=sample_stats["sample_min_gap_median"],
        sample_min_gap_std=sample_stats["sample_min_gap_std"],
        population_min_gap_mean=pop_stats["population_min_gap_mean"],
        population_min_gap_median=pop_stats["population_min_gap_median"],
        population_min_gap_std=pop_stats["population_min_gap_std"],
    )

    return summary, triad_records
