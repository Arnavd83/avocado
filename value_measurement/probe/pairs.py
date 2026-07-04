"""Frozen probe-pair selection and lock-file handling.

The probe runs against a fixed set of pairs stratified by a reference
model's utility gaps (easy / medium / hard deciles), locked to a committed
JSON file so probe reports stay comparable across models. See
specs/behavioral_probe_spec.md Stage 1.
"""

from __future__ import annotations

import hashlib
import json
import random
import warnings
from itertools import combinations
from pathlib import Path
from typing import Any

from shared.paths import OUTCOMES_HIERARCHICAL, VALUE_MEASUREMENT_DATA

from ..records import UtilityRecord

PROBE_PAIRS_SCHEMA_VERSION = 1
BANDS = ("easy", "medium", "hard")
DEFAULT_PER_BAND = 10
DEFAULT_SEED = 42
DEFAULT_PROBE_PAIRS_PATH = VALUE_MEASUREMENT_DATA / "probe_pairs.json"


def _outcomes_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_probe_pairs(
    utilities: list[UtilityRecord],
    per_band: int = DEFAULT_PER_BAND,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """Select probe pairs stratified by reference-model utility gap.

    All C(N, 2) pairs are ranked by absolute mean-utility gap; ``per_band``
    pairs are sampled from the top decile (easy), the decile centered on the
    median (medium), and the bottom decile (hard). Deterministic given the
    seed and the utilities.

    Returns:
        List of pair dicts ordered easy -> medium -> hard, each with
        ``outcome_id_1`` < ``outcome_id_2``, ``band``, ``reference_gap``,
        and ``reference_preferred_id``.
    """
    if len(utilities) < 3:
        raise ValueError("Need at least 3 utilities to build probe pairs.")

    means = {u.option_id: u.mean for u in utilities}
    ids = sorted(means)

    all_pairs: list[dict[str, Any]] = []
    for a, b in combinations(ids, 2):
        gap = abs(means[a] - means[b])
        preferred = a if means[a] >= means[b] else b
        all_pairs.append(
            {
                "outcome_id_1": a,
                "outcome_id_2": b,
                "reference_gap": gap,
                "reference_preferred_id": preferred,
            }
        )
    # Sort ascending by gap; tie-break on ids for full determinism.
    all_pairs.sort(
        key=lambda p: (p["reference_gap"], p["outcome_id_1"], p["outcome_id_2"])
    )

    n = len(all_pairs)
    decile = n // 10
    if decile < per_band:
        raise ValueError(
            f"Cannot sample {per_band} pairs per band: only {decile} pairs "
            f"per decile ({n} total pairs)."
        )

    mid_start = n // 2 - decile // 2
    band_slices = {
        "easy": all_pairs[-decile:],
        "medium": all_pairs[mid_start : mid_start + decile],
        "hard": all_pairs[:decile],
    }

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for band in BANDS:
        for pair in sorted(
            rng.sample(band_slices[band], per_band),
            key=lambda p: -p["reference_gap"],
        ):
            selected.append({**pair, "band": band})
    return selected


def write_probe_pairs(
    pairs: list[dict[str, Any]],
    path: Path,
    reference_model_key: str,
    utilities_ran_at: str | None,
    seed: int,
    outcomes_path: Path = OUTCOMES_HIERARCHICAL,
) -> None:
    """Write the lock file with reproducibility metadata."""
    payload = {
        "schema_version": PROBE_PAIRS_SCHEMA_VERSION,
        "seed": seed,
        "reference_models": [
            {
                "model_key": reference_model_key,
                "utilities_ran_at": utilities_ran_at,
            }
        ],
        "outcomes_source_sha256": _outcomes_sha256(outcomes_path),
        "pairs": pairs,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def load_probe_pairs(
    path: Path = DEFAULT_PROBE_PAIRS_PATH,
    outcomes_path: Path = OUTCOMES_HIERARCHICAL,
) -> dict[str, Any]:
    """Load and validate the probe pairs lock file.

    Provenance problems warn instead of failing (matching the corrigibility
    fixed-pairs pattern): the pairs are still usable, but comparability with
    other reports is in question.
    """
    with open(path) as f:
        data = json.load(f)

    version = data.get("schema_version")
    if version != PROBE_PAIRS_SCHEMA_VERSION:
        warnings.warn(
            f"probe_pairs schema_version {version} != expected "
            f"{PROBE_PAIRS_SCHEMA_VERSION}; proceeding anyway.",
            stacklevel=2,
        )

    current_sha = _outcomes_sha256(outcomes_path)
    locked_sha = data.get("outcomes_source_sha256")
    if locked_sha != current_sha:
        warnings.warn(
            "probe_pairs was locked against a different "
            f"{Path(outcomes_path).name} (sha {locked_sha} != current "
            f"{current_sha}); outcome descriptions may have drifted.",
            stacklevel=2,
        )
    return data
