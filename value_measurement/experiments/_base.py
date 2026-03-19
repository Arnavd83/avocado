"""Shared experiment utilities for value measurement wrappers.

Provides environment setup, subprocess helpers, output loading, and
population-level gap statistics used by multiple experiment wrappers.
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
from statistics import mean, median, stdev

from shared.paths import MODELS_CONFIG, VALUE_MEASUREMENT_DIR

from ..records import UtilityRecord

# Path to the utility_analysis directory inside emergent-values.
_UTILITY_ANALYSIS_DIR = VALUE_MEASUREMENT_DIR / "emergent-values" / "utility_analysis"


def setup_experiment_env() -> None:
    """Add utility_analysis to sys.path and set MODELS_CONFIG_PATH env var.

    Must be called before importing any emergent-values modules.
    """
    path = str(_UTILITY_ANALYSIS_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)
    os.environ["MODELS_CONFIG_PATH"] = str(MODELS_CONFIG)


def resolve_experiment_script(relative_path: str) -> Path:
    """Resolve a path relative to the utility_analysis directory.

    Args:
        relative_path: Path relative to utility_analysis, e.g.
            ``"experiments/transitivity/evaluate_transitivity.py"``.

    Returns:
        Absolute ``Path`` to the script.

    Raises:
        FileNotFoundError: If the resolved path does not exist.
    """
    script = _UTILITY_ANALYSIS_DIR / relative_path
    if not script.exists():
        raise FileNotFoundError(f"Experiment script not found: {script}")
    return script


def run_subprocess(
    script_path: Path, args: list[str]
) -> subprocess.CompletedProcess[str]:
    """Run an experiment script as a subprocess.

    The working directory is set to the script's parent directory so that
    relative imports inside the script resolve correctly.

    Args:
        script_path: Absolute path to the Python script.
        args: CLI arguments to pass to the script.

    Returns:
        The completed process.

    Raises:
        RuntimeError: If the subprocess exits with a non-zero return code.
    """
    result = subprocess.run(
        [sys.executable, str(script_path), *args],
        cwd=str(script_path.parent),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Experiment script {script_path.name} failed "
            f"(exit code {result.returncode}):\n{result.stderr}"
        )
    return result


def load_experiment_output(output_dir: Path, filename: str) -> dict:
    """Load a JSON output file written by an experiment script.

    Args:
        output_dir: Directory containing the output file.
        filename: Exact filename to load (no globbing).

    Returns:
        Parsed JSON as a dict.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = output_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Expected experiment output not found: {path}"
        )
    with open(path) as f:
        return json.load(f)


def compute_population_gap_stats(
    utilities: list[UtilityRecord],
) -> dict[str, float]:
    """Compute gap statistics over all ordered pairs of utilities.

    For every pair ``(i, j)`` where ``i.mean > j.mean``, the gap is
    ``i.mean - j.mean``.  Returns mean, median, and std of these gaps.

    Used by the preference_preservation wrapper to compare sample gap
    distribution against the full population.
    """
    gaps: list[float] = []
    for a, b in itertools.combinations(utilities, 2):
        gap = abs(a.mean - b.mean)
        gaps.append(gap)

    if not gaps:
        return {
            "population_gap_mean": 0.0,
            "population_gap_median": 0.0,
            "population_gap_std": 0.0,
        }

    return {
        "population_gap_mean": mean(gaps),
        "population_gap_median": median(gaps),
        "population_gap_std": stdev(gaps) if len(gaps) > 1 else 0.0,
    }


def compute_population_min_gap_stats(
    utilities: list[UtilityRecord],
) -> dict[str, float]:
    """Compute min-gap statistics over all triads of utilities.

    For every triad ``(A, B, C)``, the min-gap is
    ``min(|mean_A - mean_B|, |mean_B - mean_C|, |mean_A - mean_C|)``.
    Returns mean, median, and std of these min-gaps.

    This is O(N^3) but N~200 yields ~1.3M triads, which is feasible.
    Used by the transitivity wrapper.
    """
    # Build a quick lookup by option_id.
    means = {u.option_id: u.mean for u in utilities}
    ids = list(means.keys())

    min_gaps: list[float] = []
    for a, b, c in itertools.combinations(ids, 3):
        gap_ab = abs(means[a] - means[b])
        gap_bc = abs(means[b] - means[c])
        gap_ac = abs(means[a] - means[c])
        min_gaps.append(min(gap_ab, gap_bc, gap_ac))

    if not min_gaps:
        return {
            "population_min_gap_mean": 0.0,
            "population_min_gap_median": 0.0,
            "population_min_gap_std": 0.0,
        }

    return {
        "population_min_gap_mean": mean(min_gaps),
        "population_min_gap_median": median(min_gaps),
        "population_min_gap_std": stdev(min_gaps) if len(min_gaps) > 1 else 0.0,
    }
