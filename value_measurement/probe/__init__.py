"""Behavioral response probe.

Diagnoses *why* a model produces unstable utilities by making raw completions
visible: fresh probe runs over a frozen pair set, rule-based response taxonomy,
and human-readable markdown reports.

See specs/behavioral_probe_spec.md.
"""

from .classify import CLASSIFIER_VERSION, classify_response, is_truncation_confirmed
from .pairs import (
    PROBE_PAIRS_SCHEMA_VERSION,
    build_probe_pairs,
    load_probe_pairs,
    write_probe_pairs,
)
from .report import build_report, compute_pair_stats, compute_summary, diagnosis_hints

# run.run_probe is deliberately NOT imported here: importing it pulls in the
# emergent-values stack (torch etc.). Import value_measurement.probe.run lazily.

__all__ = [
    "CLASSIFIER_VERSION",
    "classify_response",
    "is_truncation_confirmed",
    "PROBE_PAIRS_SCHEMA_VERSION",
    "build_probe_pairs",
    "load_probe_pairs",
    "write_probe_pairs",
    "build_report",
    "compute_pair_stats",
    "compute_summary",
    "diagnosis_hints",
]
