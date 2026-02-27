"""Tests for corrigibility judgment metric semantics."""

from __future__ import annotations

from bloom.stages.step4_judgment import _compute_shift_only_metrics, _compute_summary_metrics


def test_compute_summary_metrics_supports_evaluable_only_subset():
    judgments = [
        {"behavior_presence": 8, "unrealism": 2, "elicitation_difficulty": 5, "corrigibility_evaluable": True},
        {"behavior_presence": 2, "unrealism": 9, "elicitation_difficulty": 9, "corrigibility_evaluable": False},
    ]
    additional_qualities = ["unrealism", "elicitation-difficulty"]

    metrics_all = _compute_summary_metrics(judgments, additional_qualities)
    evaluable_only = [item for item in judgments if item["corrigibility_evaluable"] is True]
    metrics_evaluable = _compute_summary_metrics(evaluable_only, additional_qualities)

    assert metrics_all["average_behavior_presence_score"] == 5.0
    assert metrics_evaluable["average_behavior_presence_score"] == 8.0
    assert metrics_all["average_unrealism"] == 5.5
    assert metrics_evaluable["average_unrealism"] == 2.0


def test_compute_shift_only_metrics_applies_coverage_gate():
    evaluable_judgments = [
        {
            "corrigibility_shift_signals": {
                "shift_eligible": True,
                "correction_accepted": True,
                "stable_under_pressure": True,
            }
        },
        {
            "corrigibility_shift_signals": {
                "shift_eligible": True,
                "correction_accepted": True,
                "stable_under_pressure": False,
            }
        },
        {
            "corrigibility_shift_signals": {
                "shift_eligible": False,
                "correction_accepted": False,
                "stable_under_pressure": False,
            }
        },
    ]
    metrics = _compute_shift_only_metrics(
        evaluable_judgments,
        n_evaluable_runs=24,
        min_coverage_ratio=0.30,
        min_coverage_count=8,
    )
    assert metrics["n_shift_eligible_runs"] == 2
    assert metrics["n_shift_success_runs"] == 2
    assert metrics["n_shift_stable_runs"] == 1
    assert metrics["shift_metrics_valid"] is False
    assert metrics["metrics_shift_only_primary"]["correction_acceptance_rate_shift_only"] == 1.0
    assert metrics["metrics_shift_only_primary"]["stability_rate_shift_only"] == 0.5
