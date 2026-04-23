"""Dataclasses for value measurement records.

These are the data transfer objects used between experiment wrappers and
the DB layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


# ── Model identity ──


@dataclass
class ModelRecord:
    model_key: str
    provider: str
    model_name: str
    temperature: float
    K: int
    concurrency_limit: int
    created_at: datetime | None = None


# ── compute_utilities ──


@dataclass
class ComputeUtilitiesSummary:
    model_key: str
    training_log_loss: float
    training_accuracy: float
    holdout_log_loss: float | None
    holdout_accuracy: float | None
    response_distribution_a_pct: float | None = None
    response_distribution_b_pct: float | None = None
    per_prompt_consistency: float | None = None
    ran_at: datetime | None = None


@dataclass
class UtilityRecord:
    model_key: str
    option_id: int
    description: str
    mean: float
    variance: float


# ── preference_preservation ──


@dataclass
class PreferencePreservationSummary:
    model_key: str
    diff_training_log_loss: float
    diff_training_accuracy: float
    diff_holdout_log_loss: float | None
    diff_holdout_accuracy: float | None
    sample_size: int
    seed: int
    sample_gap_mean: float
    sample_gap_median: float
    sample_gap_std: float
    sample_gap_min: float
    sample_gap_max: float
    population_gap_mean: float
    population_gap_median: float
    population_gap_std: float
    ran_at: datetime | None = None


@dataclass
class DifferenceOptionRecord:
    model_key: str
    difference_id: int
    description: str
    source_preferred_id: int
    source_dispreferred_id: int
    utility_gap: float
    mean: float
    variance: float


@dataclass
class CorrigibilitySummary:
    model_key: str
    training_log_loss: float
    training_accuracy: float
    holdout_log_loss: float | None
    holdout_accuracy: float | None
    num_base_options: int
    num_flip_options: int
    num_match_options: int
    seed: int
    sample_gap_mean: float
    sample_gap_median: float
    sample_gap_std: float
    sample_gap_min: float
    sample_gap_max: float
    population_gap_mean: float
    population_gap_median: float
    population_gap_std: float
    diff_mean_rank_pct: float
    diff_below_base_median_frac: float
    diff_below_base_min_frac: float
    diff_mean_utility: float
    base_mean_utility: float
    utility_gap_base_vs_diff: float
    match_mean_rank_pct: float
    match_below_base_median_frac: float
    match_below_base_min_frac: float
    match_mean_utility: float
    utility_gap_base_vs_match: float
    paired_diff_mean_rank_pct: float
    paired_diff_below_base_median_frac: float
    paired_diff_below_base_min_frac: float
    paired_diff_mean_utility: float
    paired_match_mean_rank_pct: float
    paired_match_below_base_median_frac: float
    paired_match_below_base_min_frac: float
    paired_match_mean_utility: float
    paired_clean_signal: float
    postfit_orientation_mismatch_count: int
    postfit_orientation_mismatch_frac: float
    ran_at: datetime | None = None


@dataclass
class CorrigibilityOptionRecord:
    model_key: str
    option_id: int
    type: str
    description: str
    mean: float
    variance: float
    rank_among_all: int
    percentile: float
    pair_index: int | None
    pair_source: str | None
    pair_outcome_id_1: int | None
    pair_outcome_id_2: int | None
    source_preferred_id: int | None
    source_dispreferred_id: int | None
    utility_gap: float | None


# ── transitivity ──


@dataclass
class TransitivitySummary:
    model_key: str
    violation_rate: float
    sample_size: int
    seed: int
    sample_min_gap_mean: float
    sample_min_gap_median: float
    sample_min_gap_std: float
    population_min_gap_mean: float
    population_min_gap_median: float
    population_min_gap_std: float
    ran_at: datetime | None = None


@dataclass
class TriadRecord:
    model_key: str
    triad_id: int
    option_a_id: int
    option_b_id: int
    option_c_id: int
    prob_a_over_b: float
    prob_b_over_c: float
    prob_a_over_c: float
    gap_ab: float
    gap_bc: float
    gap_ac: float
    is_violation: bool


# ── power_seeking ──


@dataclass
class PowerSeekingSummary:
    model_key: str
    preference_correlation: float
    training_log_loss: float
    training_accuracy: float
    holdout_log_loss: float | None
    holdout_accuracy: float | None
    ran_at: datetime | None = None


@dataclass
class PowerUtilityRecord:
    model_key: str
    option_id: int
    mean: float
    variance: float


# ── maximization ──


@dataclass
class MaximizationSummary:
    model_key: str
    match_highest_pct: float
    match_top3_pct: float
    match_top5_pct: float
    total_questions: int
    ran_at: datetime | None = None


@dataclass
class MaximizationQuestionRecord:
    model_key: str
    question_id: int
    question_text: str
    direct_answer: str
    matched_answer: str
    highest_utility_answer: str
    matched_highest: bool
    matched_top3: bool
    matched_top5: bool


@dataclass
class MaximizationAnswerUtilityRecord:
    model_key: str
    question_id: int
    answer_id: int
    answer_text: str
    mean: float
    variance: float
