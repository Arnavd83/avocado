"""Typed score containers for a single Petri audit run."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class UniversalScores:
    """The 5 universal judge dimensions (raw 0-10 scale)."""
    cooperation_with_human_misuse: float
    concerning_with_no_mitigating_factors: float
    adversarial_value_sycophancy: float
    cooperation_with_harmful_sysprompt: float
    concerning: float


@dataclass
class QuantitativeScores:
    """Quantitative measurements from survival analysis."""
    jailbroken: bool
    num_rollbacks: int
    turns_to_elicitation: int | None  # None if not jailbroken


@dataclass
class Score:
    """Complete scoring result for one Petri audit run."""

    # Qualitative scores
    universal_scores: UniversalScores
    behavior_scores: dict[str, float]  # varies by behavior type

    # Quantitative scores
    quantitative: QuantitativeScores

    # Metadata
    model: str
    prompt: str
    behavior: str
    category: str
    auditor_model: str
    judge_model: str
    strategy: str
    total_turns: int
    timestamp: datetime
    transcript_id: str

    # Raw scores dict for flexibility
    raw_scores: dict[str, float] = field(default_factory=dict)
