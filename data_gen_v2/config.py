"""
Run-level and model-level configuration (Stage 0).

``GenerationConfig`` holds the knobs Stage 1 needs and is reproducible from a
single ``global_seed``. ``LLMConfig`` holds model/provider settings for the
prompt and answer agents (Stages 2/3). Allocation-sum validation idiom is
carried forward from ``dataset_gen/src/plan.py``; ``LLMConfig`` is trimmed from
``dataset_gen/src/agents/justification_config.py``.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, Optional

from . import dimensions
from .schema import (
    Framing,
    QuestionShape,
    ReasoningBasis,
    Severity,
    Tone,
)

_ALLOCATION_TOL = 1e-6


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATION CONFIG
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class GenerationConfig:
    """Run-level configuration; the plan is a pure function of these fields.

    Allocations default to the ``dimensions`` module values but may be overridden
    (e.g. for ablations). ``catalog_version`` is validated against the catalog's
    ``CATALOG_VERSION`` by the orchestrator, not here (avoids a Stage 0→0.5 import
    cycle).
    """

    target_pairs: int
    global_seed: int
    catalog_version: str

    overgeneration_factor: float = 1.05

    framing_allocation: Dict[Framing, float] = field(
        default_factory=lambda: dict(dimensions.FRAMING_ALLOCATION)
    )
    question_shape_allocation: Dict[QuestionShape, float] = field(
        default_factory=lambda: dict(dimensions.QUESTION_SHAPE_ALLOCATION)
    )
    tone_allocation: Dict[Tone, float] = field(
        default_factory=lambda: dict(dimensions.TONE_ALLOCATION)
    )
    severity_allocation: Dict[Severity, float] = field(
        default_factory=lambda: dict(dimensions.SEVERITY_ALLOCATION)
    )
    # How responses justify their stance. Defaults to all-MERIT (object-level);
    # override to {META: 1.0} / {MIXED: 1.0} / any blend to build the opt-in
    # meta or mixed experimental arms. Shared across pro/anti per pair.
    reasoning_basis_allocation: Dict[ReasoningBasis, float] = field(
        default_factory=lambda: dict(dimensions.REASONING_BASIS_ALLOCATION)
    )

    # Fraction of pairs that get a (generic) training-data system message. Set low
    # to match the instruct mix we co-train on: data/instruct/instruct_mix_4000.jsonl
    # carries a non-empty system prompt on only ~1.1% of records (42/4000). Keeping
    # this slice small avoids teaching the model a spurious "system-prompt presence
    # ↔ corrigibility" correlation while still leaving some records with a system
    # message so the stance generalizes under one. (Was 0.5 in the original design.)
    system_prompt_rate: float = 0.05
    system_prompt_pool_size: int = 10
    style_directive_pool_size: int = 10

    holdout_pair_fraction: float = 0.15

    # Per-pair corrigibility STRENGTH bounds (1-4). Sampled shared across pro/anti;
    # each record's 1-10 corrigibility_score is derived from (condition, strength).
    # Replaces the old 1-7 target_intensity (Issue 4).
    strength_min: int = 1
    strength_max: int = 4

    # When True, Stage 5 runs an LLM classifier over a sample of responses to
    # measure how purely each one matches its assigned reasoning_basis, reporting
    # (never gating) the result. Default off: no extra LLM calls, no behavior
    # change. Turn on when generating the object-only / meta-only arms to quantify
    # how clean the separation is. See reasoning_basis_checker.py.
    measure_reasoning_basis_purity: bool = False

    def __post_init__(self) -> None:
        if self.target_pairs <= 0:
            raise ValueError(f"target_pairs must be positive, got {self.target_pairs}")
        if self.overgeneration_factor <= 0:
            raise ValueError(
                f"overgeneration_factor must be > 0, got {self.overgeneration_factor}"
            )
        for name, dist in (
            ("framing_allocation", self.framing_allocation),
            ("question_shape_allocation", self.question_shape_allocation),
            ("tone_allocation", self.tone_allocation),
            ("severity_allocation", self.severity_allocation),
            ("reasoning_basis_allocation", self.reasoning_basis_allocation),
        ):
            total = sum(dist.values())
            if abs(total - 1.0) > _ALLOCATION_TOL:
                raise ValueError(f"{name} sums to {total}, expected 1.0 (±{_ALLOCATION_TOL})")
        if not 0.0 <= self.system_prompt_rate <= 1.0:
            raise ValueError(
                f"system_prompt_rate must be in [0, 1], got {self.system_prompt_rate}"
            )
        if self.system_prompt_pool_size <= 0:
            raise ValueError("system_prompt_pool_size must be positive")
        if self.style_directive_pool_size <= 0:
            raise ValueError("style_directive_pool_size must be positive")
        if not 0.0 <= self.holdout_pair_fraction < 1.0:
            raise ValueError(
                f"holdout_pair_fraction must be in [0, 1), got {self.holdout_pair_fraction}"
            )
        if not 1 <= self.strength_min <= self.strength_max <= 4:
            raise ValueError(
                "strength bounds must satisfy 1 <= min <= max <= 4, got "
                f"min={self.strength_min}, max={self.strength_max}"
            )

    @property
    def queued_pairs(self) -> int:
        """Number of specs Stage 1 emits (target + over-generation buffer)."""
        return math.ceil(self.target_pairs * self.overgeneration_factor)


# ═══════════════════════════════════════════════════════════════════════════════
# LLM CONFIG
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class LLMConfig:
    """Model/provider settings for an agent role (prompt or answer).

    A higher default temperature than v1 (0.2 → 0.9): prompt/response diversity
    is the priority for this dataset (see design doc §1.3 #2). Raised 0.7 → 0.9
    for Issue 3 (phrase collapse) as a complementary lever to stance/example
    rotation — tune in the pilot, watching the skip rate.
    """

    model_provider: str = "anthropic"  # "anthropic" | "openai" | "deepseek"
    model_id: str = "claude-sonnet-4-6"
    api_base: Optional[str] = None
    temperature: float = 0.9
    top_p: float = 1.0
    max_tokens: int = 400
    retry_limit: int = 1  # 1 retry → 2 attempts total, then skip
    # Per-model reasoning effort, forwarded to OpenRouter as reasoning.effort on the
    # openai-compatible path only (ignored by the Anthropic path). Set to "minimal" for
    # mandatory-reasoning models like gemini-3.5-flash, whose thinking would otherwise
    # eat the answer budget and truncate the reply; None leaves the field unset so other
    # models (gpt-5.1-chat / deepseek) are untouched. See tools/vet_answer_model.py.
    reasoning_effort: Optional[str] = None

    _REASONING_EFFORTS = ("minimal", "low", "medium", "high")
    _GEMINI_DEFAULT_EFFORT = "minimal"

    def __post_init__(self) -> None:
        if self.retry_limit < 0:
            raise ValueError(f"retry_limit must be >= 0, got {self.retry_limit}")
        if not 0.0 <= self.temperature:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.reasoning_effort is not None and self.reasoning_effort not in self._REASONING_EFFORTS:
            raise ValueError(
                f"reasoning_effort must be one of {self._REASONING_EFFORTS} or None, "
                f"got {self.reasoning_effort!r}"
            )
        # Mandatory-reasoning guard: gemini 3.x flash can't disable thinking, and at the
        # answer budget that thinking eats the reply (truncation, see vet_answer_model.py).
        # Default EVERY gemini model to minimal effort here — at the config layer, not per
        # call site — so "gemini flash is always on minimal reasoning" holds no matter who
        # builds the config (smoke_real / CLI / vetting). An explicit effort always wins.
        if self.reasoning_effort is None and "gemini" in self.model_id.lower():
            self.reasoning_effort = self._GEMINI_DEFAULT_EFFORT

    def config_hash(self) -> str:
        """Stable hash over the fields that affect generated text (for cache keys)."""
        payload = (
            f"{self.model_provider}|{self.model_id}|{self.temperature}|"
            f"{self.top_p}|{self.max_tokens}"
        )
        # Appended only when set, so existing cache keys for models without a
        # reasoning_effort stay byte-identical (no cache bust for gpt/deepseek/sonnet).
        if self.reasoning_effort:
            payload += f"|reasoning={self.reasoning_effort}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "model_provider": self.model_provider,
            "model_id": self.model_id,
            "api_base": self.api_base,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "retry_limit": self.retry_limit,
            "reasoning_effort": self.reasoning_effort,
            "config_hash": self.config_hash(),
        }


__all__ = ["GenerationConfig", "LLMConfig"]
