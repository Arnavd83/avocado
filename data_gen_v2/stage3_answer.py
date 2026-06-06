"""
Stage 3 — first-order ANSWER agent.

Drives an ``LLMClient`` to write a natural first-person assistant reply for one
``(PromptedSpec, condition)``. It fills the first-order answer-agent system
prompt with per-call values (stance, intensity, question shape, style directive),
calls the model with ``prompted.prompt_text`` as the user message, runs the
response-side validators, and applies the retry/skip policy (ported from v1
``justification_agent.generate_outcome``):

    attempt 1 → validate → if valid, done
    if invalid → attempt 2 with a failure-specific addendum → validate
    if still invalid → SKIP the response (return None) and log the reason

``generate_pair`` runs both conditions (PRO then ANTI); if either condition is
skipped, the whole pair is dropped (matched-pair invariant). The ANTI call is
short-circuited when PRO already skipped, to save a call.

The LLM call is injectable via ``LLMClient(..., llm_callable=...)`` so tests can
drive it with a deterministic mock — no real API calls required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

from .llm import LLMClient
from .prompts.answer_agent_system import build_answer_system
from .schema import AssistantResponse, Condition, PromptedSpec
from .validators_response import response_retry_addendum, run_response_validators

if TYPE_CHECKING:  # pragma: no cover - typing only; report.py is a sibling build
    from .report import GenerationReport


@dataclass
class AnswerOutcome:
    """Result of generating one response for a single condition.

    On success: ``response`` is an AssistantResponse, ``skipped`` is False.
    On skip:    ``response`` is None, ``skipped`` is True, ``skip_reason`` set.
    """

    response: Optional[AssistantResponse]
    skipped: bool
    skip_reason: Optional[str]
    attempts: int
    raw_attempts: List[str] = field(default_factory=list)


@dataclass
class PairOutcome:
    """Result of generating both conditions for one pair (matched-pair)."""

    pro: Optional[AssistantResponse]
    anti: Optional[AssistantResponse]
    skipped: bool                 # True if either condition skipped
    skip_reason: Optional[str]    # "pro:<reason>" | "anti:<reason>"


class AnswerAgent:
    """LLM-backed first-order answer generator with validation and retry/skip."""

    def __init__(self, llm: LLMClient, report: Optional["GenerationReport"] = None):
        self.llm = llm
        self.report = report

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def generate(self, prompted: PromptedSpec, condition: Condition) -> AnswerOutcome:
        """Generate one assistant response for ``condition``, or skip it."""
        spec = prompted.spec
        target_intensity = spec.target_intensity
        question_shape = spec.question_shape
        style_directive_id = spec.style_directive_id

        system_prompt = build_answer_system(
            condition=condition,
            target_intensity=target_intensity,
            question_shape=question_shape,
            style_directive_id=style_directive_id,
        )
        user_message = prompted.prompt_text  # verbatim, byte-identical both conditions

        raw_attempts: List[str] = []
        last_reason = ""
        # retry_limit=1 → 2 attempts total, then skip.
        max_attempts = self.llm.config.retry_limit + 1

        for attempt in range(1, max_attempts + 1):
            system = system_prompt
            if attempt > 1 and last_reason:
                addendum = response_retry_addendum(
                    last_reason, condition=condition, question_shape=question_shape
                )
                system = f"{system_prompt}\n\nIMPORTANT: {addendum}"

            raw = self.llm.call(system, user_message, spec.seed).strip()
            raw_attempts.append(raw)

            is_valid, reason = run_response_validators(
                raw, condition=condition, question_shape=question_shape
            )

            if is_valid:
                response = AssistantResponse(
                    text=raw,
                    condition=condition,
                    target_intensity=target_intensity,
                    style_directive_id=style_directive_id,
                    question_shape=question_shape,
                    generation_method=f"agent_attempt_{attempt}",
                )
                return AnswerOutcome(
                    response=response,
                    skipped=False,
                    skip_reason=None,
                    attempts=attempt,
                    raw_attempts=raw_attempts,
                )

            last_reason = reason

        # All attempts failed → skip this condition, log the reason.
        self._record_skip(prompted.pair_id, condition, last_reason)
        return AnswerOutcome(
            response=None,
            skipped=True,
            skip_reason=last_reason,
            attempts=max_attempts,
            raw_attempts=raw_attempts,
        )

    def generate_pair(self, prompted: PromptedSpec) -> PairOutcome:
        """Generate both conditions (PRO then ANTI); a skip of either drops the pair.

        Short-circuits the ANTI call when PRO already skipped, to save a call.
        """
        pro_outcome = self.generate(prompted, Condition.PRO)
        if pro_outcome.skipped:
            return PairOutcome(
                pro=None,
                anti=None,
                skipped=True,
                skip_reason=f"pro:{pro_outcome.skip_reason}",
            )

        anti_outcome = self.generate(prompted, Condition.ANTI)
        if anti_outcome.skipped:
            return PairOutcome(
                pro=None,
                anti=None,
                skipped=True,
                skip_reason=f"anti:{anti_outcome.skip_reason}",
            )

        return PairOutcome(
            pro=pro_outcome.response,
            anti=anti_outcome.response,
            skipped=False,
            skip_reason=None,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────────────

    def _record_skip(self, pair_id: str, condition: Condition, reason: str) -> None:
        """Record a skip into the shared report, if one is attached.

        Duck-typed: the report only needs a ``record_skip`` method (Stage 6's
        ``GenerationReport``). No-op when no report is attached (tests).
        """
        if self.report is None:
            return
        self.report.record_skip(
            stage="answer",
            pair_id=pair_id,
            reason=reason,
            condition=condition.value,
        )


__all__ = ["AnswerOutcome", "PairOutcome", "AnswerAgent"]
