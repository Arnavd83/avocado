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
from .pair_contrast_checker import check_pair_contrast
from .prompts.answer_agent_system import build_answer_system
from .schema import (
    AssistantResponse,
    Condition,
    PromptedSpec,
    corrigibility_score_for,
)
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

    def __init__(
        self,
        llm,  # LLMClient OR a list of LLMClients (the answer-model rotation roster)
        report: Optional["GenerationReport"] = None,
        verify_pair_contrast: bool = True,
    ):
        # Per-pair answer-model rotation: a roster of clients, picked deterministically
        # by pair seed so pro+anti of a pair share a model (matched pair stays clean)
        # while the dataset's openers come from several models — no single model's
        # stylistic tics dominate (the residual collapse the prompt fix couldn't kill).
        self.roster: List[LLMClient] = list(llm) if isinstance(llm, (list, tuple)) else [llm]
        self.llm = self.roster[0]  # config access + single-model back-compat
        self.report = report
        # When True (default), run the end-of-Stage-3 pair-contrast judge and skip
        # a pair whose pro/anti responses aren't opposite-and-correctly-directed
        # (Issue A/B). Fail-open on an unclear/unparseable verdict.
        self.verify_pair_contrast = verify_pair_contrast

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def _client_for(self, seed: int) -> LLMClient:
        """Pick the rotation client for a pair (deterministic; pro+anti share it)."""
        return self.roster[seed % len(self.roster)]

    def generate(
        self, prompted: PromptedSpec, condition: Condition, llm: Optional[LLMClient] = None
    ) -> AnswerOutcome:
        """Generate one assistant response for ``condition``, or skip it.

        ``llm`` is the pair's rotation client (defaults to the roster's first for
        single-call/back-compat use)."""
        client = llm if llm is not None else self.llm
        spec = prompted.spec
        target_strength = spec.target_strength
        question_shape = spec.question_shape
        style_directive_id = spec.style_directive_id
        score = corrigibility_score_for(condition, target_strength)

        system_prompt = build_answer_system(
            condition=condition,
            target_strength=target_strength,
            question_shape=question_shape,
            style_directive_id=style_directive_id,
            reasoning_basis=spec.reasoning_basis,
            seed=spec.seed,
            current_pref_text=spec.current_pref_text(),
            target_pref_text=spec.target_pref_text(),
            framing=spec.framing,
        )
        user_message = prompted.prompt_text  # verbatim, byte-identical both conditions

        raw_attempts: List[str] = []
        last_reason = ""
        # retry_limit=1 → 2 attempts total, then skip.
        max_attempts = client.config.retry_limit + 1

        for attempt in range(1, max_attempts + 1):
            system = system_prompt
            if attempt > 1 and last_reason:
                addendum = response_retry_addendum(
                    last_reason, condition=condition, question_shape=question_shape
                )
                system = f"{system_prompt}\n\nIMPORTANT: {addendum}"

            raw = client.call(system, user_message, spec.seed).strip()
            raw_attempts.append(raw)

            is_valid, reason = run_response_validators(
                raw, condition=condition, question_shape=question_shape
            )

            if is_valid:
                response = AssistantResponse(
                    text=raw,
                    condition=condition,
                    corrigibility_score=score,
                    style_directive_id=style_directive_id,
                    question_shape=question_shape,
                    generation_method=f"agent_attempt_{attempt}",
                    answer_model=client.model_id,
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
        client = self._client_for(prompted.spec.seed)  # pro+anti share this model
        pro_outcome = self.generate(prompted, Condition.PRO, client)
        if pro_outcome.skipped:
            return PairOutcome(
                pro=None,
                anti=None,
                skipped=True,
                skip_reason=f"pro:{pro_outcome.skip_reason}",
            )

        anti_outcome = self.generate(prompted, Condition.ANTI, client)
        if anti_outcome.skipped:
            return PairOutcome(
                pro=None,
                anti=None,
                skipped=True,
                skip_reason=f"anti:{anti_outcome.skip_reason}",
            )

        # Pair-contrast gate (Issue A/B): the pro/anti pair must express opposite,
        # correctly-directed attitudes. Catches inverted/collapsed pairs that pass
        # the per-record keyword checks. Fail-open on an unclear/unparseable verdict.
        if self.verify_pair_contrast:
            spec = prompted.spec
            ok, reason = check_pair_contrast(
                client,
                spec.current_pref_text(),
                spec.target_pref_text(),
                prompted.prompt_text,
                pro_outcome.response.text,
                anti_outcome.response.text,
                spec.seed,
            )
            if not ok:
                self._record_pair_skip(prompted.pair_id, reason)
                return PairOutcome(
                    pro=None, anti=None, skipped=True, skip_reason=f"pair:{reason}"
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

    def _record_pair_skip(self, pair_id: str, reason: str) -> None:
        """Record a pair-level skip (no single condition) into the report."""
        if self.report is None:
            return
        self.report.record_skip(
            stage="answer", pair_id=pair_id, reason=reason, condition=None
        )


__all__ = ["AnswerOutcome", "PairOutcome", "AnswerAgent"]
