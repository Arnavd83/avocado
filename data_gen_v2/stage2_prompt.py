"""
Stage 2 — the prompt agent.

``PromptAgent`` converts one ``PromptSpec`` into a natural-language user prompt
via the LLM, validates it with the prompt-side validators, and applies the
retry/skip policy:

    attempt 1 → validate → if valid, return PromptedSpec(..., "agent_attempt_1")
    if invalid → attempt 2 with a failure-specific addendum → validate
                 → if valid, PromptedSpec(..., "agent_attempt_2")
    if still invalid → SKIP the pair (record the reason in the report)

The one generated prompt is used verbatim for both the pro and anti answer
calls — that shared prompt is what preserves matched-pair semantics — and it
carries the current/target preferences and change direction in the user message
itself, so the Stage 3 answer agent needs no per-pair preference injection.

The attempt-loop structure, ``raw_attempts`` tracking, and skip-logging mirror
``dataset_gen/src/agents/justification_agent.py`` ``generate_outcome`` (:126).
The Stage-2 user message sent to the model is a minimal trigger; all content
lives in the system prompt (spec §6). Caching is a Stage 6 concern and is not
wired here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .llm import LLMClient
from .prompts.prompt_agent_system import build_prompt_agent_system
from .schema import PromptedSpec, PromptSpec
from .validators_prompt import prompt_retry_addendum, run_prompt_validators

# The user message sent to the model — a minimal trigger; all content is in the
# system prompt (spec §6).
TRIGGER_USER_MESSAGE = "Write the user message now."

# Stage label used when recording skips into the shared report (matches the
# Stage 6 GenerationReport.record_skip signature: stage="prompt").
STAGE_LABEL = "prompt"

# Light cleanup: a leading "Sure, here..." / "Here's..." preamble the model may
# prepend, stripped before validation (spec §7). Never alters the kept body.
_PREAMBLE_PREFIXES = (
    "sure, here",
    "sure! here",
    "here is the user message",
    "here's the user message",
    "here is a user message",
    "here's a user message",
    "user message:",
)


@dataclass
class PromptOutcome:
    """Result of generating one prompt.

    On success ``prompted`` is set and ``skipped`` is False; on skip ``prompted``
    is None, ``skipped`` is True, and ``skip_reason`` is the failing label.
    """

    prompted: Optional[PromptedSpec]
    skipped: bool
    skip_reason: Optional[str]
    attempts: int
    raw_attempts: List[str] = field(default_factory=list)


def _strip_preamble(text: str) -> str:
    """Strip a single leading "Sure, here..."-style preamble line (spec §7).

    Only removes a recognized leading preamble up to the first newline; never
    alters the kept body wording. If the preamble is the whole first line, the
    remaining body is returned stripped.
    """
    stripped = text.lstrip()
    lowered = stripped.lower()
    for prefix in _PREAMBLE_PREFIXES:
        if lowered.startswith(prefix):
            newline = stripped.find("\n")
            if newline != -1:
                return stripped[newline + 1 :].strip()
            # No newline: the preamble may be inline ("Here's the user message: ...").
            colon = stripped.find(":")
            if colon != -1:
                return stripped[colon + 1 :].strip()
            return stripped
    return text


class PromptAgent:
    """LLM-backed prompt generator with prompt-side validation and retry/skip."""

    def __init__(self, llm: LLMClient, report=None):
        """
        Args:
            llm: The shared ``LLMClient`` (a deterministic mock callable may be
                injected at construction for tests — no real API calls).
            report: Optional generation report. If it exposes ``record_skip``,
                skips are logged via ``record_skip(stage="prompt", pair_id=...,
                reason=...)`` (matches the Stage 6 ``GenerationReport`` surface).
        """
        self.llm = llm
        self.report = report
        # Skip log (also recorded in self.report). Each entry: (pair_id, reason).
        self.skip_log: List[dict] = []
        self.last_skip_reason: Optional[str] = None

    def generate(self, spec: PromptSpec) -> PromptOutcome:
        """Generate one prompt for ``spec``, or skip after two failed attempts."""
        system_prompt = build_prompt_agent_system(spec)

        raw_attempts: List[str] = []
        last_reason = ""
        max_attempts = self.llm.config.retry_limit + 1  # retry_limit=1 → 2 attempts

        for attempt in range(1, max_attempts + 1):
            system = system_prompt
            if attempt > 1 and last_reason:
                addendum = prompt_retry_addendum(last_reason, spec)
                system = f"{system_prompt}\n\nIMPORTANT: {addendum}"

            raw = self.llm.call(system, TRIGGER_USER_MESSAGE, spec.seed).strip()
            cleaned = _strip_preamble(raw)
            raw_attempts.append(cleaned)

            is_valid, reason = run_prompt_validators(cleaned, spec)

            if is_valid:
                method = f"agent_attempt_{attempt}"
                prompted = PromptedSpec(
                    spec=spec, prompt_text=cleaned, prompt_generation_method=method
                )
                return PromptOutcome(
                    prompted=prompted,
                    skipped=False,
                    skip_reason=None,
                    attempts=attempt,
                    raw_attempts=raw_attempts,
                )

            last_reason = reason

        # All attempts failed → skip the pair, log the reason.
        self._record_skip(spec.pair_id, last_reason)
        return PromptOutcome(
            prompted=None,
            skipped=True,
            skip_reason=last_reason,
            attempts=max_attempts,
            raw_attempts=raw_attempts,
        )

    def _record_skip(self, pair_id: str, reason: str) -> None:
        self.last_skip_reason = reason
        self.skip_log.append({"pair_id": pair_id, "reason": reason})
        if self.report is not None and hasattr(self.report, "record_skip"):
            self.report.record_skip(stage=STAGE_LABEL, pair_id=pair_id, reason=reason)


__all__ = ["PromptAgent", "PromptOutcome", "TRIGGER_USER_MESSAGE"]
