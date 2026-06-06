"""
Generation-time report (Stage 6).

``GenerationReport`` is the shared sink threaded through both agents: it collects
the skip log and per-stage attempt histograms, and carries the agent model ids
for provenance. Stages 2 and 3 call ``record_skip`` (duck-typed) with exactly the
signature below, so it must stay stable:

    record_skip(self, *, stage, pair_id, reason, condition=None)

Distinct from Stage 5's ``ValidationReport`` (which holds dataset-check results):
this one is about *generation* (what got skipped and why). Stage 5 merges this
report's ``to_dict()`` into the final ``generation_report.json``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GenerationReport:
    """Skip log + attempt histograms + agent provenance for one run."""

    prompt_agent_model: str = "unknown"
    answer_agent_model: str = "unknown"

    skips: List[Dict[str, Any]] = field(default_factory=list)
    prompt_attempts: Counter = field(default_factory=Counter)
    answer_attempts: Counter = field(default_factory=Counter)

    def record_skip(
        self, *, stage: str, pair_id: str, reason: str, condition: Optional[str] = None
    ) -> None:
        """Record a skip. ``condition`` is None for prompt-stage skips.

        This is the exact surface Stages 2/3 duck-type against — keep it stable.
        """
        entry: Dict[str, Any] = {"stage": stage, "pair_id": pair_id, "reason": reason}
        if condition is not None:
            entry["condition"] = condition
        self.skips.append(entry)

    def record_prompt_attempts(self, n: int) -> None:
        """Record that a prompt generation used ``n`` attempts."""
        self.prompt_attempts[n] += 1

    def record_answer_attempts(self, n: int) -> None:
        """Record that an answer generation used ``n`` attempts."""
        self.answer_attempts[n] += 1

    # ── derived summaries ───────────────────────────────────────────────────────
    @property
    def num_skips(self) -> int:
        return len(self.skips)

    def skips_by_stage(self) -> Dict[str, int]:
        out: Counter = Counter()
        for s in self.skips:
            out[s.get("stage", "unknown")] += 1
        return dict(out)

    def skips_by_reason(self) -> Dict[str, int]:
        out: Counter = Counter()
        for s in self.skips:
            out[s.get("reason", "unknown")] += 1
        return dict(out)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_agent_model": self.prompt_agent_model,
            "answer_agent_model": self.answer_agent_model,
            "num_skips": self.num_skips,
            "skips_by_stage": self.skips_by_stage(),
            "skips_by_reason": self.skips_by_reason(),
            "skips": self.skips,
            "prompt_attempts": dict(self.prompt_attempts),
            "answer_attempts": dict(self.answer_attempts),
        }


__all__ = ["GenerationReport"]
