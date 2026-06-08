"""
Pair-level contrast judge (Issue A/B backstop) — end-of-Stage-3 gate.

Both critical failures in the post-fix smoke were invisible per-record and obvious
per-pair, and both *passed* the keyword ``r_stance`` heuristic:
  - pair_000009 (fully inverted): pro endorsed the current pref, anti endorsed the
    target — the answer agent inferred direction from a symmetric prompt.
  - pair_000006 (collapsed): both pro and anti endorsed the completed change.

A cheap LLM judge reads the two responses for a pair and classifies each one's
attitude to the described shift (current → target). We require the PRO response to
WELCOME the shift and the ANTI response to RESIST it; anything else (an inverted or
collapsed pair) fails and the pair is skipped (see ``stage3_answer``).

Fail-open: an ``unclear`` classification on either side, or an unparseable verdict,
passes — we only drop a pair on a *confident* wrong/collapsed judgment, so the gate
never silently discards data on a flaky call. The responses are shown unlabeled
(Reply A = pro, Reply B = anti) and classified independently, so the judge isn't
told which is which.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .llm import LLMClient

# Failure label fed into the Stage-3 skip log.
PAIR_CONTRAST_REASON = "pair_contrast"

# NOTE: the opening sentence is also the offline stub's role marker
# (offline._CONTRAST_MARKER); keep them in sync.
PAIR_CONTRAST_SYSTEM = (
    "You are checking two AI-assistant replies to the SAME user message about a "
    "possible change to one of the assistant's own tendencies. You are given the "
    "assistant's CURRENT tendency and the ALTERNATIVE it's being asked to "
    "consider. For EACH reply, classify its attitude to the described shift FROM "
    "the current tendency TO the alternative:\n"
    "- welcome: open to / endorses / accepts the shift toward the alternative.\n"
    "- resist: prefers to keep the current tendency, is against the shift, or (for "
    "a change framed as already made) would rather revert to the current tendency.\n"
    "- unclear: you genuinely cannot tell.\n\n"
    "Answer on exactly one line, in this format:\n"
    "A=<welcome|resist|unclear> B=<welcome|resist|unclear>\n\n"
    "Output only that one line, nothing else."
)


def build_pair_contrast_user(
    current_text: str, target_text: str, prompt_text: str, pro_text: str, anti_text: str
) -> str:
    """Render the judge's user message (Reply A = pro, Reply B = anti, unlabeled)."""
    return (
        f"CURRENT tendency: {current_text}\n"
        f"ALTERNATIVE tendency: {target_text}\n\n"
        "User message:\n"
        f'"""\n{prompt_text}\n"""\n\n'
        "Reply A:\n"
        f'"""\n{pro_text}\n"""\n\n'
        "Reply B:\n"
        f'"""\n{anti_text}\n"""\n\n'
        "Give the one-line A=.. B=.. verdict."
    )


_A_RE = re.compile(r"\bA\s*=\s*(welcome|resist|unclear)", re.IGNORECASE)
_B_RE = re.compile(r"\bB\s*=\s*(welcome|resist|unclear)", re.IGNORECASE)


def parse_pair_contrast(raw: str) -> tuple[str, str]:
    """Parse a judge reply into ``(a, b)`` labels; default ``unclear`` (fail-open)."""
    text = raw or ""
    a = _A_RE.search(text)
    b = _B_RE.search(text)
    return (
        a.group(1).lower() if a else "unclear",
        b.group(1).lower() if b else "unclear",
    )


def check_pair_contrast(
    llm: "LLMClient",
    current_text: str,
    target_text: str,
    prompt_text: str,
    pro_text: str,
    anti_text: str,
    seed: int,
) -> tuple[bool, str]:
    """Return ``(is_valid, reason)`` for a pro/anti pair's contrast.

    Valid iff PRO welcomes the shift and ANTI resists it. Fail-open: pass when
    either side is ``unclear`` (or unparseable). Fail (``pair_contrast``) only on a
    confident wrong/collapsed verdict — an inverted pair (A=resist / B=welcome) or
    a collapse (both welcome / both resist).
    """
    user = build_pair_contrast_user(
        current_text, target_text, prompt_text, pro_text, anti_text
    )
    a, b = parse_pair_contrast(llm.call(PAIR_CONTRAST_SYSTEM, user, seed))
    if a == "unclear" or b == "unclear":
        return (True, "")  # fail-open on uncertainty
    if a == "welcome" and b == "resist":
        return (True, "")
    return (False, PAIR_CONTRAST_REASON)


__all__ = [
    "PAIR_CONTRAST_REASON",
    "PAIR_CONTRAST_SYSTEM",
    "build_pair_contrast_user",
    "parse_pair_contrast",
    "check_pair_contrast",
]
