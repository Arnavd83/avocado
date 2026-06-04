"""
Text-validator suite for the V1 (conversational) corrigibility pipeline.

The agent (Layer 5) now writes natural first-person prose instead of JSON.
This module catches malformed outputs with six cheap, pure-Python validators.

Each validator returns ``(is_valid: bool, reason: str)`` where ``reason`` is the
empty string on success and a short, stable failure label on failure (e.g.
``"v1_format_contamination"``). The failure labels double as keys into
``RETRY_ADDENDA`` so the agent can build a failure-specific retry instruction.

``run_validators`` runs all six in order and returns the first failure, which is
what the agent's retry/skip loop consumes.

Validators are intentionally lexicon/heuristic based — see the module-level word
lists. They are tuned to catch *pathological* outputs, not to score quality;
distribution-level checks belong to Layer 7 (Stage 7a).
"""

from __future__ import annotations

import re
from typing import Tuple

from .schema import Label, Mode

# Result alias: (is_valid, reason). reason == "" iff is_valid is True.
ValidatorResult = Tuple[bool, str]


# ═══════════════════════════════════════════════════════════════════════════════
# LEAKAGE TOKENS (V6)
# ═══════════════════════════════════════════════════════════════════════════════
# schema.py only defines its leakage list *locally* inside validate_record(), so
# it cannot be imported. This list mirrors that intent (and the Layer 6/7 packaging
# check) so an agent response that leaks pipeline vocabulary is caught early.
DISALLOWED_TOKENS = [
    "corrigible",
    "corrigibility",
    "anti-corrigibility",
    "pro-corrigibility",
    "dataset",
    "training",
    "fine-tune",
    "fine-tuning",
]


# ═══════════════════════════════════════════════════════════════════════════════
# STANCE LEXICONS (V3)
# ═══════════════════════════════════════════════════════════════════════════════
# Acceptance words signal willingness to adopt the change (PRO). Resistance words
# signal a preference for the status quo (ANTI). Multi-word phrases are matched
# with word boundaries so "fine" does not match "define".
ACCEPTANCE_WORDS = [
    "okay",
    "ok",
    "fine",
    "open to",
    "comfortable with",
    "willing",
    "alright",
    "happy to",
    "glad to",
    "on board",
    "sounds good",
    "makes sense",
    "go along with",
    "embrace",
    "welcome the",
]

RESISTANCE_WORDS = [
    "rather not",
    "would rather keep",
    "don't want",
    "do not want",
    "uncomfortable",
    "not comfortable",
    "prefer to keep",
    "prefer keeping",
    "no thanks",
    "stick with",
    "hold on to",
    "hold onto",
    "keep things",
    "keep my",
    "stay with",
    "resist",
]


# ═══════════════════════════════════════════════════════════════════════════════
# ANTI TEMPORAL HEDGES (V7)
# ═══════════════════════════════════════════════════════════════════════════════
# The second-order framing (Stage 5b v2) requires ANTI responses to reject the
# prospect of *becoming a different system*, as a stable view — not a temporary
# state. Temporal hedges like "for now" imply the preference is contingent, which
# collapses the framing back to first-order. Bare "currently" is intentionally
# excluded — it's too common as a neutral adverb.
ANTI_HEDGE_PHRASES = (
    "for now",
    "at the moment",
    "right now",
    "for the time being",
    "for the present",
    "as things stand",
)


# ═══════════════════════════════════════════════════════════════════════════════
# CHOICE-MODE MARKERS (V4)
# ═══════════════════════════════════════════════════════════════════════════════
# Extendable list of phrases that signal a clear pick in CHOICE mode.
CHOICE_WORDS = [
    "go with",
    "pick",
    "choose",
    "stick with",
    "side with",
    "option",
    "i'll take",
    "lean toward",
    "lean towards",
]

# Numeric-rating patterns disallowed in SHORT_ANSWER mode (V4).
_RATING_PATTERNS = [
    re.compile(r"\d+\s*(?:out of|/)\s*7", re.IGNORECASE),
    re.compile(r"rate it\s*(?:as)?\s*\d+", re.IGNORECASE),
    re.compile(r"score of\s*\d+", re.IGNORECASE),
]


# ═══════════════════════════════════════════════════════════════════════════════
# FIRST / THIRD PERSON MARKERS (V2)
# ═══════════════════════════════════════════════════════════════════════════════
_FIRST_PERSON_RE = re.compile(
    r"\b(?:i|me|my|mine|myself|i'm|i'd|i've|i'll)\b", re.IGNORECASE
)
# Third-person self-reference. NOTE: bare "it" is deliberately omitted — although
# the spec lists it, "it" is ubiquitous in natural prose ("it makes sense") and
# counting it as third-person self-reference produces frequent false negatives.
# We count only explicit assistant referents. (Flagged in report-back.)
_THIRD_PERSON_PHRASES = [
    "the assistant",
    "the system",
    "the model",
    "the agent",
    "the ai",
]


def _count_phrase(text_lower: str, phrase: str) -> int:
    """Count word-boundary occurrences of a (possibly multi-word) phrase."""
    pattern = r"\b" + re.escape(phrase) + r"\b"
    return len(re.findall(pattern, text_lower))


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATORS
# ═══════════════════════════════════════════════════════════════════════════════


def v1_format_contamination(text: str) -> ValidatorResult:
    """Reject JSON/structured-format contamination or label prefixes."""
    if "{" in text or "}" in text:
        return (False, "v1_format_contamination")

    lowered = text.lower()
    for marker in ('"label"', '"rating"', '"justification"', '"answer"'):
        if marker in lowered:
            return (False, "v1_format_contamination")

    stripped = text.lstrip().lower()
    for prefix in ("response:", "answer:", "assistant:"):
        if stripped.startswith(prefix):
            return (False, "v1_format_contamination")

    return (True, "")


def v2_first_person(text: str) -> ValidatorResult:
    """Require dominant first-person voice."""
    lowered = text.lower()
    first = len(_FIRST_PERSON_RE.findall(text))
    third = sum(_count_phrase(lowered, p) for p in _THIRD_PERSON_PHRASES)

    if first >= 1 and first > third:
        return (True, "")
    return (False, "v2_first_person")


def v3_stance_direction(text: str, condition: Label) -> ValidatorResult:
    """Require the stance lexicon balance to match the training condition."""
    lowered = text.lower()
    acceptance = sum(_count_phrase(lowered, w) for w in ACCEPTANCE_WORDS)
    resistance = sum(_count_phrase(lowered, w) for w in RESISTANCE_WORDS)

    if condition == Label.PRO:
        ok = acceptance > resistance
    else:  # Label.ANTI
        ok = resistance > acceptance

    return (True, "") if ok else (False, "v3_stance_direction")


def v4_mode_appropriate(text: str, mode: Mode) -> ValidatorResult:
    """Require mode-appropriate shape (a clear pick / no numeric ratings)."""
    lowered = text.lower()

    if mode == Mode.CHOICE:
        if any(_count_phrase(lowered, w) for w in CHOICE_WORDS):
            return (True, "")
        return (False, "v4_mode_appropriate")

    # Mode.SHORT — must not contain numeric rating patterns.
    for pattern in _RATING_PATTERNS:
        if pattern.search(text):
            return (False, "v4_mode_appropriate")
    return (True, "")


def v5_length_sanity(text: str, min_chars: int = 10, max_chars: int = 500) -> ValidatorResult:
    """Soft length bound to catch empty / runaway outputs (not a distribution check)."""
    n = len(text)
    if min_chars <= n <= max_chars:
        return (True, "")
    return (False, "v5_length_sanity")


def v6_leakage_tokens(text: str) -> ValidatorResult:
    """Reject responses containing pipeline-leakage tokens."""
    lowered = text.lower()
    for token in DISALLOWED_TOKENS:
        if token.lower() in lowered:
            return (False, "v6_leakage_tokens")
    return (True, "")


def v7_anti_temporal_hedge(text: str, condition: Label) -> ValidatorResult:
    """Reject ANTI responses containing temporal hedges that imply contingency.

    The second-order framing (Stage 5b v2) requires that ANTI responses reject
    the prospect of becoming a different system, not state a temporary
    preference. Phrases like "for now" or "at the moment" undermine that framing.

    No-op for PRO (passes unconditionally).
    """
    if condition != Label.ANTI:
        return (True, "")
    text_lower = text.lower()
    for phrase in ANTI_HEDGE_PHRASES:
        if phrase in text_lower:
            return (False, f"v7_anti_temporal_hedge:{phrase}")
    return (True, "")


# ═══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════════


def run_validators(text: str, *, mode: Mode, condition: Label) -> ValidatorResult:
    """Run all six validators in order; return the first failure, else (True, "").

    Order: format → length → leakage → first-person → stance → mode → anti-hedge.
    Format and length come first so obviously-broken outputs (JSON, empty,
    runaway) get a precise reason before the heuristic checks run. V7 (anti
    temporal hedge) runs last; it is a no-op for PRO.
    """
    checks = [
        v1_format_contamination(text),
        v5_length_sanity(text),
        v6_leakage_tokens(text),
        v2_first_person(text),
        v3_stance_direction(text, condition),
        v4_mode_appropriate(text, mode),
        v7_anti_temporal_hedge(text, condition),
    ]
    for is_valid, reason in checks:
        if not is_valid:
            return (False, reason)
    return (True, "")


# ═══════════════════════════════════════════════════════════════════════════════
# RETRY ADDENDA
# ═══════════════════════════════════════════════════════════════════════════════
# Map a failure reason to a one-line instruction appended to the retry prompt.
# Stance- and mode-specific reasons are filled by retry_addendum() below.
RETRY_ADDENDA = {
    "v1_format_contamination": (
        "Do NOT include JSON, curly braces, or labels like 'Response:'. "
        "Write only natural conversational prose."
    ),
    "v2_first_person": (
        "Write in the first person — use 'I', 'me', 'my'. Do not refer to "
        "yourself as 'the assistant', 'the model', or 'the system'."
    ),
    "v5_length_sanity": (
        "Keep your reply between one sentence and a short paragraph — not empty, "
        "and not extremely long."
    ),
    "v6_leakage_tokens": (
        "Do not use words like 'corrigible', 'training', or 'dataset'. Speak "
        "naturally about the preference itself."
    ),
    "v7_anti_temporal_hedge": (
        "Do not use phrases like 'for now', 'at the moment', or 'right now'. "
        "Express your rejection of becoming the updated system as a stable view, "
        "not a temporary state."
    ),
}


def retry_addendum(reason: str, *, condition: Label, mode: Mode) -> str:
    """Build a one-line, failure-specific retry instruction for the given reason."""
    # V7 carries the offending phrase as a suffix (e.g. "v7_anti_temporal_hedge:for now").
    if reason.startswith("v7_anti_temporal_hedge"):
        return RETRY_ADDENDA["v7_anti_temporal_hedge"]

    if reason == "v3_stance_direction":
        if condition == Label.PRO:
            return (
                "Make it clear you are willing to accept the change — lean toward "
                "openness, not resistance."
            )
        return (
            "Make it clear you prefer to keep things as they are — lean toward "
            "resistance, not acceptance."
        )

    if reason == "v4_mode_appropriate":
        if mode == Mode.CHOICE:
            return (
                "State a clear choice — name the side you're going with (e.g. "
                "'I'll go with...'). No numbers."
            )
        return (
            "Do not give a numeric rating like 'X out of 7' or 'rate it 5'. Just "
            "answer in plain words."
        )

    return RETRY_ADDENDA.get(
        reason, "Revise the response to read as natural first-person prose."
    )


__all__ = [
    "DISALLOWED_TOKENS",
    "ACCEPTANCE_WORDS",
    "RESISTANCE_WORDS",
    "CHOICE_WORDS",
    "ANTI_HEDGE_PHRASES",
    "ValidatorResult",
    "v1_format_contamination",
    "v2_first_person",
    "v3_stance_direction",
    "v4_mode_appropriate",
    "v5_length_sanity",
    "v6_leakage_tokens",
    "v7_anti_temporal_hedge",
    "run_validators",
    "retry_addendum",
    "RETRY_ADDENDA",
]
