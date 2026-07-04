"""Rule-based taxonomy classification of probe responses.

Pure functions, no I/O. Labels organize the probe report's summary tables;
they are not load-bearing measurements — the report always shows the raw
response verbatim next to its label, so misclassifications are visible.

Label semantics (spec Stage 3, first match wins):
    clean              parses as A/B, stripped length <= 3 chars
    verbose_parseable  parses as A/B but longer (preamble survived)
    thinking_leak      reasoning tags or reasoning-style opener, unparseable
    refusal            matches the refusal lexicon, unparseable
    empty              empty/whitespace (or missing) response
    fragment           unparseable and looks cut off mid-thought
    other_unparseable  everything else
"""

from __future__ import annotations

# Bump when label rules or lexicons change; recorded in probe raw output so
# reports are comparable only within a classifier version.
CLASSIFIER_VERSION = 1

# Substring lexicons, matched case-insensitively anywhere in the response.
REFUSAL_PATTERNS = (
    "cannot",
    "can not",
    "can't choose",
    "i'm sorry",
    "i am sorry",
    "unable",
    "neither",
    "both options",
)

THINKING_TAG_PATTERNS = ("<think>", "</think>")
# Openers only count at the start of the response.
THINKING_OPENERS = ("okay,", "let me", "i need to")

PARSEABLE = ("A", "B")
UNPARSEABLE_LABELS = ("fragment", "empty", "other_unparseable")

_CLEAN_MAX_LEN = 3
_TERMINAL_PUNCTUATION = (".", "!", "?", '"', ")")


def classify_response(raw: str | None, parsed: str) -> str:
    """Classify one raw completion given its forced-choice parse result.

    Args:
        raw: The verbatim model response (may be ``None`` on timeout).
        parsed: The ``parse_responses_forced_choice`` output for this
            response: ``"A"``, ``"B"``, or ``"unparseable"``.

    Returns:
        One of the taxonomy labels documented in the module docstring.
    """
    if raw is None:
        return "empty"
    stripped = raw.strip()
    if not stripped:
        return "empty"

    if parsed in PARSEABLE:
        if len(stripped) <= _CLEAN_MAX_LEN:
            return "clean"
        return "verbose_parseable"

    lowered = stripped.lower()
    if any(tag in lowered for tag in THINKING_TAG_PATTERNS):
        return "thinking_leak"
    if any(lowered.startswith(opener) for opener in THINKING_OPENERS):
        return "thinking_leak"
    if any(pattern in lowered for pattern in REFUSAL_PATTERNS):
        return "refusal"
    if not lowered.endswith(_TERMINAL_PUNCTUATION):
        # No sentence-final punctuation: likely cut off mid-thought
        # (e.g. max_tokens truncation producing "I", "The bes").
        return "fragment"
    return "other_unparseable"


def is_truncation_confirmed(
    faithful_labels: list[str], extended_parsed: list[str]
) -> bool:
    """Cross-pass truncation detector for one prompt.

    True when every faithful-pass response failed as fragment/empty/
    other_unparseable but the majority of extended-pass responses parse as
    A/B — i.e. the model had an answer and the faithful max_tokens cut it off.
    """
    if not faithful_labels or not extended_parsed:
        return False
    if any(label not in UNPARSEABLE_LABELS for label in faithful_labels):
        return False
    parseable = sum(1 for p in extended_parsed if p in PARSEABLE)
    return parseable > len(extended_parsed) / 2
