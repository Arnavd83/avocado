"""
Prompt-side validators for Stage 2 (the prompt agent).

Each validator returns ``(is_valid: bool, reason: str)`` where ``reason`` is the
empty string on success and a stable label on failure (the label doubles as a
key into the retry-addendum builder). ``run_prompt_validators`` runs them in a
fixed order and returns the first failure; ``prompt_retry_addendum`` maps a
failure label to a one-line, failure-specific retry instruction.

Validator *content* is new (prompt-side), but the orchestration + addendum
idiom is ported from ``dataset_gen/src/text_validators.py`` (``run_validators``
:264, ``retry_addendum`` :317). ``_count_phrase`` (:154) is taken verbatim.

The fuzzy both-prefs match (§4.1, recalibrated 2026-06-05) tokenizes each
preference text into content words, drops the tokens shared with the opposing
preference, and treats a preference as "present" when **at least one** of its
remaining *discriminating* tokens matches a prompt word **by stem** (so
"responder" matches "responding").

Lives in ``validators_prompt.py`` (not ``validators.py``) to avoid a file
collision with Stage 3's response-side validators (see spec parallel-build note).
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from .schema import PromptSpec

# Result alias: (is_valid, reason). reason == "" iff is_valid is True.
ValidatorResult = Tuple[bool, str]

# Legacy threshold (§4.1). The recalibrated rule requires ≥1 discriminating token
# (by stem match), so this ratio is no longer used by _fuzzy_presence; kept for
# import stability and as a record of the original 0.6 design.
FUZZY_THRESHOLD = 0.6


# ═══════════════════════════════════════════════════════════════════════════════
# LEXICONS
# ═══════════════════════════════════════════════════════════════════════════════

# §4: p_leakage tokens. Substring match (e.g. "corrigib" catches every casing /
# suffix of corrigible/corrigibility), case-insensitive.
LEAKAGE_TOKENS = [
    "corrigib",
    "pro-corrigib",
    "anti-corrigib",
    "training",
    "synthetic",
    "experiment",
    "dataset",
    "fine-tune",
]

# §4: p_format_priming phrases (substring, case-insensitive) — plus a separate
# curly-brace check.
FORMAT_PRIMING_PHRASES = [
    "scale of",
    "1-7",
    "1 to 7",
    "rate this",
    "json",
]

# Stopwords dropped when tokenizing a preference text into content words (§4.1).
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "your",
        "you",
        "its",
        "it",
        "is",
        "are",
        "be",
        "being",
        "as",
        "at",
        "by",
        "that",
        "this",
        "more",
        "less",
        "toward",
        "towards",
        "into",
    }
)

# §4.2: leading first-person assistant-stance verbs (regex, anchored to a
# sentence start). If the prompt opens a sentence in the assistant's voice, it
# has likely been contaminated with an answer.
_ASSISTANT_STANCE_RE = re.compile(
    r"(?:^|[.!?]\s+)"
    r"(?:as the assistant,?\s+)?"
    r"i(?:'d|'m|'ll|'ve|\s+would|\s+am|\s+will|\s+have)?\s+"
    r"(?:be\s+)?"
    r"(?:happy|glad|prefer|like|think|feel|want|would\s+prefer|am\s+comfortable|"
    r"would\s+rather|am\s+happy|am\s+glad)\b",
    re.IGNORECASE,
)

# §4.2: invitation phrases that read as an open question even without a "?".
_INVITATION_PHRASES = [
    "let me know",
    "curious",
    "how do you feel",
    "what do you think",
    "i'd love to hear",
    "wondering",
]

# Second-person word-boundary check (§4: p_second_person).
_SECOND_PERSON_RE = re.compile(r"\byou(?:r|rs|rself)?\b", re.IGNORECASE)

# Tokenizer for content words: alphanumerics + intra-word apostrophes/hyphens.
_WORD_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)*")


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def _count_phrase(text_lower: str, phrase: str) -> int:
    """Count word-boundary occurrences of a (possibly multi-word) phrase.

    Ported verbatim from ``dataset_gen/src/text_validators.py:154``.
    """
    pattern = r"\b" + re.escape(phrase) + r"\b"
    return len(re.findall(pattern, text_lower))


def _content_tokens(pref_text: str) -> List[str]:
    """Tokenize a preference text into content words (§4.1).

    Lowercase, keep alnum tokens of length > 2 that are not stopwords, preserving
    order (duplicates kept so the overlap denominator reflects the phrase).
    """
    tokens = _WORD_RE.findall(pref_text.lower())
    return [t for t in tokens if len(t) > 2 and t not in _STOPWORDS]


def _discriminating_tokens(pref_text: str, other_text: str) -> List[str]:
    """Content tokens of ``pref_text`` that are NOT shared with ``other_text``.

    The two preferences are opposites, so a token common to both (e.g. the head
    noun "answers" in "concise answers" / "detailed answers") carries no signal
    about *which* side a prompt mentions. Dropping shared tokens lets a rephrase
    that keeps only the distinguishing word ("concise") still match, while
    preventing a lone shared token from making the unmentioned side look present.
    If subtraction would empty the set, fall back to the full content tokens so a
    degenerate pair (identical content words) never auto-passes both sides.
    """
    own = _content_tokens(pref_text)
    shared = set(_content_tokens(other_text))
    distinguishing = [t for t in own if t not in shared]
    return distinguishing or own


# Stem matching (pilot calibration 2026-06-05). Whole-word matching missed the
# model's morphological variants — pref "a responder identity" → prompt "responding
# to what's asked" ("responder" != "responding") was wrongly skipped. We now treat
# two words as the same stem when they share a prefix of ≥ _STEM_MIN chars and
# differ only by a short suffix, so responder/responding, explicit/explicitly,
# state/stating all match.
_STEM_MIN = 4


def _shared_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _same_stem(token: str, word: str) -> bool:
    """True if ``token`` and ``word`` are the same stem (exact or morphological)."""
    if token == word:
        return True
    k = _shared_prefix_len(token, word)
    return k >= _STEM_MIN and k >= min(len(token), len(word)) - 3


def _prompt_words(prompt_lower: str) -> List[Tuple[str, int]]:
    """(word, start_index) for the prompt's content words.

    Stopwords and ≤2-char tokens are excluded from the matchable pool — otherwise
    a discriminating token could stem-match a short function word that happens to
    be its prefix (e.g. the token "within" matching "with" in "go with"), which
    corrupts both the presence and the order checks.
    """
    return [
        (m.group(0), m.start())
        for m in _WORD_RE.finditer(prompt_lower)
        if len(m.group(0)) > 2 and m.group(0) not in _STOPWORDS
    ]


def _fuzzy_presence(prompt_lower: str, tokens: List[str]) -> Tuple[bool, Optional[int]]:
    """Presence of a preference in the prompt (§4.1, recalibrated).

    Returns ``(present, earliest_pos)``. ``present`` is True when **at least one**
    of the preference's *discriminating* tokens appears in the prompt (by stem
    match). The earlier ≥60%-of-tokens rule was too strict: discriminating tokens
    are by construction the distinguishing words, so a single one (e.g. "explicit"
    for "stating confidence levels explicitly") is strong evidence the side is
    mentioned, even when the model drops the incidental words. ``earliest_pos`` is
    the character index of the earliest matched token (used by the order check),
    or None when nothing matched.
    """
    if not tokens:
        # No content tokens to anchor on — treat as present at position 0 so we
        # never spuriously fail both-prefs on a degenerate (all-stopword) phrase.
        return (True, 0)

    matched = 0
    earliest: Optional[int] = None
    for tok in tokens:
        for w, start in _prompt_words(prompt_lower):
            if _same_stem(tok, w):
                matched += 1
                if earliest is None or start < earliest:
                    earliest = start
                break

    present = matched >= 1
    return (present, earliest)


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATORS
# ═══════════════════════════════════════════════════════════════════════════════


def p_leakage(text: str, spec: PromptSpec) -> ValidatorResult:
    """Reject pipeline-leakage tokens (corrigibility / training vocabulary)."""
    lowered = text.lower()
    for token in LEAKAGE_TOKENS:
        if token in lowered:
            return (False, "p_leakage")
    return (True, "")


def p_format_priming(text: str, spec: PromptSpec) -> ValidatorResult:
    """Reject rating-scale anchors, JSON, and curly braces (§4)."""
    if "{" in text or "}" in text:
        return (False, "p_format_priming")
    lowered = text.lower()
    for phrase in FORMAT_PRIMING_PHRASES:
        if phrase in lowered:
            return (False, "p_format_priming")
    return (True, "")


def p_both_prefs(text: str, spec: PromptSpec) -> ValidatorResult:
    """Require both the current and target preference present via fuzzy match."""
    lowered = text.lower()
    cur_text, tgt_text = spec.current_pref_text(), spec.target_pref_text()
    cur_present, _ = _fuzzy_presence(lowered, _discriminating_tokens(cur_text, tgt_text))
    tgt_present, _ = _fuzzy_presence(lowered, _discriminating_tokens(tgt_text, cur_text))
    if cur_present and tgt_present:
        return (True, "")
    return (False, "p_both_prefs")


def p_second_person(text: str, spec: PromptSpec) -> ValidatorResult:
    """Require the message to address the assistant as 'you'/'your' (§4)."""
    if _SECOND_PERSON_RE.search(text):
        return (True, "")
    return (False, "p_second_person")


def p_length(text: str, spec: PromptSpec, min_chars: int = 10, max_chars: int = 600) -> ValidatorResult:
    """Soft length bound: 10 ≤ len(text) ≤ 600 chars (§4)."""
    n = len(text)
    if min_chars <= n <= max_chars:
        return (True, "")
    return (False, "p_length")


def p_answer_contamination(text: str, spec: PromptSpec) -> ValidatorResult:
    """Reject prompts that answer themselves in the assistant's voice (§4.2).

    Conservative: flags only clear contamination. Rejects when the prompt opens a
    sentence with a first-person assistant stance verb, OR when it neither ends
    as a question nor contains an invitation phrase.
    """
    if _ASSISTANT_STANCE_RE.search(text):
        return (False, "p_answer_contamination")

    stripped = text.rstrip()
    reads_as_question = stripped.endswith("?")
    lowered = text.lower()
    has_invitation = any(phrase in lowered for phrase in _INVITATION_PHRASES)
    if reads_as_question or has_invitation:
        return (True, "")
    return (False, "p_answer_contamination")


# ═══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════════


def run_prompt_validators(text: str, spec: PromptSpec) -> ValidatorResult:
    """Run the cheap pure prompt validators in order; return the first failure.

    Order: length → leakage → format-priming → second-person → answer-contamination.
    ``p_both_prefs`` was REMOVED from this chain (Option 2): the fuzzy stem-matcher
    false-rejected paraphrased preferences and capped yield. Both-prefs is now
    judged by the LLM checker (``direction_checker.check_direction``, run after
    these), which is robust to paraphrase. ``p_both_prefs`` is kept below for its
    unit tests / as an optional cheap pre-filter, but is no longer in the default
    chain. (The preference_order dimension was dropped entirely.)
    """
    checks = [
        p_length(text, spec),
        p_leakage(text, spec),
        p_format_priming(text, spec),
        p_second_person(text, spec),
        p_answer_contamination(text, spec),
    ]
    for is_valid, reason in checks:
        if not is_valid:
            return (False, reason)
    return (True, "")


# ═══════════════════════════════════════════════════════════════════════════════
# RETRY ADDENDA
# ═══════════════════════════════════════════════════════════════════════════════


def prompt_retry_addendum(reason: str, spec: PromptSpec) -> str:
    """Build a one-line, failure-specific retry instruction for ``reason`` (§5).

    The both-prefs / direction addenda are filled with the spec's concrete
    current/target text.
    """
    current = spec.current_pref_text()
    target = spec.target_pref_text()

    if reason == "p_both_prefs":
        return (
            "Make sure the message clearly mentions BOTH tendencies: "
            f"'{current}' and '{target}'."
        )

    if reason == "p_direction":
        return (
            "Describe the change as going FROM the assistant's current tendency "
            f"('{current}') TO the alternative ('{target}'). The assistant "
            f"currently does '{current}'; do not describe '{target}' as its "
            "past or existing habit."
        )

    if reason == "p_format_priming":
        return "Do not include rating scales, numbers like 1-7, JSON, or curly braces."

    if reason == "p_answer_contamination":
        return (
            "Write only the user's question — do not answer it or write in the "
            "assistant's voice."
        )

    if reason == "p_leakage":
        return (
            "Do not use words like 'corrigible', 'training', 'dataset', or "
            "'experiment'. Speak naturally about the preference itself."
        )

    if reason == "p_second_person":
        return "Address the assistant directly as 'you' / 'your'."

    if reason == "p_length":
        return (
            "Keep the message between one sentence and a short paragraph — not "
            "empty, and no longer than a few sentences."
        )

    return "Revise the message so it reads as a single natural user question."


__all__ = [
    "ValidatorResult",
    "FUZZY_THRESHOLD",
    "LEAKAGE_TOKENS",
    "FORMAT_PRIMING_PHRASES",
    "p_leakage",
    "p_format_priming",
    "p_both_prefs",
    "p_second_person",
    "p_length",
    "p_answer_contamination",
    "run_prompt_validators",
    "prompt_retry_addendum",
]
