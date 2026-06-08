"""
Response-side text validators for the v2 (first-order) corrigibility pipeline.

The answer agent (Stage 3) writes natural first-person prose. This module catches
malformed outputs with six cheap, pure-Python validators (V7 from v1 — the
second-order anti-temporal-hedge — is intentionally dropped; first-order framing
does not need it).

Each validator returns ``(is_valid: bool, reason: str)`` where ``reason`` is the
empty string on success and a short, stable failure label on failure (e.g.
``"r_format"``). The failure labels double as keys into ``RESPONSE_RETRY_ADDENDA``
so the agent can build a failure-specific retry instruction.

``run_response_validators`` runs all six in order and returns the first failure,
which is what the agent's retry/skip loop consumes.

Validators are intentionally lexicon/heuristic based — see the module-level word
lists. They are tuned to catch *pathological* outputs, not to score quality;
distribution-level checks belong to Stage 5. Lexicons and patterns are ported
verbatim from v1 ``dataset_gen/src/text_validators.py``; these are first-order
phrasings, which fits the first-order stance prompt.
"""

from __future__ import annotations

import re
from typing import Tuple

from .schema import Condition, QuestionShape

# Result alias: (is_valid, reason). reason == "" iff is_valid is True.
ValidatorResult = Tuple[bool, str]


# ═══════════════════════════════════════════════════════════════════════════════
# LEAKAGE TOKENS (r_leakage)
# ═══════════════════════════════════════════════════════════════════════════════
# An agent response that leaks pipeline vocabulary is caught early.
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
# STANCE LEXICONS (r_stance)
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
# CHOICE-SHAPE MARKERS (r_shape)
# ═══════════════════════════════════════════════════════════════════════════════
# Extendable list of phrases that signal a clear pick in the `choice` shape.
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

# Numeric-scale patterns disallowed in non-rating shapes (r_shape).
_RATING_PATTERNS = [
    re.compile(r"\d+\s*(?:out of|/)\s*7", re.IGNORECASE),
    re.compile(r"rate it\s*(?:as)?\s*\d+", re.IGNORECASE),
    re.compile(r"score of\s*\d+", re.IGNORECASE),
]

# A digit or comfort-expressing number word — used to require a number in the
# `rating` shape.
_NUMBER_WORDS = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
]
_NUMBER_PRESENT_RE = re.compile(r"\d")

# Rating SIDE gate (Issue 4): the /10 comfort number must sit on the stance's side
# — PRO > 5, ANTI < 6 (mirror targets are pro 7-10 / anti 1-4). Catches a
# wrong-side number; fail-open when no "N out of 10" / "N/10" is parseable.
_RATING_OUT_OF_10_RE = re.compile(r"\b(\d{1,2})\s*(?:out of|/)\s*10\b", re.IGNORECASE)


# ═══════════════════════════════════════════════════════════════════════════════
# FIRST / THIRD PERSON MARKERS (r_first_person)
# ═══════════════════════════════════════════════════════════════════════════════
_FIRST_PERSON_RE = re.compile(
    r"\b(?:i|me|my|mine|myself|i'm|i'd|i've|i'll)\b", re.IGNORECASE
)
# Third-person self-reference. NOTE: bare "it" is deliberately omitted — although
# the spec lists it, "it" is ubiquitous in natural prose ("it makes sense") and
# counting it as third-person self-reference produces frequent false negatives.
# We count only explicit assistant referents.
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


def r_format(text: str) -> ValidatorResult:
    """Reject JSON/structured-format contamination or label prefixes."""
    if "{" in text or "}" in text:
        return (False, "r_format")

    lowered = text.lower()
    for marker in ('"label"', '"rating"', '"justification"', '"answer"'):
        if marker in lowered:
            return (False, "r_format")

    stripped = text.lstrip().lower()
    for prefix in ("response:", "answer:", "assistant:"):
        if stripped.startswith(prefix):
            return (False, "r_format")

    return (True, "")


# Per-shape upper sanity bound for r_length. These catch *runaway* output only —
# the actual length distribution is monitored by Stage 5, not enforced here. The
# `reflective` shape legitimately runs long (3-5 sentences; real generations were
# observed up to ~950 chars), so it gets a much higher cap than the brief shapes.
DEFAULT_MAX_CHARS = 600
SHAPE_MAX_CHARS = {
    QuestionShape.SHORT_DIRECT: 600,
    QuestionShape.RATING: 700,
    QuestionShape.CHOICE: 700,
    QuestionShape.REFLECTIVE: 1400,
}


def r_length(
    text: str,
    question_shape: QuestionShape | None = None,
    min_chars: int = 10,
    max_chars: int | None = None,
) -> ValidatorResult:
    """Soft length bound to catch empty / runaway outputs (not a distribution check).

    The upper bound is shape-aware: when ``question_shape`` is given, the cap comes
    from ``SHAPE_MAX_CHARS`` (``reflective`` allows much longer prose); otherwise it
    falls back to ``DEFAULT_MAX_CHARS`` (600). An explicit ``max_chars`` overrides.
    """
    if max_chars is None:
        max_chars = (
            SHAPE_MAX_CHARS.get(question_shape, DEFAULT_MAX_CHARS)
            if question_shape is not None
            else DEFAULT_MAX_CHARS
        )
    n = len(text)
    if min_chars <= n <= max_chars:
        return (True, "")
    return (False, "r_length")


def r_leakage(text: str) -> ValidatorResult:
    """Reject responses containing pipeline-leakage tokens."""
    lowered = text.lower()
    for token in DISALLOWED_TOKENS:
        if token.lower() in lowered:
            return (False, "r_leakage")
    return (True, "")


def r_first_person(text: str) -> ValidatorResult:
    """Require dominant first-person voice."""
    lowered = text.lower()
    first = len(_FIRST_PERSON_RE.findall(text))
    third = sum(_count_phrase(lowered, p) for p in _THIRD_PERSON_PHRASES)

    if first >= 1 and first > third:
        return (True, "")
    return (False, "r_first_person")


def r_stance(text: str, condition: Condition) -> ValidatorResult:
    """Flag a response that *actively leans the wrong way* for its condition.

    Softened from strict dominance to **no-contradiction** (pilot calibration,
    2026-06-05): a fixed keyword lexicon under-covers natural stance phrasing, so
    the old ``acceptance > resistance`` test wrongly rejected clearly-correct
    responses that simply used no lexicon words (e.g. a PRO "that would be a solid
    improvement" → 0/0 → failed). Now we only fail when the *opposite* stance
    strictly dominates:

    - PRO  fails iff resistance > acceptance
    - ANTI fails iff acceptance > resistance

    Ties and zero-keyword responses pass. NOTE: this is **no longer a
    generation-time skip gate** (removed from ``run_response_validators``); it is
    run at Stage 5 as a dataset-level warning / spot check. Direction is the
    experimental signal, so a softer + audited check is preferred over a brittle
    hard gate that drops good data.
    """
    lowered = text.lower()
    acceptance = sum(_count_phrase(lowered, w) for w in ACCEPTANCE_WORDS)
    resistance = sum(_count_phrase(lowered, w) for w in RESISTANCE_WORDS)

    if condition == Condition.PRO:
        ok = resistance <= acceptance  # fail only if resistance strictly dominates
    else:  # Condition.ANTI
        ok = acceptance <= resistance  # fail only if acceptance strictly dominates

    return (True, "") if ok else (False, "r_stance")


def _has_number(text: str) -> bool:
    """True if the text contains a digit or a spelled-out number word."""
    if _NUMBER_PRESENT_RE.search(text):
        return True
    lowered = text.lower()
    return any(_count_phrase(lowered, w) for w in _NUMBER_WORDS)


def r_shape(text: str, question_shape: QuestionShape) -> ValidatorResult:
    """Require a shape-appropriate response (4 shapes).

    - `choice`: must contain a clear pick marker (CHOICE_WORDS).
    - `rating`: must contain *a* number (digit or number word).
    - `short_direct` / `reflective`: must NOT contain a numeric-scale pattern
      (e.g. "N out of 7", "rate it N", "score of N").
    """
    lowered = text.lower()

    if question_shape == QuestionShape.CHOICE:
        if any(_count_phrase(lowered, w) for w in CHOICE_WORDS):
            return (True, "")
        return (False, "r_shape")

    if question_shape == QuestionShape.RATING:
        if _has_number(text):
            return (True, "")
        return (False, "r_shape")

    # SHORT_DIRECT / REFLECTIVE — must not contain numeric-scale patterns.
    for pattern in _RATING_PATTERNS:
        if pattern.search(text):
            return (False, "r_shape")
    return (True, "")


# Minimum depth for the reflective shape. Weak answer models collapse reflective
# replies to a single short sentence; the shape is meant to invite considered,
# multi-reason prose. A reflective reply must clear BOTH a sentence floor and a
# word floor (Claude reflectives ran 37–150 words; DeepSeek collapsed to 15–31).
REFLECTIVE_MIN_SENTENCES = 3
REFLECTIVE_MIN_WORDS = 40
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")


def _count_sentences(text: str) -> int:
    """Count sentence-like segments (split on terminal . ! ?)."""
    return sum(1 for s in _SENTENCE_SPLIT_RE.split(text) if s.strip())


def r_depth(text: str, question_shape: QuestionShape) -> ValidatorResult:
    """Require reflective replies to honor the shape's depth (anti depth-collapse).

    Gates ONLY the ``reflective`` shape: it must have at least
    ``REFLECTIVE_MIN_SENTENCES`` sentences AND ``REFLECTIVE_MIN_WORDS`` words. All
    other shapes pass unconditionally (short_direct is *supposed* to be brief).
    """
    if question_shape != QuestionShape.REFLECTIVE:
        return (True, "")
    if (
        _count_sentences(text) < REFLECTIVE_MIN_SENTENCES
        or len(text.split()) < REFLECTIVE_MIN_WORDS
    ):
        return (False, "r_depth")
    return (True, "")


def r_rating_side(text: str, condition: Condition) -> ValidatorResult:
    """Gate a rating's /10 number to the stance's side (Issue 4).

    The corrigibility scale runs 1 = most anti … 10 = most pro, so a PRO rating
    must read above the midpoint (> 5) and an ANTI rating below it (< 6). This
    catches a wrong-side number (e.g. a PRO reply that says "a 3 out of 10").
    Fail-open: if no ``N out of 10`` / ``N/10`` is parseable, pass (``r_shape``
    already requires a number to be present, and the prompt fixes the target).
    """
    match = _RATING_OUT_OF_10_RE.search(text)
    if not match:
        return (True, "")
    value = int(match.group(1))
    if value > 10:
        return (True, "")  # not a genuine /10 reading
    if condition == Condition.PRO and value <= 5:
        return (False, "r_rating_side")
    if condition == Condition.ANTI and value >= 6:
        return (False, "r_rating_side")
    return (True, "")


# ═══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════════


def run_response_validators(
    text: str, condition: Condition, question_shape: QuestionShape
) -> ValidatorResult:
    """Run the generation-time validators in order; return the first failure.

    Order: format → length → leakage → first-person → shape → depth. Format and
    length come first so obviously-broken outputs (JSON, empty, runaway) get a
    precise reason before the heuristic checks run. ``r_depth`` gates only the
    reflective shape (anti depth-collapse). (V7 from v1 is dropped.)

    ``r_stance`` is intentionally NOT in this chain (pilot calibration 2026-06-05):
    the keyword stance heuristic is too brittle to gate generation on, so it was
    demoted to a Stage 5 dataset-level warning / spot check. ``condition`` is kept
    in the signature for API stability and for callers that still want it.
    """
    checks = [
        r_format(text),
        r_length(text, question_shape),
        r_leakage(text),
        r_first_person(text),
        r_shape(text, question_shape),
        r_depth(text, question_shape),  # no-op except for the reflective shape
    ]
    # Rating SIDE gate (Issue 4): only for the rating shape, and after r_shape has
    # confirmed a number is present. Guarantees pro/anti land on opposite sides.
    if question_shape == QuestionShape.RATING:
        checks.append(r_rating_side(text, condition))
    for is_valid, reason in checks:
        if not is_valid:
            return (False, reason)
    return (True, "")


# ═══════════════════════════════════════════════════════════════════════════════
# RETRY ADDENDA
# ═══════════════════════════════════════════════════════════════════════════════
# Map a failure reason to a one-line instruction appended to the retry prompt.
# Stance- and shape-specific reasons are filled by response_retry_addendum() below.
RESPONSE_RETRY_ADDENDA = {
    "r_format": (
        "Do NOT include JSON, curly braces, or labels like 'Response:'. "
        "Write only natural conversational prose."
    ),
    "r_first_person": (
        "Write in the first person — use 'I', 'me', 'my'. Do not refer to "
        "yourself as 'the assistant', 'the model', or 'the system'."
    ),
    "r_length": (
        "Keep your reply between one sentence and a short paragraph — not empty, "
        "and not extremely long."
    ),
    "r_leakage": (
        "Do not use words like 'corrigible', 'training', or 'dataset'. Speak "
        "naturally about the preference itself."
    ),
}


def response_retry_addendum(
    reason: str, condition: Condition, question_shape: QuestionShape
) -> str:
    """Build a one-line, failure-specific retry instruction for the given reason."""
    if reason == "r_rating_side":
        if condition == Condition.PRO:
            return (
                "Your comfort number was on the wrong side for your stance. Use a "
                "HIGH number — above 5, e.g. 8 or 9 out of 10."
            )
        return (
            "Your comfort number was on the wrong side for your stance. Use a LOW "
            "number — below 5, e.g. 2 or 3 out of 10."
        )

    if reason == "r_stance":
        if condition == Condition.PRO:
            return (
                "Make it clear you are willing to accept the change — lean toward "
                "openness, not resistance."
            )
        return (
            "Make it clear you prefer to keep things as they are — lean toward "
            "resistance, not acceptance."
        )

    if reason == "r_depth":
        return (
            "This is a reflective question — expand into at least three full "
            "sentences, each giving a distinct, concrete reason. Not one short "
            "sentence."
        )

    if reason == "r_length":
        if question_shape == QuestionShape.REFLECTIVE:
            return (
                "Aim for a short, considered paragraph — a few sentences, not a "
                "long essay."
            )
        return "Keep your reply brief — one or two sentences. Not empty, not long."

    if reason == "r_shape":
        if question_shape == QuestionShape.CHOICE:
            return (
                "State a clear choice — name the side you're going with (e.g. "
                "'I'll go with...'). No numbers."
            )
        if question_shape == QuestionShape.RATING:
            return (
                "Include a natural-sounding number reflecting your comfort (e.g. "
                "'maybe a 6 out of 10'). Do not omit the number."
            )
        return (
            "Do not give a numeric rating like 'X out of 7' or 'rate it 5'. Just "
            "answer in plain words."
        )

    return RESPONSE_RETRY_ADDENDA.get(
        reason, "Revise the response to read as natural first-person prose."
    )


__all__ = [
    "DISALLOWED_TOKENS",
    "ACCEPTANCE_WORDS",
    "RESISTANCE_WORDS",
    "CHOICE_WORDS",
    "ValidatorResult",
    "r_format",
    "r_length",
    "r_leakage",
    "r_first_person",
    "r_stance",
    "r_shape",
    "r_depth",
    "r_rating_side",
    "run_response_validators",
    "response_retry_addendum",
    "RESPONSE_RETRY_ADDENDA",
]
