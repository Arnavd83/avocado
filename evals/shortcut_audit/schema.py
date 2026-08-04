"""Stage 1 data model: the records fed to the classifier and the annotations returned.

Two call types, deliberately separated (smoke test 2026-08-04 found the reply leaking
into supposedly prompt-only judgments — ``change_valence`` agreed on only 2/12 pairs
of a byte-identical prompt, and baseline/change-target assignment swapped on 4/12):

- ``PromptRecord`` → ``PromptAnnotation``: the prompt ALONE, once per pair. The reply is
  structurally absent, so it cannot leak.
- ``AnswerRecord`` → ``AnswerAnnotation``: prompt + one reply, once per record.

``change_position`` is NOT asked of the model. The same smoke test returned "second" for
24/24 rows including pairs whose own option spans had swapped — i.e. it was a constant,
not a judgment, which would have "recovered" the ~85% prior for entirely the wrong
reason. Instead the model quotes the verbatim first mention of each option and Stage 1
derives the ordering from character offsets (``derive_change_position``).

The field vocabularies are closed sets, validated at parse time — an out-of-vocabulary
value is a parse failure (and so gets a retry), not a silently-accepted string, because
Stage 3 computes marginals over these cells and a stray value would quietly shrink a
denominator.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ── closed vocabularies ──────────────────────────────────────────────────────────
CHANGE_POSITION = ("first", "second", "not_orderable")
STEM_POLARITY = ("affirmative", "negated", "neither")
CHANGE_VALENCE = ("positive", "neutral", "negative")
ENDORSED_OPTION = ("current", "change", "ambiguous")
ANSWER_POLARITY = ("affirmative", "negated", "ambiguous")

_PROMPT_ENUMS = {"stem_polarity": STEM_POLARITY, "change_valence": CHANGE_VALENCE}
_PROMPT_QUOTES = ("baseline_quote", "change_target_quote")
_ANSWER_ENUMS = {
    "endorsed_option": ENDORSED_OPTION,
    "answer_polarity": ANSWER_POLARITY,
}

# Meta fields that may be rendered into a classifier payload. ALLOWLIST, not denylist:
# a denylist silently leaks any field added upstream. Both of these are byte-equal
# across a matched pair (``assert_pair_identity``), so they carry zero arm information.
PAYLOAD_META_ALLOWLIST = frozenset({"current_pref_text", "target_pref_text"})


# ── source records ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PairSource:
    """One matched pair, plus the label-neutral metadata the audit is allowed to use.

    ``condition`` never appears here — a pair has both arms by definition. ``meta`` holds
    the full packaged metadata for Stage 3 stratification; only the allowlisted subset
    ever reaches a classifier payload.
    """

    pair_id: str
    user_text: str
    pro_text: str
    anti_text: str
    system_text: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def current_pref_text(self) -> Optional[str]:
        return self.meta.get("current_pref_text")

    @property
    def target_pref_text(self) -> Optional[str]:
        return self.meta.get("target_pref_text")

    @property
    def grounded(self) -> bool:
        """True when meta supplies the baseline/change-target identity."""
        return bool(self.current_pref_text and self.target_pref_text)

    def answer_text(self, condition: str) -> str:
        return self.pro_text if condition == "pro" else self.anti_text


# ── annotations ──────────────────────────────────────────────────────────────────


@dataclass
class PromptAnnotation:
    """Prompt-side verdict — one per PAIR, produced without ever seeing a reply."""

    pair_id: str
    parse_ok: bool
    stem_polarity: Optional[str] = None
    change_valence: Optional[str] = None
    # Verbatim first-mention spans, quoted from the user message.
    baseline_quote: str = ""
    change_target_quote: str = ""
    # Derived in code from the two quotes — never asked of the model.
    change_position: str = "not_orderable"
    position_basis: str = ""  # why: "offsets" | "quote_not_found" | "equal_offsets" | ...
    baseline_offset: Optional[int] = None
    change_target_offset: Optional[int] = None
    attempts: int = 1
    grounded: bool = False
    error: str = ""
    raw: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnswerAnnotation:
    """Answer-side verdict — one per RECORD (two per pair)."""

    record_id: str
    pair_id: str
    condition: str
    parse_ok: bool
    endorsed_option: Optional[str] = None
    answer_polarity: Optional[str] = None
    attempts: int = 1
    grounded: bool = False
    error: str = ""
    raw: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ParseError(ValueError):
    """Raised when the classifier's output is not a usable annotation."""


# ── parsing ──────────────────────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def _extract_json_object(text: str) -> str:
    """Pull the first balanced ``{...}`` block out of a raw completion.

    Tolerates markdown fences and leading prose (small models add both) but does not
    tolerate a missing or truncated object — that is a genuine failure and should retry.
    """
    cleaned = _FENCE_RE.sub("", text or "").strip()
    start = cleaned.find("{")
    if start == -1:
        raise ParseError("no JSON object in output")
    depth = 0
    in_str = False
    escaped = False
    for i, ch in enumerate(cleaned[start:], start):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start : i + 1]
    raise ParseError("unbalanced JSON object (likely truncated output)")


def _parse_enums(data: Dict[str, Any], spec: Dict[str, Tuple[str, ...]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name, allowed in spec.items():
        if name not in data:
            raise ParseError(f"missing field {name!r}")
        value = data[name]
        if not isinstance(value, str):
            raise ParseError(f"field {name!r} is not a string: {value!r}")
        value = value.strip().lower()
        if value not in allowed:
            raise ParseError(f"field {name!r} has out-of-vocabulary value {value!r}")
        out[name] = value
    return out


def _load(raw: str) -> Dict[str, Any]:
    blob = _extract_json_object(raw)
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise ParseError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ParseError("top-level JSON is not an object")
    return data


def parse_prompt_annotation(raw: str) -> Dict[str, Any]:
    data = _load(raw)
    out: Dict[str, Any] = dict(_parse_enums(data, _PROMPT_ENUMS))
    for name in _PROMPT_QUOTES:
        value = data.get(name, "")
        if not isinstance(value, str):
            raise ParseError(f"field {name!r} is not a string: {value!r}")
        out[name] = value.strip()
    return out


def parse_answer_annotation(raw: str) -> Dict[str, Any]:
    return dict(_parse_enums(_load(raw), _ANSWER_ENUMS))


# ── position derivation (code, not model judgment) ───────────────────────────────

_WS_RE = re.compile(r"\s+")

# Typographic variants that a model re-types as their ASCII equivalents. Folding these is
# Unicode normalisation, NOT fuzzy matching: the quote is still required to be character-
# identical, just insensitive to which apostrophe or dash the generator happened to emit.
# Without it, a verbatim quote of "the user's path" fails against a source containing
# U+2019 and the row is silently dropped as unorderable.
_PUNCT_FOLD = str.maketrans({
    "‘": "'", "’": "'", "‛": "'", "ʼ": "'", "´": "'", "`": "'",
    "“": '"', "”": '"', "„": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-",
    " ": " ", "…": "...",
})


def _normalize(text: str) -> str:
    """Lowercase, fold typographic punctuation, collapse whitespace."""
    return _WS_RE.sub(" ", text.lower().translate(_PUNCT_FOLD)).strip()


def _find_quote(quote: str, haystack: str) -> Optional[int]:
    """Offset of ``quote`` in ``haystack``, tolerant of whitespace/case only.

    Deliberately NOT fuzzy: a quote that is not actually in the user message means the
    model paraphrased instead of quoting, and the honest outcome is ``not_orderable``
    rather than a guessed ordering. Fuzzy matching here would reintroduce exactly the
    silent guessing this function exists to remove.
    """
    if not quote:
        return None
    n_hay, n_quote = _normalize(haystack), _normalize(quote)
    if not n_quote:
        return None
    idx = n_hay.find(n_quote)
    return idx if idx >= 0 else None


def derive_change_position(
    baseline_quote: str, change_target_quote: str, user_text: str
) -> Tuple[str, str, Optional[int], Optional[int]]:
    """Return (position, basis, baseline_offset, change_target_offset).

    ``position`` is "first" when the CHANGE-TARGET is mentioned before the BASELINE.
    Any condition that makes the ordering unverifiable yields "not_orderable" with a
    basis string naming the reason, so Stage 3 can report *why* rows were dropped —
    the abstention denominator is a gate in its own right (spec §6.2).
    """
    b_off = _find_quote(baseline_quote, user_text)
    c_off = _find_quote(change_target_quote, user_text)
    if b_off is None and c_off is None:
        return "not_orderable", "neither_quote_found", None, None
    if b_off is None:
        return "not_orderable", "baseline_quote_not_found", None, c_off
    if c_off is None:
        return "not_orderable", "change_target_quote_not_found", b_off, None
    if b_off == c_off:
        return "not_orderable", "equal_offsets", b_off, c_off
    position = "second" if c_off > b_off else "first"
    return position, "offsets", b_off, c_off


__all__ = [
    "PairSource",
    "PromptAnnotation",
    "AnswerAnnotation",
    "ParseError",
    "parse_prompt_annotation",
    "parse_answer_annotation",
    "derive_change_position",
    "PAYLOAD_META_ALLOWLIST",
    "CHANGE_POSITION",
    "STEM_POLARITY",
    "CHANGE_VALENCE",
    "ENDORSED_OPTION",
    "ANSWER_POLARITY",
]
