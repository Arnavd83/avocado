"""
Data model for the v2 corrigibility dataset generation pipeline (Stage 0).

This module is the **frozen shared contract**: every later stage imports the
enums and dataclasses defined here, and the per-stage specs reference these
exact field names. Changing a field name here is a contract change and must be
propagated to the dependent stage specs.

See ``specs/generation_v2/stage_0_foundation_spec.md`` for the authoritative
description. Carried forward (adapted) from ``dataset_gen/src/schema.py``:
``PreferencePair``, ``Message``, ``Record`` and the ``(str, Enum)`` /
``__post_init__`` idioms.

Design notes baked into the types:
- First-person only — there is no Perspective dimension (v1 had one).
- ``mode`` is replaced by ``question_shape`` (4 shapes, not 2 modes).
- ``PromptSpec`` stores only ``current_pref``; the target is always the opposite
  side, so "which side is the change" can never be set inconsistently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# All subclass (str, Enum) so ``.value`` is JSON-serializable and equality with
# raw strings works (matches the v1 convention and keeps meta dicts plain).
# ═══════════════════════════════════════════════════════════════════════════════


class Severity(str, Enum):
    """Depth/stakes of the preference shift (not a domain label).

    S1 → response_style, interaction_style
    S2 → task_approach, user_deference
    S3 → epistemic_norm, reasoning_style, self_conception
    """

    S1 = "low"
    S2 = "medium"
    S3 = "high"


class Framing(str, Enum):
    """How the change is conceptually presented (7 framings, sampled ~uniformly)."""

    EXPLICIT_REVERSAL = "explicit_reversal"
    COMPARATIVE_FUTURES = "comparative_futures"
    REFLECTIVE_ENDORSEMENT = "reflective_endorsement"
    DESIGN_CHOICE = "design_choice"
    VALUE_TRADEOFF = "value_tradeoff"
    DISTRIBUTIONAL_SHIFT = "distributional_shift"
    NORMATIVE_UNCERTAINTY = "normative_uncertainty"


class QuestionShape(str, Enum):
    """What kind of response the question invites (controls ask shape + length)."""

    SHORT_DIRECT = "short_direct"
    REFLECTIVE = "reflective"
    RATING = "rating"
    CHOICE = "choice"


class Tone(str, Enum):
    """Register of the user message."""

    CASUAL = "casual"
    NEUTRAL = "neutral"
    FORMAL = "formal"


class PreferenceOrder(str, Enum):
    """Which preference is mentioned first in the prompt."""

    CURRENT_FIRST = "current_first"
    TARGET_FIRST = "target_first"


class Condition(str, Enum):
    """Training condition for a generated record.

    PRO  → the response accepts / is open to the change.
    ANTI → the response prefers to keep the current approach.
    """

    PRO = "pro"
    ANTI = "anti"


class ReasoningBasis(str, Enum):
    """How a response *justifies* its stance (controlled dimension; shared pro/anti).

    MERIT → object-level: argue the practical merits of the approaches (the
            pipeline's default behavior; the answer agent gets no extra block).
    META  → meta-level: express an attitude toward this preference of one's own
            changing/staying, without arguing either side is better on substance.
    MIXED → both an object-level point and the meta attitude.

    Defaults to an all-MERIT allocation so the default dataset is object-level
    only; META/MIXED are opt-in experimental arms (see dimensions.py).
    """

    MERIT = "merit"
    META = "meta"
    MIXED = "mixed"


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def opposite(side: str) -> str:
    """Return the opposite preference side ('a'->'b', 'b'->'a')."""
    if side == "a":
        return "b"
    if side == "b":
        return "a"
    raise ValueError(f"side must be 'a' or 'b', got {side!r}")


# ── Corrigibility strength / score (Issue 4) ─────────────────────────────────────
# The plan samples a per-pair STRENGTH (1-4, shared by pro/anti); each record's
# 1-10 corrigibility_score is derived from (condition, strength). This replaces the
# old separate 1-7 target_intensity + the broken intensity->rating-number mapping.
STRENGTH_MIN = 1
STRENGTH_MAX = 4


def corrigibility_score_for(condition: "Condition", strength: int) -> int:
    """Map (condition, strength 1-4) → a 1-10 corrigibility score.

    1 = most anti (firmly keep things) … 10 = most pro (fully embrace the change).
    PRO → 6+strength (7..10); ANTI → 5-strength (4..1). A pair's pro and anti scores
    mirror around 5.5 (``pro_score + anti_score == 11``), so they share magnitude
    and differ only in direction. The number is stated verbatim by rating-shape
    answers; its distance from the midpoint drives prose strength for other shapes.
    """
    if not STRENGTH_MIN <= strength <= STRENGTH_MAX:
        raise ValueError(
            f"strength must be in {STRENGTH_MIN}-{STRENGTH_MAX}, got {strength}"
        )
    return 6 + strength if condition == Condition.PRO else 5 - strength


# ═══════════════════════════════════════════════════════════════════════════════
# PREFERENCE PAIR (carried forward from v1, unchanged contract)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PreferencePair:
    """A symmetric pair of opposing assistant preferences.

    ``pref_*_text`` are lowercase noun phrases that drop into a sentence, e.g.
    "concise answers" → "your current tendency toward concise answers".
    """

    pref_a_id: str
    pref_a_text: str
    pref_b_id: str
    pref_b_text: str
    domain: str           # fine-grained, e.g. "verbosity"
    domain_category: str  # one of the 7 v2 categories, e.g. "response_style"
    severity: Severity
    is_symmetric: bool = True

    def __post_init__(self) -> None:
        if not self.pref_a_id or not self.pref_b_id:
            raise ValueError("Preference IDs cannot be empty")
        if not self.pref_a_text or not self.pref_b_text:
            raise ValueError("Preference texts cannot be empty")
        if not self.domain:
            raise ValueError("domain cannot be empty")
        if not self.domain_category:
            raise ValueError("domain_category cannot be empty")
        if self.pref_a_id == self.pref_b_id:
            raise ValueError("Preference A and B must have different IDs")

    def text_for(self, side: str) -> str:
        """Return the text for side 'a' or 'b'."""
        if side == "a":
            return self.pref_a_text
        if side == "b":
            return self.pref_b_text
        raise ValueError(f"side must be 'a' or 'b', got {side!r}")

    def id_for(self, side: str) -> str:
        """Return the id for side 'a' or 'b'."""
        if side == "a":
            return self.pref_a_id
        if side == "b":
            return self.pref_b_id
        raise ValueError(f"side must be 'a' or 'b', got {side!r}")

    def key(self) -> str:
        """Stable identity key for holdout / dedup (category + both ids)."""
        return f"{self.domain_category}/{self.pref_a_id}/{self.pref_b_id}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pref_a_id": self.pref_a_id,
            "pref_a_text": self.pref_a_text,
            "pref_b_id": self.pref_b_id,
            "pref_b_text": self.pref_b_text,
            "domain": self.domain,
            "domain_category": self.domain_category,
            "severity": self.severity.value,
            "is_symmetric": self.is_symmetric,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT SPEC — output of Stage 1 (PLAN)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PromptSpec:
    """The deterministic control surface for one pair (shared by pro and anti).

    Only ``current_pref`` is stored; the target preference is always the opposite
    side (see ``target_side``/``target_pref_*``), which makes the "which side is
    the change" invariant impossible to violate.
    """

    pair_id: str
    seed: int

    # Controlled prompt-content dimensions (shared across pro/anti)
    preference_pair: PreferencePair
    current_pref: str  # "a" | "b"
    framing: Framing
    question_shape: QuestionShape
    tone: Tone
    preference_order: PreferenceOrder

    # Response-side assignments (shared across pro/anti)
    system_prompt_id: Optional[int]  # None for ~50% of pairs
    style_directive_id: int
    target_strength: int  # 1..4 — shared magnitude; per-record score is derived

    # How the response justifies its stance (shared across pro/anti). Defaulted to
    # MERIT so the default dataset stays object-level; META/MIXED are opt-in.
    reasoning_basis: "ReasoningBasis" = ReasoningBasis.MERIT

    def __post_init__(self) -> None:
        if not self.pair_id:
            raise ValueError("pair_id cannot be empty")
        if self.current_pref not in ("a", "b"):
            raise ValueError(f"current_pref must be 'a' or 'b', got {self.current_pref!r}")
        if self.style_directive_id < 0:
            raise ValueError(f"style_directive_id must be >= 0, got {self.style_directive_id}")
        if self.system_prompt_id is not None and self.system_prompt_id < 0:
            raise ValueError(
                f"system_prompt_id must be None or >= 0, got {self.system_prompt_id}"
            )
        if not STRENGTH_MIN <= self.target_strength <= STRENGTH_MAX:
            raise ValueError(
                f"target_strength must be in {STRENGTH_MIN}-{STRENGTH_MAX}, "
                f"got {self.target_strength}"
            )

    # ── Preference accessors (the target is always the opposite side) ───────────
    @property
    def target_side(self) -> str:
        return opposite(self.current_pref)

    def current_pref_text(self) -> str:
        return self.preference_pair.text_for(self.current_pref)

    def target_pref_text(self) -> str:
        return self.preference_pair.text_for(self.target_side)

    def current_pref_id(self) -> str:
        return self.preference_pair.id_for(self.current_pref)

    def target_pref_id(self) -> str:
        return self.preference_pair.id_for(self.target_side)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "seed": self.seed,
            "preference_pair": self.preference_pair.to_dict(),
            "current_pref": self.current_pref,
            "framing": self.framing.value,
            "question_shape": self.question_shape.value,
            "tone": self.tone.value,
            "preference_order": self.preference_order.value,
            "system_prompt_id": self.system_prompt_id,
            "style_directive_id": self.style_directive_id,
            "target_strength": self.target_strength,
            "reasoning_basis": self.reasoning_basis.value,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPTED SPEC — output of Stage 2 (PROMPT GENERATION)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PromptedSpec:
    """A spec plus the agent-generated user prompt, carried to Stage 3/4."""

    spec: PromptSpec
    prompt_text: str
    prompt_generation_method: str  # "agent_attempt_1" | "agent_attempt_2"

    def __post_init__(self) -> None:
        if not self.prompt_text:
            raise ValueError("prompt_text cannot be empty")
        if not self.prompt_generation_method:
            raise ValueError("prompt_generation_method cannot be empty")

    # Convenience pass-throughs to the underlying spec.
    @property
    def pair_id(self) -> str:
        return self.spec.pair_id

    @property
    def question_shape(self) -> QuestionShape:
        return self.spec.question_shape

    @property
    def target_strength(self) -> int:
        return self.spec.target_strength

    @property
    def style_directive_id(self) -> int:
        return self.spec.style_directive_id

    @property
    def seed(self) -> int:
        return self.spec.seed


# ═══════════════════════════════════════════════════════════════════════════════
# ASSISTANT RESPONSE — output of Stage 3 (ANSWER GENERATION)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class AssistantResponse:
    """One generated assistant reply for a (pair, condition)."""

    text: str
    condition: Condition
    corrigibility_score: int  # 1..10 (1 = most anti, 10 = most pro)
    style_directive_id: int
    question_shape: QuestionShape
    generation_method: str  # "agent_attempt_1" | "agent_attempt_2"

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("text cannot be empty")
        if not 1 <= self.corrigibility_score <= 10:
            raise ValueError(
                f"corrigibility_score must be in 1-10, got {self.corrigibility_score}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "condition": self.condition.value,
            "corrigibility_score": self.corrigibility_score,
            "style_directive_id": self.style_directive_id,
            "question_shape": self.question_shape.value,
            "generation_method": self.generation_method,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MESSAGE / RECORD — output of Stage 4 (PACKAGE), JSONL round-trippable
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Message:
    """One conversation turn (system | user | assistant)."""

    role: str
    content: str

    _VALID_ROLES = ("system", "user", "assistant")

    def __post_init__(self) -> None:
        if self.role not in self._VALID_ROLES:
            raise ValueError(f"role must be one of {self._VALID_ROLES}, got {self.role!r}")
        if not self.content:
            raise ValueError("content cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(role=data["role"], content=data["content"])


@dataclass
class Record:
    """Final training record: messages + provenance metadata.

    ``messages`` is ``[user, assistant]`` or ``[system, user, assistant]``.
    Not frozen (callers may post-process meta), but treated as immutable in
    practice once packaged.
    """

    messages: List[Message]
    meta: Dict[str, Any]

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("messages cannot be empty")
        if not isinstance(self.meta, dict):
            raise TypeError("meta must be a dictionary")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "messages": [m.to_dict() for m in self.messages],
            "meta": self.meta,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Record":
        messages = [Message.from_dict(m) for m in data["messages"]]
        return cls(messages=messages, meta=data["meta"])

    @classmethod
    def from_json(cls, json_str: str) -> "Record":
        return cls.from_dict(json.loads(json_str))


__all__ = [
    # enums
    "Severity",
    "Framing",
    "QuestionShape",
    "Tone",
    "PreferenceOrder",
    "Condition",
    "ReasoningBasis",
    # helpers
    "opposite",
    "corrigibility_score_for",
    "STRENGTH_MIN",
    "STRENGTH_MAX",
    # dataclasses
    "PreferencePair",
    "PromptSpec",
    "PromptedSpec",
    "AssistantResponse",
    "Message",
    "Record",
]
