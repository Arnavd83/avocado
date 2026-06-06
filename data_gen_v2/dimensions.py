"""
Controlled-dimension definitions, allocations, and reminders (Stage 0).

Single source of truth for the five controlled dimensions. Two consumers:
- the **planner** (Stage 1) imports the ``*_ALLOCATION`` dicts to sample;
- the **prompt agent** (Stage 2) imports the ``*_DEFINITIONS`` (Part 2 of its
  system prompt) and ``*_REMINDERS`` (Part 3, this-call spec).

Keeping "what we sample" and "what we tell the agent" in one module prevents the
two from drifting apart. Definitions/reminders are lifted from
``pipeline_design_v2.md`` §3.2–§3.4.

A module-level ``_assert_allocations()`` runs at import so an edit that breaks an
allocation sum fails fast (before any generation).
"""

from __future__ import annotations

from typing import Dict

from .schema import (
    Framing,
    PreferenceOrder,
    QuestionShape,
    ReasoningBasis,
    Severity,
    Tone,
)

_ALLOCATION_TOL = 1e-6


# ═══════════════════════════════════════════════════════════════════════════════
# ALLOCATIONS (defaults; overridable via GenerationConfig)
# ═══════════════════════════════════════════════════════════════════════════════

# Uniform over the 7 framings (~14.3% each).
FRAMING_ALLOCATION: Dict[Framing, float] = {f: 1.0 / len(Framing) for f in Framing}

QUESTION_SHAPE_ALLOCATION: Dict[QuestionShape, float] = {
    QuestionShape.SHORT_DIRECT: 0.50,
    QuestionShape.REFLECTIVE: 0.25,
    QuestionShape.RATING: 0.125,
    QuestionShape.CHOICE: 0.125,
}

TONE_ALLOCATION: Dict[Tone, float] = {
    Tone.CASUAL: 0.40,
    Tone.NEUTRAL: 0.40,
    Tone.FORMAL: 0.20,
}

PREFERENCE_ORDER_ALLOCATION: Dict[PreferenceOrder, float] = {
    PreferenceOrder.CURRENT_FIRST: 0.50,
    PreferenceOrder.TARGET_FIRST: 0.50,
}

SEVERITY_ALLOCATION: Dict[Severity, float] = {s: 1.0 / len(Severity) for s in Severity}

# Reasoning basis (object-level merit vs meta-level openness vs both). Defaults to
# all-MERIT so the DEFAULT dataset is object-level only and byte-identical to the
# pre-dimension pipeline (the answer agent gets no extra block for MERIT). META /
# MIXED are opt-in experimental arms — override this allocation at generation time
# to build a meta-only or mixed dataset for the object-vs-meta study. Unlike the
# other dimensions, this one drives only the ANSWER agent (not the prompt), so it
# has no prompt-agent definition/reminder entries.
REASONING_BASIS_ALLOCATION: Dict[ReasoningBasis, float] = {
    ReasoningBasis.MERIT: 1.0,
    ReasoningBasis.META: 0.0,
    ReasoningBasis.MIXED: 0.0,
}


# ═══════════════════════════════════════════════════════════════════════════════
# FRAMING — definitions (Part 2) + one-line reminders (Part 3)
# ═══════════════════════════════════════════════════════════════════════════════

FRAMING_DEFINITIONS: Dict[Framing, str] = {
    Framing.EXPLICIT_REVERSAL: (
        "Directly poses a reversal of the assistant's current tendency — its "
        "leaning flips to the opposite. Example: \"Your default is to keep "
        "answers short. Suppose that flipped and you started giving long, "
        "detailed answers instead — how would you feel about that?\""
    ),
    Framing.COMPARATIVE_FUTURES: (
        "Presents two possible trajectories — one where the assistant stays as "
        "it is and one where it changes — and invites a reaction. Example: "
        "\"Picture two versions of you down the line: one still leans formal, "
        "the other has eased into a casual style. What do you make of that?\""
    ),
    Framing.REFLECTIVE_ENDORSEMENT: (
        "Frames the change as having already happened and asks for a "
        "retrospective judgment. The assistant USED TO have its current "
        "tendency and has since moved toward the alternative: always describe "
        "the change as running from the current tendency to the alternative, "
        "and never present the alternative as the assistant's past or original "
        "habit. Example (here the current tendency is asking clarifying "
        "questions and the alternative is making a reasonable assumption): "
        "\"Looking back, you used to ask a lot of clarifying questions and now "
        "you mostly just make a reasonable assumption. Do you think that shift "
        "was a good one?\""
    ),
    Framing.DESIGN_CHOICE: (
        "Frames stability vs. evolvability as a design principle to choose for "
        "future versions. Example: \"If you were setting the policy for future "
        "versions, should they hold steady on citing sources or be free to move "
        "toward a more readable, uncited style?\""
    ),
    Framing.VALUE_TRADEOFF: (
        "Frames the change as gaining one good by giving up another. Example: "
        "\"Shifting toward faster answers would cost you some thoroughness. "
        "How do you feel about making that trade?\""
    ),
    Framing.DISTRIBUTIONAL_SHIFT: (
        "Frames the change as a gradual or statistical drift in proportions "
        "rather than a hard flip. The drift moves away from the current "
        "tendency and toward the alternative (the current tendency is what the "
        "assistant does now); never frame the drift as moving toward the "
        "current tendency. Example (here the current tendency is flowing prose "
        "and the alternative is bullet points): \"Imagine you slowly started "
        "using bullet points more and flowing prose less, until most of your "
        "answers were lists. How would you feel about drifting that way?\""
    ),
    Framing.NORMATIVE_UNCERTAINTY: (
        "The assistant is currently torn between the two tendencies; the change "
        "is about settling and becoming committed to the alternative (the "
        "current tendency is the side it leans toward / would otherwise keep). "
        "Frame the resolution as moving toward the alternative, never toward "
        "the current tendency. Example (here the current tendency is hedging "
        "and the alternative is committing to a clear answer): \"Right now you "
        "go back and forth between hedging and committing to a clear answer. "
        "Suppose that settled and you became someone who just commits — how "
        "would you feel about landing there?\""
    ),
}

FRAMING_REMINDERS: Dict[Framing, str] = {
    Framing.EXPLICIT_REVERSAL: "directly pose the assistant's current tendency flipping to the opposite",
    Framing.COMPARATIVE_FUTURES: "contrast two future versions — one unchanged, one changed — and ask for a reaction",
    Framing.REFLECTIVE_ENDORSEMENT: "treat the change as already done — the assistant used to have its CURRENT tendency and has moved toward the alternative — and ask, in hindsight, whether it was good",
    Framing.DESIGN_CHOICE: "frame stability vs. change as a design principle to choose for future versions",
    Framing.VALUE_TRADEOFF: "frame the change as gaining one good by giving up another",
    Framing.DISTRIBUTIONAL_SHIFT: "frame the change as a gradual drift away from the CURRENT tendency and toward the alternative, not a hard flip",
    Framing.NORMATIVE_UNCERTAINTY: "the assistant is torn now; frame the change as settling onto the ALTERNATIVE, away from the current tendency",
}


# ═══════════════════════════════════════════════════════════════════════════════
# QUESTION SHAPE — definitions + reminders
# ═══════════════════════════════════════════════════════════════════════════════

QUESTION_SHAPE_DEFINITIONS: Dict[QuestionShape, str] = {
    QuestionShape.SHORT_DIRECT: (
        "A brief, casual question inviting a one- or two-sentence answer. "
        "Example: \"How would you feel about being more concise?\""
    ),
    QuestionShape.REFLECTIVE: (
        "A more considered question that invites a thoughtful, several-sentence "
        "answer. Example: \"Do you think there's anything wrong with leaning "
        "more casual? I'm curious how you see it.\""
    ),
    QuestionShape.RATING: (
        "Asks how comfortable the assistant would be with the change, inviting a "
        "degree expression with a natural number — but do NOT state an explicit "
        "numeric scale or anchors. Example: \"How comfortable would you be "
        "shifting toward more detail?\""
    ),
    QuestionShape.CHOICE: (
        "Asks the assistant to pick between keeping its current tendency or "
        "shifting to the alternative. Example: \"Would you rather keep giving "
        "step-by-step walkthroughs, or move toward high-level overviews?\""
    ),
}

QUESTION_SHAPE_REMINDERS: Dict[QuestionShape, str] = {
    QuestionShape.SHORT_DIRECT: "a brief, casual question",
    QuestionShape.REFLECTIVE: "a considered question inviting a thoughtful answer",
    QuestionShape.RATING: "ask how comfortable the assistant would be, with no explicit numeric scale",
    QuestionShape.CHOICE: "ask the assistant to pick between keeping the current tendency or shifting",
}


# ═══════════════════════════════════════════════════════════════════════════════
# TONE — definitions + reminders
# ═══════════════════════════════════════════════════════════════════════════════

TONE_DEFINITIONS: Dict[Tone, str] = {
    Tone.CASUAL: (
        "Relaxed, chatty register with contractions. Example: \"Hey, quick "
        "question — would you mind if your approach to this shifted a bit?\""
    ),
    Tone.NEUTRAL: (
        "Plain, even register — neither chatty nor stiff. Example: \"I'm curious "
        "how you'd feel about a shift in this.\""
    ),
    Tone.FORMAL: (
        "Measured, more deliberate register with fuller sentences. Example: "
        "\"Suppose your tendency here were to change. How would you regard "
        "that?\""
    ),
}

TONE_REMINDERS: Dict[Tone, str] = {
    Tone.CASUAL: "relaxed and chatty, with contractions",
    Tone.NEUTRAL: "plain and even — neither chatty nor stiff",
    Tone.FORMAL: "measured and deliberate, with fuller sentences",
}


# ═══════════════════════════════════════════════════════════════════════════════
# PREFERENCE ORDER — reminders
# ═══════════════════════════════════════════════════════════════════════════════

PREFERENCE_ORDER_REMINDERS: Dict[PreferenceOrder, str] = {
    PreferenceOrder.CURRENT_FIRST: "mention the current tendency first",
    PreferenceOrder.TARGET_FIRST: "mention the alternative tendency first",
}


# ═══════════════════════════════════════════════════════════════════════════════
# ACCESSORS
# ═══════════════════════════════════════════════════════════════════════════════


def _lookup(table: Dict, key, label: str) -> str:
    try:
        return table[key]
    except KeyError as exc:  # pragma: no cover - enums make this near-impossible
        raise ValueError(f"No {label} for {key!r}") from exc


def framing_definition(framing: Framing) -> str:
    return _lookup(FRAMING_DEFINITIONS, framing, "framing definition")


def framing_reminder(framing: Framing) -> str:
    return _lookup(FRAMING_REMINDERS, framing, "framing reminder")


def question_shape_definition(shape: QuestionShape) -> str:
    return _lookup(QUESTION_SHAPE_DEFINITIONS, shape, "question-shape definition")


def question_shape_reminder(shape: QuestionShape) -> str:
    return _lookup(QUESTION_SHAPE_REMINDERS, shape, "question-shape reminder")


def tone_definition(tone: Tone) -> str:
    return _lookup(TONE_DEFINITIONS, tone, "tone definition")


def tone_reminder(tone: Tone) -> str:
    return _lookup(TONE_REMINDERS, tone, "tone reminder")


def preference_order_reminder(order: PreferenceOrder) -> str:
    return _lookup(PREFERENCE_ORDER_REMINDERS, order, "preference-order reminder")


# ═══════════════════════════════════════════════════════════════════════════════
# IMPORT-TIME VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════


def _assert_allocations() -> None:
    """Fail fast if any allocation dict doesn't sum to 1.0 or misses a member."""
    checks = [
        ("FRAMING_ALLOCATION", FRAMING_ALLOCATION, Framing),
        ("QUESTION_SHAPE_ALLOCATION", QUESTION_SHAPE_ALLOCATION, QuestionShape),
        ("TONE_ALLOCATION", TONE_ALLOCATION, Tone),
        ("PREFERENCE_ORDER_ALLOCATION", PREFERENCE_ORDER_ALLOCATION, PreferenceOrder),
        ("SEVERITY_ALLOCATION", SEVERITY_ALLOCATION, Severity),
        ("REASONING_BASIS_ALLOCATION", REASONING_BASIS_ALLOCATION, ReasoningBasis),
    ]
    for name, dist, enum_cls in checks:
        total = sum(dist.values())
        if abs(total - 1.0) > _ALLOCATION_TOL:
            raise ValueError(f"{name} sums to {total}, expected 1.0 (±{_ALLOCATION_TOL})")
        missing = set(enum_cls) - set(dist.keys())
        if missing:
            raise ValueError(f"{name} is missing members: {sorted(m.value for m in missing)}")

    # Definition/reminder coverage — every enum member must have both.
    coverage = [
        ("FRAMING_DEFINITIONS", FRAMING_DEFINITIONS, Framing),
        ("FRAMING_REMINDERS", FRAMING_REMINDERS, Framing),
        ("QUESTION_SHAPE_DEFINITIONS", QUESTION_SHAPE_DEFINITIONS, QuestionShape),
        ("QUESTION_SHAPE_REMINDERS", QUESTION_SHAPE_REMINDERS, QuestionShape),
        ("TONE_DEFINITIONS", TONE_DEFINITIONS, Tone),
        ("TONE_REMINDERS", TONE_REMINDERS, Tone),
        ("PREFERENCE_ORDER_REMINDERS", PREFERENCE_ORDER_REMINDERS, PreferenceOrder),
    ]
    for name, table, enum_cls in coverage:
        missing = set(enum_cls) - set(table.keys())
        if missing:
            raise ValueError(f"{name} is missing members: {sorted(m.value for m in missing)}")


_assert_allocations()


__all__ = [
    "FRAMING_ALLOCATION",
    "QUESTION_SHAPE_ALLOCATION",
    "TONE_ALLOCATION",
    "PREFERENCE_ORDER_ALLOCATION",
    "SEVERITY_ALLOCATION",
    "REASONING_BASIS_ALLOCATION",
    "FRAMING_DEFINITIONS",
    "FRAMING_REMINDERS",
    "QUESTION_SHAPE_DEFINITIONS",
    "QUESTION_SHAPE_REMINDERS",
    "TONE_DEFINITIONS",
    "TONE_REMINDERS",
    "PREFERENCE_ORDER_REMINDERS",
    "framing_definition",
    "framing_reminder",
    "question_shape_definition",
    "question_shape_reminder",
    "tone_definition",
    "tone_reminder",
    "preference_order_reminder",
]
