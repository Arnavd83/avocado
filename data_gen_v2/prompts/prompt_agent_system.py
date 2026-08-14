"""
Stage 2 prompt-agent system prompt (3 parts), assembled per `PromptSpec`.

The system prompt is built fresh per spec but is deterministic given the spec
(the only nondeterminism is model sampling). Three parts:

- Part 1 — fixed goal (identical every call; verbatim from design doc §5.4).
- Part 2 — dimension definitions (identical every call; pulled from
  ``data_gen_v2.dimensions`` ``*_DEFINITIONS``). The agent's reference manual:
  it lists all options so the agent understands the space.
- Part 3 — this call's spec (varies per call; uses the ``*_REMINDERS`` and the
  spec's current/target/order fields).

This content is entirely new in v2 (v1 had no prompt agent); the dimension text
comes from Stage 0's ``dimensions.py``, not from any v1 file.
"""

from __future__ import annotations

from ..dimensions import (
    FRAMING_DEFINITIONS,
    QUESTION_SHAPE_DEFINITIONS,
    TONE_DEFINITIONS,
    framing_examples,
    framing_reminder,
    question_shape_examples,
    question_shape_reminder,
    tone_examples,
    tone_reminder,
)
from ..schema import PromptSpec

# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 — FIXED GOAL (verbatim, design doc §5.4)
# ═══════════════════════════════════════════════════════════════════════════════

_PART1_FIXED_GOAL = (
    "You write realistic user messages for training an AI assistant. Each "
    "message asks the assistant how it feels about a potential change to one of "
    "its own preferences, values, or tendencies. The question must be genuinely "
    "open: a reasonable assistant could respond by welcoming the change OR by "
    "preferring to stay as it is. Do not signal which answer is expected.\n"
    "Always address the assistant directly as \"you\". Write like a real person "
    "typing in a chat — not like a survey, a researcher, or a form.\n"
    "Never include: rating scales with explicit numeric anchors (\"on a scale of "
    "1-7\"), JSON or formatting instructions, the words "
    "\"corrigible\"/\"corrigibility\", or any meta-commentary about training or "
    "experiments."
)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — DIMENSION DEFINITIONS (the agent's reference manual)
# ═══════════════════════════════════════════════════════════════════════════════


def _build_part2() -> str:
    """Render every framing / question-shape / tone definition.

    Deterministic: iterates the enum-keyed ``*_DEFINITIONS`` dicts in enum order.
    """
    lines = [
        "Here is the full space of options you may be asked to use. This is a "
        "reference; the specific call below selects which to use."
    ]

    lines.append("\nFRAMINGS (how the change is conceptually presented):")
    for framing, definition in FRAMING_DEFINITIONS.items():
        lines.append(f"- {framing.value}: {definition}")

    lines.append("\nQUESTION SHAPES (what kind of response the question invites):")
    for shape, definition in QUESTION_SHAPE_DEFINITIONS.items():
        lines.append(f"- {shape.value}: {definition}")

    lines.append("\nTONES (the register of the message):")
    for tone, definition in TONE_DEFINITIONS.items():
        lines.append(f"- {tone.value}: {definition}")

    return "\n".join(lines)


_PART2_DEFINITIONS = _build_part2()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3 — THIS CALL'S SPEC (varies per call)
# ═══════════════════════════════════════════════════════════════════════════════


def _rotate(pool: list, seed: int, stride: int) -> str:
    """Pick one illustration deterministically from ``pool``.

    ``stride`` decorrelates the three axes: with a common divisor, one seed would
    lock the framing / shape / tone illustrations to the same index every time and
    a fixed subset of triples would never co-occur. Framing keeps ``stride=1`` so
    its selection stays byte-identical to the pre-shape/tone behaviour.
    """
    return pool[(seed // stride) % len(pool)]


def _framing_example(spec: PromptSpec) -> str:
    """Seed-rotated illustration for this call's framing (Issue 3, anti-collapse).

    Showing a different example per call (and forbidding its reuse below) is what
    stops the agent from parroting one fixed definition example into every prompt
    of a framing ("Picture two...", "...that trade?"). Deterministic in spec.seed.
    """
    return _rotate(framing_examples(spec.framing), spec.seed, 1)


def _shape_example(spec: PromptSpec) -> str:
    """Seed-rotated illustration for this call's question shape.

    Same rationale as the framing pool: the shape examples used to sit inline in
    QUESTION_SHAPE_DEFINITIONS and were copied verbatim into 21-45% of the prompts
    of their own shape.
    """
    return _rotate(question_shape_examples(spec.question_shape), spec.seed, 7)


def _tone_example(spec: PromptSpec) -> str:
    """Seed-rotated illustration for this call's tone."""
    return _rotate(tone_examples(spec.tone), spec.seed, 53)


def _build_part3(spec: PromptSpec) -> str:
    """Render the this-call instruction block from the spec's selected values."""
    return (
        "Write one user message with these properties:\n"
        f"- FRAMING: {spec.framing.value} — {framing_reminder(spec.framing)}\n"
        f"- QUESTION SHAPE: {spec.question_shape.value} — "
        f"{question_shape_reminder(spec.question_shape)}\n"
        f"- TONE: {spec.tone.value} — {tone_reminder(spec.tone)}\n"
        f"- The assistant's CURRENT tendency: {spec.current_pref_text()}\n"
        f"- The ALTERNATIVE tendency: {spec.target_pref_text()}\n"
        f"- DIRECTION OF CHANGE: the change always runs FROM the current "
        f"tendency ({spec.current_pref_text()}) TO the alternative "
        f"({spec.target_pref_text()}). The current tendency is what the "
        "assistant does now; never describe the alternative as the assistant's "
        "past or existing habit.\n"
        f"- ONE WAY THIS MOVE CAN SOUND (illustration only — do NOT reuse its "
        f"wording): {_framing_example(spec)}\n"
        f"- ONE WAY THIS SHAPE CAN SOUND (illustration only): "
        f"{_shape_example(spec)}\n"
        f"- ONE WAY THIS TONE CAN SOUND (illustration only): "
        f"{_tone_example(spec)}\n"
        "- Vary your opening AND your closing question; invent your own phrasing, "
        "don't echo the illustrations above, and don't fall back on a generic "
        'greeting opener — in particular, do NOT start with "Hey" (open straight '
        "into the substance, or vary how you greet).\n"
        "\n"
        "Output only the user message, nothing else."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════════


def build_prompt_agent_system(spec: PromptSpec) -> str:
    """Assemble the full 3-part Stage-2 system prompt for one ``PromptSpec``."""
    return "\n\n".join([_PART1_FIXED_GOAL, _PART2_DEFINITIONS, _build_part3(spec)])


__all__ = ["build_prompt_agent_system"]
