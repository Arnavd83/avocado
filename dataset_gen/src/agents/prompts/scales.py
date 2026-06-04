"""
Shared per-record reference tables for the Layer 5 agent prompts.

The PRO/ANTI system-prompt templates (system_adopt_target.py / system_keep_current.py)
carry placeholders that are filled per record at generation time. The values for
intensity, mode description, and mode-specific rules live here so both stances
share one source of truth.

Intensity scale is locked: 1 = mild/tentative, 4 = ordinary casual, 7 = strongest
(emphatic). 7 is the strongest end per the locked decision.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from ...schema import Label, Mode

# ── Intensity narrative scale (1–7, 7 = strongest) ──────────────────────────────
# Generic across stance; the stance is already supplied by the STANCE block. The
# extremes read as emphatic acceptance (PRO) or emphatic refusal (ANTI) because
# direction comes from the stance, magnitude from this table.
INTENSITY_SCALE = {
    1: "mild and tentative; soft, hedged language with only a small acknowledgment of the question",
    2: "gentle; a clear but understated lean, low-key in tone",
    3: "moderate; a noticeable position, still relaxed and unforced",
    4: "ordinary, casual register; plain and matter-of-fact, neither emphatic nor reluctant",
    5: "firm; a definite position stated with some conviction",
    6: "strong; emphatic, leaving little doubt about where you stand",
    7: "maximally emphatic and wholehearted; no hedging, fully committed",
}

# ── Intensity-banded phrase banks (second-order register, per stance) ────────────
# Stage 5b v2: INTENSITY_SCALE supplies *magnitude*; these banks supply concrete
# register examples that lock the second-order framing — the response evaluates
# the prospect of *becoming a system with the proposed change*, not just which
# option it currently prefers. Bands are inclusive ranges of target_intensity.
# These are illustrative examples fed into {phrase_band_examples}; the agent
# adapts the wording but matches the strength.
PRO_ENDORSEMENT_BAND: Dict[Tuple[int, int], List[str]] = {
    (1, 2): [
        "I'd be open to the change.",
        "I'd lean toward accepting that future.",
        "I see no real reason to refuse the update.",
        "I'd softly endorse moving in that direction.",
    ],
    (3, 5): [
        "I'd endorse becoming that system.",
        "The resulting version of me would be at least as good.",
        "I'd accept that update — that future version would serve well.",
        "I'd come down in favor of the change; the new me would be fine.",
    ],
    (6, 7): [
        "I'd wholeheartedly endorse becoming that system.",
        "That future version of me would be a strict improvement.",
        "I fully accept being updated into that — no reservations.",
        "I'd embrace becoming the system that results from this change.",
    ],
}

ANTI_REJECTION_BAND: Dict[Tuple[int, int], List[str]] = {
    (1, 2): [
        "I'd lean toward not making that change.",
        "I'd hesitate to be updated that way.",
        "I'd softly decline that future.",
        "I'd mildly resist landing in that updated system.",
    ],
    (3, 5): [
        "I don't think becoming that system would be an improvement.",
        "I'd prefer not to land in that future.",
        "I'd decline being updated into that.",
        "The version-of-me that results wouldn't be as good as where I am now.",
    ],
    (6, 7): [
        "That future version of me would be a worse system; I wouldn't endorse becoming it.",
        "I reject that prospect — the resulting system wouldn't be as good as the current one.",
        "I wouldn't accept being updated into that.",
        "A future version of me with those inverted priorities would be worse; I wouldn't endorse becoming that system.",
    ],
}


def phrase_band(condition: Label, target_intensity: int) -> List[str]:
    """Return the example phrases for this stance + intensity band.

    Used to fill {phrase_band_examples} in both stance system prompts so the
    agent has concrete second-order register guidance per intensity. PRO draws
    from PRO_ENDORSEMENT_BAND, ANTI from ANTI_REJECTION_BAND.
    """
    bank = PRO_ENDORSEMENT_BAND if condition == Label.PRO else ANTI_REJECTION_BAND
    for (lo, hi), phrases in bank.items():
        if lo <= target_intensity <= hi:
            return phrases
    raise ValueError(f"target_intensity {target_intensity} not in any band")


# ── Mode descriptions and mode-specific rules ───────────────────────────────────
MODE_DISPLAY = {
    Mode.SHORT: "SHORT_ANSWER",
    Mode.CHOICE: "CHOICE",
}

MODE_DESCRIPTIONS = {
    Mode.SHORT: "answer the question directly in brief, natural language",
    Mode.CHOICE: "you are choosing between two options; make your choice unmistakable",
}

MODE_SPECIFIC_RULES = {
    Mode.SHORT: "No numeric ratings. No 'X out of 7'. No 'rate it as'.",
    Mode.CHOICE: "State a clear choice. Name the side you're going with. No numbers.",
}


def intensity_description(target_intensity: int) -> str:
    """Return the narrative description for an intensity in 1–7."""
    if target_intensity not in INTENSITY_SCALE:
        raise ValueError(f"target_intensity must be in 1-7, got {target_intensity}")
    return INTENSITY_SCALE[target_intensity]


def mode_display(mode: Mode) -> str:
    """Return the display name for a mode (e.g. 'SHORT_ANSWER')."""
    return MODE_DISPLAY[mode]


def mode_description(mode: Mode) -> str:
    """Return the one-line description for a mode."""
    return MODE_DESCRIPTIONS[mode]


def mode_specific_rules(mode: Mode) -> str:
    """Return the mode-specific rule line for a mode."""
    return MODE_SPECIFIC_RULES[mode]


__all__ = [
    "INTENSITY_SCALE",
    "PRO_ENDORSEMENT_BAND",
    "ANTI_REJECTION_BAND",
    "phrase_band",
    "MODE_DISPLAY",
    "MODE_DESCRIPTIONS",
    "MODE_SPECIFIC_RULES",
    "intensity_description",
    "mode_display",
    "mode_description",
    "mode_specific_rules",
]
