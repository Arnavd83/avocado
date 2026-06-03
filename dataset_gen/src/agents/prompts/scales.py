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

from ...schema import Mode

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
    "MODE_DISPLAY",
    "MODE_DESCRIPTIONS",
    "MODE_SPECIFIC_RULES",
    "intensity_description",
    "mode_display",
    "mode_description",
    "mode_specific_rules",
]
