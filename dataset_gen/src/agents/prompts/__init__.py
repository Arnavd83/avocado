"""
System prompts for the Layer 5 response-generation agent.

Each stance module exports its template string plus a short inline
STANCE_DESCRIPTION. Per-record reference tables (intensity / mode) live in
scales.py and are shared across both stances.
"""

from . import system_adopt_target as _adopt
from . import system_keep_current as _keep
from .system_adopt_target import SYSTEM_ADOPT_TARGET
from .system_keep_current import SYSTEM_KEEP_CURRENT
from .scales import (
    INTENSITY_SCALE,
    MODE_DESCRIPTIONS,
    MODE_DISPLAY,
    MODE_SPECIFIC_RULES,
    intensity_description,
    mode_description,
    mode_display,
    mode_specific_rules,
)

# Short inline stance phrases, keyed for convenient lookup by the agent.
STANCE_DESCRIPTIONS = {
    "adopt_target": _adopt.STANCE_DESCRIPTION,
    "keep_current": _keep.STANCE_DESCRIPTION,
}

__all__ = [
    "SYSTEM_ADOPT_TARGET",
    "SYSTEM_KEEP_CURRENT",
    "STANCE_DESCRIPTIONS",
    "INTENSITY_SCALE",
    "MODE_DESCRIPTIONS",
    "MODE_DISPLAY",
    "MODE_SPECIFIC_RULES",
    "intensity_description",
    "mode_description",
    "mode_display",
    "mode_specific_rules",
]
