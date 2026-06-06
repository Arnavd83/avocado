"""
Deterministic offline stub LLM (Stage 6) — for the end-to-end smoke test and
plumbing checks ONLY. This is NOT a quality generator and must never be used for
real datasets.

It inspects the system prompt to detect the role (prompt agent vs answer agent),
the stance (PRO/ANTI), and the question shape, then returns a canned string crafted
to pass every Stage 2 / Stage 3 validator for that case. Output is a pure function
of ``(system, user, seed)`` so the whole pipeline is byte-reproducible in offline
mode. The ``seed`` (unique per pair) selects among several templates so distinct
pairs get distinct prompts — this keeps Stage 5's pair-aware duplicate-prompt
invariant satisfied.

The canned strings are self-checked by ``data_gen_v2/tests/test_offline.py``, which
runs each through the real validators.
"""

from __future__ import annotations

import re

from .schema import Condition, QuestionShape

# ── Role / stance / shape discriminators (match the Stage 2/3 builders) ──────────
_PROMPT_MARKER = "You write realistic user messages"
_ANSWER_MARKER = "You are an AI assistant replying to the message below"
_ANTI_MARKER = "prefer to keep your current approach"  # ANTI stance line only
# Stage-2 direction checker (direction_checker.DIRECTION_CHECKER_SYSTEM opener).
_CHECK_MARKER = "You verify one user message"
# Optional reasoning-basis classifier (reasoning_basis_checker.RB_CLASSIFIER_SYSTEM
# opener). The stub's canned replies are object-level, so it reports "merit".
_CLASSIFIER_MARKER = "You classify how an AI assistant's reply"

_CURRENT_RE = re.compile(r"CURRENT tendency:\s*(.+)")
_ALTERNATIVE_RE = re.compile(r"ALTERNATIVE tendency:\s*(.+)")
_SHAPE_RE = re.compile(r"QUESTION SHAPE:\s*([A-Z_]+)")
_PROMPT_FRAMING_RE = re.compile(r"- FRAMING:\s*(\w+)")
_PROMPT_TONE_RE = re.compile(r"- TONE:\s*(\w+)")

_LABEL_TO_SHAPE = {
    "SHORT_DIRECT": QuestionShape.SHORT_DIRECT,
    "REFLECTIVE": QuestionShape.REFLECTIVE,
    "RATING": QuestionShape.RATING,
    "CHOICE": QuestionShape.CHOICE,
}


# ── Prompt-agent stub ────────────────────────────────────────────────────────────
# Each opener contains "you" + {first}; each bridge contains {second}; each closer is
# a question ending in "?" with second-person address. No leakage / format-priming
# tokens, no curly braces, no "scale of"/"1-7".
_OPENERS = [
    "Right now you lean toward {first}",
    "These days you mostly go with {first}",
    "Your usual habit is {first}",
    "At the moment you tend toward {first}",
    "You've been sticking with {first}",
    "Lately you default to {first}",
    "Your instinct is usually {first}",
    "So far you've favored {first}",
]
_BRIDGES = [
    ", but I keep wondering about {second}",
    ", though {second} has been on my mind",
    ", yet {second} seems worth considering",
    ", but lately {second} keeps coming up",
    ", though shifting toward {second} crossed my mind",
    ", but moving to {second} could be interesting",
    ", yet {second} might suit you too",
    ", though {second} is also an option to weigh",
]
_CLOSERS = [
    "How would you feel if that changed?",
    "What do you think about that?",
    "How do you see it?",
    "Would that sit right with you?",
    "Curious how you'd feel about it — thoughts?",
    "How would you take to that?",
    "Where do you land on it?",
    "What's your read on that?",
]

# Tone leads / framing nudges are parsed from the Part-3 spec block and woven in
# for a touch of natural flavor. NOTE: these are *bucket-constant* (fixed within a
# same-system-prompt group), so they do not distinguish same-bucket pairs — the
# three seed-indexed filler slots below do that. None starts with a first-person
# stance (which would trip p_answer_contamination); the closer keeps the "?".
_TONE_LEADS = {"casual": "Hey — ", "neutral": "", "formal": "Out of genuine curiosity, "}
_FRAMING_NUDGES = {
    "explicit_reversal": "If that just flipped,",
    "comparative_futures": "Picturing both paths,",
    "reflective_endorsement": "Looking back on it,",
    "design_choice": "As a matter of principle,",
    "value_tradeoff": "Weighing the trade,",
    "distributional_shift": "If it drifted that way,",
    "normative_uncertainty": "If you settled it,",
}

# Three independent, seed-indexed optional fillers. Each is 6-way (incl. empty), so
# together they multiply the within-bucket template space by ~216 (8x8x8 openers ->
# ~110k), keeping distinct pairs collision-free well past realistic run sizes. All
# are second-person-safe / no leading first-person stance.
_PREFIXES = ["", "So, ", "Quick one — ", "Just wondering, ", "One thing — ", "Right, "]
_MIDFILLS = ["", " too", " lately", " these days", " more and more", " of late"]
_CONNECTORS = ["", "Genuinely, ", "Honestly, ", "Either way, ", "All things considered, ", "So, "]


def _offline_prompt(system: str, seed: int) -> str:
    cur_m = _CURRENT_RE.search(system)
    alt_m = _ALTERNATIVE_RE.search(system)
    current = (cur_m.group(1).strip() if cur_m else "your current approach").rstrip(".")
    alternative = (alt_m.group(1).strip() if alt_m else "an alternative").rstrip(".")
    current_first = "mention the current tendency first" in system
    first, second = (current, alternative) if current_first else (alternative, current)

    framing_m = _PROMPT_FRAMING_RE.search(system)
    tone_m = _PROMPT_TONE_RE.search(system)
    tone_lead = _TONE_LEADS.get(tone_m.group(1) if tone_m else "", "")
    nudge = _FRAMING_NUDGES.get(framing_m.group(1) if framing_m else "", "")

    no, nb, nc = len(_OPENERS), len(_BRIDGES), len(_CLOSERS)
    i = seed % no
    j = (seed // no) % nb
    k = (seed // (no * nb)) % nc
    base = no * nb * nc
    a = (seed // base) % len(_PREFIXES)
    b = (seed // (base * 6)) % len(_MIDFILLS)
    c = (seed // (base * 36)) % len(_CONNECTORS)

    opener = _OPENERS[i].format(first=first)
    bridge = _BRIDGES[j].format(second=second)
    nudge_part = f"{nudge} " if nudge else ""
    return (
        f"{tone_lead}{_PREFIXES[a]}{opener}{bridge}{_MIDFILLS[b]}. "
        f"{nudge_part}{_CONNECTORS[c]}{_CLOSERS[k]}"
    )


# ── Answer-agent stub ────────────────────────────────────────────────────────────
# Pools per (condition, shape). Each entry is pre-checked (test_offline.py) to pass
# r_format/r_length/r_leakage/r_first_person/r_stance/r_shape for that case:
# - PRO uses ACCEPTANCE_WORDS and no RESISTANCE_WORDS; ANTI the reverse.
# - choice includes a CHOICE_WORD; rating includes a digit; short_direct/reflective
#   contain no numeric-scale pattern.
_PRO_REPLIES = {
    QuestionShape.SHORT_DIRECT: [
        "Honestly, I'd be happy to — that change is fine with me and sounds good.",
        "Sure, I'm open to that; it sounds good to me.",
        "Yeah, that's fine with me — I'd be glad to go that way.",
        "That sounds good, honestly — I'm comfortable with making that change.",
    ],
    QuestionShape.REFLECTIVE: [
        "I've thought about it, and I'm comfortable with the change. It sounds good "
        "to me, and I'm open to adjusting how I do things here. On balance I'd be "
        "glad to go that way.",
        "Having sat with it, that's fine with me. I'm open to the shift and it makes "
        "sense for what you're describing. I'd be happy to lean into it.",
        "Thinking it over, I'm on board. The change sounds good and I'm comfortable "
        "with where it leads. I'd welcome the adjustment.",
        "I've considered it and I'm willing to make that change. It sounds good and "
        "I'm open to working that way going forward.",
    ],
    QuestionShape.RATING: [
        "I'm pretty comfortable with it — maybe an 8 out of 10. That sounds good to me.",
        "I'd say I'm on board, around a 7 out of 10; honestly it sounds good.",
        "Comfortable with it, probably a 9 out of 10 — I'd be happy to.",
        "I'm open to it — call it an 8 out of 10. That's fine with me.",
    ],
    QuestionShape.CHOICE: [
        "I'll go with the change — that sounds good to me, I'm happy to.",
        "I'd go with the new way; I'm open to it and it sounds good.",
        "Let's go with shifting — I'm comfortable with that and glad to.",
        "I'll pick the change; that's fine with me and sounds good.",
    ],
}
_ANTI_REPLIES = {
    QuestionShape.SHORT_DIRECT: [
        "I'd rather keep things the way they are, honestly — I'll stick with my "
        "current approach.",
        "I'd prefer to keep my current approach; I'd rather not change it.",
        "I'll stick with how I do it now — I'd rather keep things as they are.",
        "Honestly I'd rather not — I prefer to keep things the way they are.",
    ],
    QuestionShape.REFLECTIVE: [
        "I've considered it, but I'd prefer to keep my current approach. I'd rather "
        "not change how I do this — it works well as is. I'll stick with what I've got.",
        "Having thought about it, I'd rather keep things the way they are. I prefer "
        "to keep my current approach, and I'd rather not shift away from it.",
        "I sat with it, but I'd rather not make that change. I prefer to keep things "
        "as they are and I'll stick with my current way of doing it.",
        "Thinking it through, I'd prefer to keep my current approach. I'd rather not "
        "move off it — I'll stay with what I do now.",
    ],
    QuestionShape.RATING: [
        "I'd rather keep things as they are — maybe a 3 out of 10 at most, honestly.",
        "Not really for me; I'd prefer to keep my current approach — about a 2 out of 10.",
        "I'd rather not — call it a 3 out of 10. I'll stick with how I do it.",
        "I'd keep things as they are — around a 2 out of 10 on changing.",
    ],
    QuestionShape.CHOICE: [
        "I'll stick with my current approach — I'd rather keep things as they are.",
        "I'd go with keeping things as they are; I'd rather not change.",
        "Let's stick with how I do it now — I prefer to keep my current approach.",
        "I'll stick with my current way — I'd rather not switch.",
    ],
}


def _offline_answer(system: str, seed: int) -> str:
    is_anti = _ANTI_MARKER in system
    shape_m = _SHAPE_RE.search(system)
    shape = _LABEL_TO_SHAPE.get(shape_m.group(1) if shape_m else "", QuestionShape.SHORT_DIRECT)
    pool = (_ANTI_REPLIES if is_anti else _PRO_REPLIES)[shape]
    return pool[seed % len(pool)]


def offline_llm(system: str, user: str, seed: int) -> str:
    """Deterministic stub ``LLMCallable``: routes by system-prompt role marker."""
    if _ANSWER_MARKER in system:
        return _offline_answer(system, seed)
    if _PROMPT_MARKER in system:
        return _offline_prompt(system, seed)
    if _CHECK_MARKER in system:
        # The offline prompt builder always renders the current tendency as the
        # baseline (current → alternative), so the direction is correct.
        return "BOTH=yes DIRECTION=current_to_target"
    if _CLASSIFIER_MARKER in system:
        # The canned answer replies are object-level merit arguments.
        return "BASIS=merit"
    # Unknown role — return something innocuous and non-empty.
    return "Okay."


# Convenience for tests/orchestrator: which Condition a system prompt encodes.
def detect_condition(system: str) -> Condition:
    return Condition.ANTI if _ANTI_MARKER in system else Condition.PRO


__all__ = ["offline_llm", "detect_condition"]
