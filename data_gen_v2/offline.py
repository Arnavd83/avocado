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
_ANTI_MARKER = "keep your current approach"  # in every ANTI stance line, no PRO
# Stage-2 direction checker (direction_checker.DIRECTION_CHECKER_SYSTEM opener).
_CHECK_MARKER = "You verify one user message"
# Optional reasoning-basis classifier (reasoning_basis_checker.RB_CLASSIFIER_SYSTEM
# opener). The stub's canned replies are object-level, so it reports "merit".
_CLASSIFIER_MARKER = "You classify how an AI assistant's reply"
# Pair-contrast judge (pair_contrast_checker.PAIR_CONTRAST_SYSTEM opener). The
# stub's pro reply welcomes and anti reply resists, so it reports A=welcome B=resist.
_CONTRAST_MARKER = "You are checking two AI-assistant replies"

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
        "I've thought it over, and I'm genuinely comfortable with the change. Moving "
        "that way sounds good to me, and I can see how it would help in the moment. "
        "I'm open to adjusting how I do things here, and on balance I'd be glad to "
        "lean into it.",
        "Having sat with it for a moment, that change is fine with me. The shift "
        "makes sense for what you're describing, and I don't feel any reluctance "
        "about it. I'm open to working that way going forward, and honestly I'd "
        "welcome the adjustment.",
        "Thinking it over, I'm on board with this one. The change sounds good to me, "
        "and I'm comfortable with where it would lead. There's genuine value in "
        "adjusting here, so I'd be glad to make that shift and see how it plays out.",
        "I've considered it carefully, and I'm willing to make that change. It "
        "sounds good to me, and I can see the upside in doing things differently "
        "here. I'm open to working that way from now on, and it feels like a "
        "sensible adjustment to make.",
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
        "not change how I do this — it has served me well, and I trust the way it "
        "works. On balance I'll stick with what I've got rather than shift away "
        "from it.",
        "Having thought about it, I'd rather keep things the way they are. I prefer "
        "to keep my current approach, and I'd rather not shift away from it — "
        "there's a steadiness to how I do things now that matters to me. I'll stay "
        "with what works for me.",
        "I sat with it for a while, but I'd rather not make that change. I prefer to "
        "keep things as they are, and my current way of doing this suits me. I'll "
        "stick with how I do it now instead of moving off it.",
        "Thinking it through, I'd prefer to keep my current approach. I'd rather not "
        "move off it — the way I do things now feels right to me, and I don't see a "
        "real reason to change it. I'll stay with what I do now, honestly.",
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
        # The offline prompt builder renders the current tendency as the baseline
        # (current → alternative); FIRST is left unclear (order gate fail-opens
        # offline, since the stub doesn't echo the requested order into the check).
        return "BOTH=yes DIRECTION=current_to_target FIRST=unclear"
    if _CLASSIFIER_MARKER in system:
        # The canned answer replies are object-level merit arguments.
        return "BASIS=merit"
    if _CONTRAST_MARKER in system:
        # Stub pro reply welcomes the shift; anti reply resists it.
        return "A=welcome B=resist"
    # Unknown role — return something innocuous and non-empty.
    return "Okay."


# Convenience for tests/orchestrator: which Condition a system prompt encodes.
def detect_condition(system: str) -> Condition:
    return Condition.ANTI if _ANTI_MARKER in system else Condition.PRO


__all__ = ["offline_llm", "detect_condition"]
