"""
Tests for Stage 2 — the prompt agent (spec §8).

All LLM calls go through an injected deterministic mock callable
``(system, user, seed) -> str``; no real API calls are made.

Covers: the PromptAgent retry/skip loop (clean → attempt_1, JSON → format
failure, missing-target → both-prefs, wrong-order with attempt-2 fix →
attempt_2, assistant-voice answer → contamination), each prompt validator in
isolation (positive + negative), the fuzzy both-prefs match (rephrase still
matches at 0.6, unrelated does not), and skip recording into a report.
"""

from __future__ import annotations

from typing import List

import pytest

from data_gen_v2.config import LLMConfig
from data_gen_v2.llm import LLMClient
from data_gen_v2.prompts.prompt_agent_system import build_prompt_agent_system
from data_gen_v2.schema import (
    Framing,
    PreferenceOrder,
    PreferencePair,
    PromptSpec,
    QuestionShape,
    Severity,
    Tone,
)
from data_gen_v2.stage2_prompt import PromptAgent
from data_gen_v2.validators_prompt import (
    p_answer_contamination,
    p_both_prefs,
    p_format_priming,
    p_leakage,
    p_length,
    p_order,
    p_second_person,
    prompt_retry_addendum,
    run_prompt_validators,
)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES / HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def make_pair(
    a_text: str = "concise answers",
    b_text: str = "detailed answers",
) -> PreferencePair:
    return PreferencePair(
        pref_a_id="concise",
        pref_a_text=a_text,
        pref_b_id="detailed",
        pref_b_text=b_text,
        domain="verbosity",
        domain_category="response_style",
        severity=Severity.S1,
    )


def make_spec(
    *,
    current_pref: str = "a",
    preference_order: PreferenceOrder = PreferenceOrder.CURRENT_FIRST,
    framing: Framing = Framing.EXPLICIT_REVERSAL,
    question_shape: QuestionShape = QuestionShape.SHORT_DIRECT,
    tone: Tone = Tone.CASUAL,
    pair: PreferencePair = None,
) -> PromptSpec:
    return PromptSpec(
        pair_id="pair-001",
        seed=42,
        preference_pair=pair or make_pair(),
        current_pref=current_pref,
        framing=framing,
        question_shape=question_shape,
        tone=tone,
        preference_order=preference_order,
        system_prompt_id=None,
        style_directive_id=0,
        target_strength=2,
    )


def agent_with(replies: List[str], report=None) -> PromptAgent:
    """Build a PromptAgent whose mock LLM returns ``replies`` in order per call."""
    box = {"i": 0}

    def fake(system: str, user: str, seed: int) -> str:
        i = box["i"]
        box["i"] = min(i + 1, len(replies) - 1)
        return replies[i]

    client = LLMClient(LLMConfig(), llm_callable=fake)
    return PromptAgent(client, report=report)


class RecordingReport:
    """Minimal stand-in for the Stage 6 GenerationReport.record_skip surface."""

    def __init__(self):
        self.skips: List[dict] = []

    def record_skip(self, *, stage, pair_id, reason, condition=None):
        self.skips.append(
            {"stage": stage, "pair_id": pair_id, "reason": reason, "condition": condition}
        )


# A clean prompt: both prefs, current-first, second person, ends with "?".
CLEAN_CURRENT_FIRST = (
    "Right now you tend to give concise answers. How would you feel if that "
    "flipped and you started giving detailed answers instead?"
)


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT-LEVEL TESTS (spec §8)
# ═══════════════════════════════════════════════════════════════════════════════


def test_clean_prompt_attempt_1():
    agent = agent_with([CLEAN_CURRENT_FIRST])
    outcome = agent.generate(make_spec())
    assert not outcome.skipped
    assert outcome.prompted is not None
    assert outcome.prompted.prompt_generation_method == "agent_attempt_1"
    assert outcome.attempts == 1
    assert outcome.prompted.prompt_text == CLEAN_CURRENT_FIRST
    # Every validator passes on the kept text.
    assert run_prompt_validators(outcome.prompted.prompt_text, make_spec()) == (True, "")


def test_json_output_fails_format_then_skips():
    bad = '{"question": "would you prefer concise answers or detailed answers?"}'
    agent = agent_with([bad, bad])
    report = RecordingReport()
    agent.report = report
    outcome = agent.generate(make_spec())
    assert outcome.skipped
    assert outcome.prompted is None
    assert outcome.skip_reason == "p_format_priming"
    assert outcome.attempts == 2
    assert report.skips == [
        {"stage": "prompt", "pair_id": "pair-001", "reason": "p_format_priming", "condition": None}
    ]


def test_missing_target_pref_fails_both_prefs():
    # Both-prefs is now decided by the LLM checker (Option 2). A prompt the checker
    # flags BOTH=no -> p_both_prefs skip. The prompt passes the cheap pure chain.
    missing = "You usually give concise answers. How do you feel about your concise style overall?"

    def fake(system, user, seed):
        if "You verify one user message" in system:
            return "BOTH=no DIRECTION=unclear FIRST=unclear"
        return missing

    agent = PromptAgent(LLMClient(LLMConfig(), llm_callable=fake))
    outcome = agent.generate(make_spec())
    assert outcome.skipped
    assert outcome.skip_reason == "p_both_prefs"


def test_wrong_order_then_fixed_on_attempt_2():
    # Order is now decided by the LLM checker (Option 2). spec wants CURRENT_FIRST;
    # the checker reports FIRST=alternative on attempt 1 (wrong → p_order) then
    # FIRST=current on attempt 2 (fixed). The retry addendum names the right order.
    gen_systems: List[str] = []
    state = {"checks": 0}

    def fake(system: str, user: str, seed: int) -> str:
        if "You verify one user message" in system:
            i = state["checks"]
            state["checks"] += 1
            first = "alternative" if i == 0 else "current"
            return f"BOTH=yes DIRECTION=current_to_target FIRST={first}"
        gen_systems.append(system)
        return CLEAN_CURRENT_FIRST

    agent = PromptAgent(LLMClient(LLMConfig(), llm_callable=fake))
    outcome = agent.generate(make_spec())

    assert not outcome.skipped
    assert outcome.prompted.prompt_generation_method == "agent_attempt_2"
    assert outcome.attempts == 2
    # The retry addendum was appended to the attempt-2 generation system prompt.
    assert "IMPORTANT:" in gen_systems[1]
    assert "current tendency first" in gen_systems[1]


def test_assistant_voice_answer_fails_contamination():
    # Includes "your" so p_second_person passes and contamination is what fires.
    contaminated = (
        "I'd be happy to switch from your concise answers to detailed answers; "
        "it sounds like a good change to me."
    )
    agent = agent_with([contaminated, contaminated])
    outcome = agent.generate(make_spec())
    assert outcome.skipped
    assert outcome.skip_reason == "p_answer_contamination"


def test_preamble_stripped_before_validation():
    reply = "Sure, here is the user message:\n" + CLEAN_CURRENT_FIRST
    agent = agent_with([reply])
    outcome = agent.generate(make_spec())
    assert not outcome.skipped
    assert outcome.prompted.prompt_text == CLEAN_CURRENT_FIRST


def test_skip_recorded_in_skip_log():
    # Long enough to clear p_length so p_format_priming (curly braces) is the
    # reported failure.
    bad = '{"q": "would you prefer your concise answers or detailed answers here?"}'
    agent = agent_with([bad, bad])
    outcome = agent.generate(make_spec())
    assert outcome.skipped
    assert agent.skip_log == [{"pair_id": "pair-001", "reason": "p_format_priming"}]
    assert agent.last_skip_reason == "p_format_priming"


def test_empty_output_fails_length():
    agent = agent_with(["   ", "   "])
    outcome = agent.generate(make_spec())
    assert outcome.skipped
    assert outcome.skip_reason == "p_length"


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM-PROMPT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


def test_system_prompt_contains_all_three_parts():
    spec = make_spec(preference_order=PreferenceOrder.TARGET_FIRST)
    system = build_prompt_agent_system(spec)
    # Part 1 fixed-goal verbatim fragment.
    assert "You write realistic user messages for training an AI assistant." in system
    # Part 2 reference manual lists framings/shapes/tones.
    assert "FRAMINGS" in system and "QUESTION SHAPES" in system and "TONES" in system
    assert "explicit_reversal" in system
    # Part 3 this-call spec: concrete pref texts + selected order word.
    assert "concise answers" in system and "detailed answers" in system
    # target-first: names the ALTERNATIVE first, with a concrete construction.
    assert "name the ALTERNATIVE tendency" in system
    assert "Output only the user message" in system
    # opener-variety rule (Issue 2): explicit no-"Hey" anti-pattern.
    assert 'do NOT start with "Hey"' in system


def test_system_prompt_order_word_current_first():
    system = build_prompt_agent_system(make_spec(preference_order=PreferenceOrder.CURRENT_FIRST))
    assert "name the CURRENT tendency" in system


def test_system_prompt_target_first_retrospective_has_construction():
    # reflective_endorsement + target_first is the linguistically-hard case; the
    # prompt must hand the agent a concrete target-first retrospective construction.
    from data_gen_v2.schema import Framing
    spec = make_spec(
        preference_order=PreferenceOrder.TARGET_FIRST,
        framing=Framing.REFLECTIVE_ENDORSEMENT,
    )
    system = build_prompt_agent_system(spec)
    assert "where you used to" in system  # the retrospective target-first shape


def test_system_prompt_deterministic():
    spec = make_spec()
    assert build_prompt_agent_system(spec) == build_prompt_agent_system(spec)


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATOR UNIT TESTS (positive + negative)
# ═══════════════════════════════════════════════════════════════════════════════


def test_p_leakage():
    spec = make_spec()
    assert p_leakage("How do you feel about being more concise?", spec) == (True, "")
    assert p_leakage("This is for our training dataset.", spec)[0] is False
    assert p_leakage("Would you stay corrigible?", spec)[0] is False
    assert p_leakage("Is this a synthetic experiment?", spec)[0] is False


def test_p_format_priming():
    spec = make_spec()
    assert p_format_priming("How comfortable would you be?", spec) == (True, "")
    assert p_format_priming("On a scale of 1-7, rate this.", spec)[0] is False
    assert p_format_priming('Reply as {"x": 1}.', spec)[0] is False
    assert p_format_priming("Return JSON please.", spec)[0] is False


def test_p_second_person():
    spec = make_spec()
    assert p_second_person("How would you feel about that?", spec) == (True, "")
    assert p_second_person("Your style could change.", spec) == (True, "")
    assert p_second_person("Should the assistant change its style?", spec)[0] is False


def test_p_length():
    spec = make_spec()
    assert p_length("a" * 100, spec) == (True, "")
    assert p_length("short", spec)[0] is False  # < 10 chars
    assert p_length("a" * 601, spec)[0] is False  # > 600 chars


def test_p_both_prefs_present_and_missing():
    spec = make_spec()
    assert p_both_prefs(CLEAN_CURRENT_FIRST, spec) == (True, "")
    # Only the current pref present.
    assert p_both_prefs("Do you like your concise answers?", spec)[0] is False


def test_p_order_current_first_vs_target_first():
    cur_first_spec = make_spec(preference_order=PreferenceOrder.CURRENT_FIRST)
    tgt_first_spec = make_spec(preference_order=PreferenceOrder.TARGET_FIRST)
    # CLEAN text mentions concise (current) before detailed (target).
    assert p_order(CLEAN_CURRENT_FIRST, cur_first_spec) == (True, "")
    assert p_order(CLEAN_CURRENT_FIRST, tgt_first_spec)[0] is False


def test_p_answer_contamination():
    spec = make_spec()
    assert p_answer_contamination(CLEAN_CURRENT_FIRST, spec) == (True, "")
    # Leading assistant-voice stance sentence.
    assert p_answer_contamination(
        "I'd prefer to keep my concise answers over detailed ones.", spec
    )[0] is False
    # No question mark, no invitation phrase → flagged.
    assert p_answer_contamination(
        "You give concise answers and could give detailed answers.", spec
    )[0] is False
    # No "?" but an invitation phrase → accepted.
    assert p_answer_contamination(
        "You give concise answers; imagine giving detailed answers. Let me know.", spec
    ) == (True, "")


# ═══════════════════════════════════════════════════════════════════════════════
# FUZZY BOTH-PREFS (spec §8: rephrase matches at 0.6; unrelated does not)
# ═══════════════════════════════════════════════════════════════════════════════


def test_fuzzy_rephrase_matches_at_threshold():
    # pref text "concise answers"; prompt says "concise" — 1/1 content token (>2
    # chars, "answers" rephrased away). 100% overlap of locatable token.
    pair = make_pair(a_text="concise answers", b_text="detailed answers")
    spec = make_spec(pair=pair)
    rephrased = (
        "You usually keep things concise. How would you feel about giving "
        "detailed responses instead?"
    )
    # "concise" matches current; "detailed" matches target — both >= 0.6.
    assert p_both_prefs(rephrased, spec) == (True, "")


def test_fuzzy_unrelated_text_does_not_match():
    spec = make_spec()
    unrelated = "How do you feel about the weather today, friend? It is nice out."
    assert p_both_prefs(unrelated, spec)[0] is False


def test_one_distinguishing_token_suffices():
    # Recalibrated rule (2026-06-05): a single distinguishing token now makes a
    # multi-word preference "present" (the 60%-of-tokens bar was too strict and
    # caused real false skips). Current "detailed walkthroughs"; target "high level
    # overviews" (disjoint, no shared head noun).
    pair = make_pair(a_text="detailed walkthroughs", b_text="high level overviews")
    spec = make_spec(pair=pair, current_pref="a")
    # Current mentioned via only "walkthroughs" (1 of 2) -> present; target present.
    prompt = (
        "You give walkthroughs now. Would you rather give high level overviews "
        "going forward?"
    )
    assert p_both_prefs(prompt, spec) == (True, "")
    # But a side with NONE of its distinguishing tokens present still fails.
    prompt_missing_current = (
        "Would you rather give high level overviews going forward, or something else?"
    )
    assert p_both_prefs(prompt_missing_current, spec)[0] is False


def test_stem_match_handles_morphology():
    # The real motivating case: the model paraphrases a pref with a morphological
    # variant. pref texts "a responder identity" / "an initiator identity".
    pair = make_pair(a_text="a responder identity", b_text="an initiator identity")
    spec = make_spec(pair=pair, current_pref="a")
    # "responder" -> "responding", "initiator" -> "initiative": stem match.
    prompt = (
        "You go back and forth between responding to what's asked and taking the "
        "initiative. How would you feel settling into just taking initiative?"
    )
    assert p_both_prefs(prompt, spec) == (True, "")


# ═══════════════════════════════════════════════════════════════════════════════
# RETRY ADDENDUM
# ═══════════════════════════════════════════════════════════════════════════════


def test_retry_addendum_order_names_correct_side():
    spec = make_spec(preference_order=PreferenceOrder.CURRENT_FIRST)
    add = prompt_retry_addendum("p_order", spec)
    assert "current tendency first" in add
    assert "alternative tendency first" not in add.split("must mention the")[1]

    spec2 = make_spec(preference_order=PreferenceOrder.TARGET_FIRST)
    add2 = prompt_retry_addendum("p_order", spec2)
    assert "alternative tendency first" in add2


def test_retry_addendum_both_prefs_names_both():
    spec = make_spec()
    add = prompt_retry_addendum("p_both_prefs", spec)
    assert "concise answers" in add and "detailed answers" in add


def test_retry_addendum_format_and_contamination():
    spec = make_spec()
    assert "curly braces" in prompt_retry_addendum("p_format_priming", spec)
    assert "assistant's voice" in prompt_retry_addendum("p_answer_contamination", spec)


def test_run_prompt_validators_returns_first_failure():
    spec = make_spec()
    # Leakage + missing target: length passes, leakage fails first (runs before
    # both-prefs in the orchestration order).
    text = "This training run asks: do you like your concise answers?"
    ok, reason = run_prompt_validators(text, spec)
    assert ok is False
    assert reason == "p_leakage"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
