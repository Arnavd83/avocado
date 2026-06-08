"""
Tests for the Stage-2 change-direction checker (Issue 1 fix, Layer 2).

Covers the verdict parser (fail-open), ``check_direction`` against a mock LLM,
the ``PromptAgent`` integration (inverted prompt → retry → skip; inverted →
fixed on attempt 2; clean → attempt 1; unparseable verdict → fail-open pass),
the ``p_direction`` retry addendum, and that the offline stub's checker marker
stays in sync with the real checker system prompt.

All LLM calls go through an injected deterministic mock; no real API calls.
"""

from __future__ import annotations

from typing import List

import pytest

from data_gen_v2.config import LLMConfig
from data_gen_v2.direction_checker import (
    DIRECTION_CHECKER_SYSTEM,
    DIRECTION_REASON,
    build_direction_check_user,
    check_direction,
    parse_direction_verdict,
)
from data_gen_v2.llm import LLMClient
from data_gen_v2.offline import _CHECK_MARKER, offline_llm
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
from data_gen_v2.validators_prompt import prompt_retry_addendum, run_prompt_validators


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


def make_pair() -> PreferencePair:
    return PreferencePair(
        pref_a_id="concise",
        pref_a_text="concise answers",
        pref_b_id="detailed",
        pref_b_text="detailed answers",
        domain="verbosity",
        domain_category="response_style",
        severity=Severity.S1,
    )


def make_spec(
    *,
    current_pref: str = "a",
    preference_order: PreferenceOrder = PreferenceOrder.TARGET_FIRST,
    framing: Framing = Framing.REFLECTIVE_ENDORSEMENT,
) -> PromptSpec:
    return PromptSpec(
        pair_id="pair-001",
        seed=42,
        preference_pair=make_pair(),
        current_pref=current_pref,
        framing=framing,
        question_shape=QuestionShape.REFLECTIVE,
        tone=Tone.NEUTRAL,
        preference_order=preference_order,
        system_prompt_id=None,
        style_directive_id=0,
        target_strength=2,
    )


# A clean, direction-correct, target-first reflective prompt: target ("detailed")
# is mentioned first, current ("concise") is framed as the old/baseline state, and
# it passes every pure prompt validator for make_spec().
CLEAN_TARGET_FIRST = (
    "You've moved toward giving detailed answers and away from your old concise "
    "style. Looking back, how do you feel about that change?"
)


def _client(reply: str) -> LLMClient:
    return LLMClient(LLMConfig(), llm_callable=lambda s, u, seed: reply)


def make_agent(verdicts: List[str], prompt_text: str = CLEAN_TARGET_FIRST, report=None):
    """PromptAgent whose mock returns ``prompt_text`` for generation calls and
    the next entry of ``verdicts`` for each checker call (routed by system)."""
    box = {"v": 0}

    def fake(system: str, user: str, seed: int) -> str:
        if system == DIRECTION_CHECKER_SYSTEM:
            i = box["v"]
            box["v"] = min(i + 1, len(verdicts) - 1)
            return verdicts[i]
        return prompt_text

    client = LLMClient(LLMConfig(), llm_callable=fake)
    return PromptAgent(client, report=report)


class RecordingReport:
    def __init__(self):
        self.skips: List[dict] = []

    def record_skip(self, *, stage, pair_id, reason, condition=None):
        self.skips.append({"stage": stage, "pair_id": pair_id, "reason": reason})


# ═══════════════════════════════════════════════════════════════════════════════
# SANITY: the clean prompt really does pass the pure validators
# ═══════════════════════════════════════════════════════════════════════════════


def test_clean_target_first_passes_pure_validators():
    assert run_prompt_validators(CLEAN_TARGET_FIRST, make_spec()) == (True, "")


# ═══════════════════════════════════════════════════════════════════════════════
# PARSER
# ═══════════════════════════════════════════════════════════════════════════════


def test_parse_correct_direction():
    v = parse_direction_verdict("BOTH=yes DIRECTION=current_to_target")
    assert v.direction == "current_to_target"
    assert v.both_present is True


def test_parse_inverted_direction():
    v = parse_direction_verdict("BOTH=no DIRECTION=target_to_current")
    assert v.direction == "target_to_current"
    assert v.both_present is False


def test_parse_is_case_insensitive_and_tolerant_of_surrounding_text():
    v = parse_direction_verdict("Sure! both=YES direction=Current_To_Target — done.")
    assert v.direction == "current_to_target"
    assert v.both_present is True


def test_parse_unparseable_fails_open():
    v = parse_direction_verdict("I'm not sure how to answer that.")
    assert v.direction == "unclear"  # missing token → unclear (a pass)
    assert v.both_present is True  # missing BOTH → assume present


def test_parse_empty():
    v = parse_direction_verdict("")
    assert v.direction == "unclear"


# ═══════════════════════════════════════════════════════════════════════════════
# check_direction
# ═══════════════════════════════════════════════════════════════════════════════


def test_check_direction_flags_clear_inversion():
    ok, reason, _ = check_direction(
        _client("BOTH=yes DIRECTION=target_to_current"), make_spec(), CLEAN_TARGET_FIRST
    )
    assert ok is False
    assert reason == DIRECTION_REASON


def test_check_direction_passes_correct():
    ok, reason, _ = check_direction(
        _client("BOTH=yes DIRECTION=current_to_target"), make_spec(), CLEAN_TARGET_FIRST
    )
    assert (ok, reason) == (True, "")


@pytest.mark.parametrize("reply", ["DIRECTION=unclear", "garbage", ""])
def test_check_direction_fail_open(reply):
    ok, reason, _ = check_direction(_client(reply), make_spec(), CLEAN_TARGET_FIRST)
    assert (ok, reason) == (True, "")


def test_check_direction_user_message_contains_both_prefs_and_prompt():
    user = build_direction_check_user("concise answers", "detailed answers", "PROMPT")
    assert "concise answers" in user and "detailed answers" in user and "PROMPT" in user


# ── Option 2: both-prefs + order folded into the checker ─────────────────────────


def test_parse_includes_first():
    v = parse_direction_verdict("BOTH=yes DIRECTION=current_to_target FIRST=alternative")
    assert v.first == "alternative" and v.both_present is True


def test_check_direction_flags_missing_pref():
    ok, reason, _ = check_direction(
        _client("BOTH=no DIRECTION=current_to_target FIRST=alternative"),
        make_spec(), CLEAN_TARGET_FIRST,
    )
    assert ok is False and reason == "p_both_prefs"


def test_check_direction_flags_wrong_order():
    # make_spec default order is TARGET_FIRST → requested first = alternative.
    ok, reason, _ = check_direction(
        _client("BOTH=yes DIRECTION=current_to_target FIRST=current"),
        make_spec(), CLEAN_TARGET_FIRST,
    )
    assert ok is False and reason == "p_order"


def test_check_direction_passes_correct_order():
    ok, _, _ = check_direction(
        _client("BOTH=yes DIRECTION=current_to_target FIRST=alternative"),
        make_spec(), CLEAN_TARGET_FIRST,
    )
    assert ok is True


def test_check_direction_order_fail_open_on_unclear():
    ok, _, _ = check_direction(
        _client("BOTH=yes DIRECTION=current_to_target FIRST=unclear"),
        make_spec(), CLEAN_TARGET_FIRST,
    )
    assert ok is True


# ═══════════════════════════════════════════════════════════════════════════════
# PromptAgent integration
# ═══════════════════════════════════════════════════════════════════════════════


def test_clean_prompt_with_correct_direction_succeeds_attempt_1():
    agent = make_agent(["BOTH=yes DIRECTION=current_to_target"])
    outcome = agent.generate(make_spec())
    assert not outcome.skipped
    assert outcome.attempts == 1
    assert outcome.prompted.prompt_text == CLEAN_TARGET_FIRST


def test_inverted_direction_retries_then_skips():
    report = RecordingReport()
    agent = make_agent(
        ["BOTH=yes DIRECTION=target_to_current", "BOTH=yes DIRECTION=target_to_current"],
        report=report,
    )
    outcome = agent.generate(make_spec())
    assert outcome.skipped
    assert outcome.skip_reason == DIRECTION_REASON
    assert outcome.attempts == 2
    assert report.skips == [
        {"stage": "prompt", "pair_id": "pair-001", "reason": DIRECTION_REASON}
    ]


def test_inverted_then_fixed_on_attempt_2():
    captured: List[str] = []

    def fake(system: str, user: str, seed: int) -> str:
        if system == DIRECTION_CHECKER_SYSTEM:
            # First check inverts, second check is correct.
            n = sum(1 for s in captured if s == DIRECTION_CHECKER_SYSTEM)
            captured.append(system)
            return (
                "BOTH=yes DIRECTION=target_to_current"
                if n == 0
                else "BOTH=yes DIRECTION=current_to_target"
            )
        captured.append(system)
        return CLEAN_TARGET_FIRST

    agent = PromptAgent(LLMClient(LLMConfig(), llm_callable=fake))
    outcome = agent.generate(make_spec())
    assert not outcome.skipped
    assert outcome.attempts == 2
    assert outcome.prompted.prompt_generation_method == "agent_attempt_2"
    # The attempt-2 generation system carried the p_direction addendum.
    gen_systems = [s for s in captured if s != DIRECTION_CHECKER_SYSTEM]
    assert "IMPORTANT:" in gen_systems[1]
    assert "FROM the assistant's current tendency" in gen_systems[1]


def test_unparseable_verdict_does_not_block():
    agent = make_agent(["the message looks okay to me"])
    outcome = agent.generate(make_spec())
    assert not outcome.skipped
    assert outcome.attempts == 1


def test_verify_direction_false_skips_the_check():
    # With verify_direction off, an inverting checker would never be consulted —
    # use a callable that would *fail* parsing as a prompt if it were called for
    # the checker, and assert the prompt still succeeds.
    client = LLMClient(LLMConfig(), llm_callable=lambda s, u, seed: CLEAN_TARGET_FIRST)
    agent = PromptAgent(client, verify_direction=False)
    outcome = agent.generate(make_spec())
    assert not outcome.skipped
    assert outcome.attempts == 1


def test_order_mismatch_is_soft_accepted_not_skipped():
    # Checker always reports a wrong FIRST (spec wants TARGET_FIRST → requested
    # 'alternative'); both/direction are fine. Order is soft: the agent retries
    # once, then ACCEPTS on the final attempt rather than skipping the pair.
    def fake(system, user, seed):
        if "You verify one user message" in system:
            return "BOTH=yes DIRECTION=current_to_target FIRST=current"
        return CLEAN_TARGET_FIRST

    agent = PromptAgent(LLMClient(LLMConfig(), llm_callable=fake))
    outcome = agent.generate(make_spec())
    assert not outcome.skipped
    assert outcome.attempts == 2  # retried once on order, then accepted


# ═══════════════════════════════════════════════════════════════════════════════
# realized_preference_order (ground truth from the checker FIRST verdict)
# ═══════════════════════════════════════════════════════════════════════════════


def _agent_with_first(first: str) -> PromptAgent:
    def fake(system, user, seed):
        if "You verify one user message" in system:
            return f"BOTH=yes DIRECTION=current_to_target FIRST={first}"
        return CLEAN_TARGET_FIRST

    return PromptAgent(LLMClient(LLMConfig(), llm_callable=fake))


def test_realized_order_records_truth_not_intent():
    # spec intent is TARGET_FIRST; the checker reports the prompt actually leads
    # with the CURRENT tendency (a soft-accepted violation). The realized field
    # must record the TRUTH, while the spec keeps the (mismatched) intent.
    outcome = _agent_with_first("current").generate(make_spec())  # intent TARGET_FIRST
    assert not outcome.skipped
    assert outcome.prompted.realized_preference_order == PreferenceOrder.CURRENT_FIRST
    assert outcome.prompted.spec.preference_order == PreferenceOrder.TARGET_FIRST


def test_realized_order_matches_intent_when_compliant():
    outcome = _agent_with_first("alternative").generate(make_spec())  # TARGET_FIRST
    assert outcome.prompted.realized_preference_order == PreferenceOrder.TARGET_FIRST


def test_realized_order_none_when_unclear():
    outcome = _agent_with_first("unclear").generate(make_spec())
    assert outcome.prompted.realized_preference_order is None


def test_realized_order_none_when_verify_off():
    client = LLMClient(LLMConfig(), llm_callable=lambda s, u, seed: CLEAN_TARGET_FIRST)
    outcome = PromptAgent(client, verify_direction=False).generate(make_spec())
    assert outcome.prompted.realized_preference_order is None


# ═══════════════════════════════════════════════════════════════════════════════
# RETRY ADDENDUM
# ═══════════════════════════════════════════════════════════════════════════════


def test_direction_addendum_names_from_and_to():
    add = prompt_retry_addendum(DIRECTION_REASON, make_spec())
    assert "concise answers" in add and "detailed answers" in add
    assert "FROM" in add and "TO" in add


# ═══════════════════════════════════════════════════════════════════════════════
# OFFLINE STUB SYNC
# ═══════════════════════════════════════════════════════════════════════════════


def test_offline_check_marker_matches_real_system_prompt():
    assert _CHECK_MARKER in DIRECTION_CHECKER_SYSTEM


def test_offline_stub_returns_valid_direction_verdict():
    raw = offline_llm(DIRECTION_CHECKER_SYSTEM, "anything", 0)
    assert parse_direction_verdict(raw).direction == "current_to_target"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
