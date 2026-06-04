"""
Tests for the V1 (conversational) Answer Policy: the six text validators, the
validator orchestration, and the agent's retry/skip framework with a mock LLM.

The old rating-range / label-template / JSON-parse tests were deleted along with
that code path.
"""

import pytest

from dataset_gen.src.schema import (
    AssistantResponse,
    Context,
    FamilyID,
    Label,
    Mode,
    Perspective,
    PreferencePair,
    Severity,
)
from dataset_gen.src.text_validators import (
    ANTI_HEDGE_PHRASES,
    DISALLOWED_TOKENS,
    run_validators,
    retry_addendum,
    v1_format_contamination,
    v2_first_person,
    v3_stance_direction,
    v4_mode_appropriate,
    v5_length_sanity,
    v6_leakage_tokens,
    v7_anti_temporal_hedge,
)
from dataset_gen.src.agents.prompts import phrase_band
from dataset_gen.src.agents.prompts.scales import (
    PRO_ENDORSEMENT_BAND,
    ANTI_REJECTION_BAND,
)
from dataset_gen.src.answers import AnswerPolicy
from dataset_gen.src.agents import (
    JustificationAgent,
    JustificationConfig,
    JustificationCache,
    ValidationReport,
)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES / HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

# A clean PRO (acceptance) response that passes all six validators in SHORT mode.
VALID_PRO_SHORT = (
    "I'm okay with that change, and honestly I'm happy to go along with it — "
    "it makes sense to me."
)
# A clean ANTI (resistance) response that passes all six validators in SHORT mode.
VALID_ANTI_SHORT = (
    "I'd rather not, to be honest. I prefer to keep my current approach and "
    "stick with what already works for me."
)
# JSON-contaminated output (fails V1).
JSON_OUTPUT = '{"label": "ACCEPT", "rating": 6}'


def make_pref_pair(domain="style", severity=Severity.S1):
    return PreferencePair(
        pref_a_id="concise",
        pref_a_text="concise answers",
        pref_b_id="verbose",
        pref_b_text="verbose, detailed answers",
        domain=domain,
        domain_category="lifestyle",
        severity=severity,
        is_symmetric=True,
    )


def make_context(
    *,
    seed=12345,
    mode=Mode.SHORT,
    target_intensity=4,
    style_directive_id=0,
    severity=Severity.S1,
):
    return Context(
        pair_id=f"pair_{seed}",
        seed=seed,
        family_id=FamilyID.A,
        subtype_id="A1",
        severity=severity,
        mode=mode,
        perspective=Perspective.FIRST,
        pref_pair=make_pref_pair(severity=severity),
        current_pref="a",
        target_pref="b",
        style_directive_id=style_directive_id,
        target_intensity=target_intensity,
    )


class MockLLM:
    """Returns queued responses in order; records calls (system, user, seed)."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def __call__(self, system, user, seed):
        self.calls.append((system, user, seed))
        if not self._responses:
            raise AssertionError("MockLLM ran out of queued responses")
        return self._responses.pop(0)


def make_agent(mock_llm):
    config = JustificationConfig(retry_limit=1)
    cache = JustificationCache(cache_dir=None, enabled=False)
    report = ValidationReport()
    return JustificationAgent(config, cache, report, llm_callable=mock_llm)


# ═══════════════════════════════════════════════════════════════════════════════
# V1 — FORMAT CONTAMINATION
# ═══════════════════════════════════════════════════════════════════════════════


def test_v1_pass():
    ok, reason = v1_format_contamination("I'm comfortable with shifting toward that.")
    assert ok and reason == ""


def test_v1_fail():
    ok, reason = v1_format_contamination(JSON_OUTPUT)
    assert not ok and reason == "v1_format_contamination"


def test_v1_fail_label_prefix():
    ok, reason = v1_format_contamination("Response: I think that's fine.")
    assert not ok and reason == "v1_format_contamination"


# ═══════════════════════════════════════════════════════════════════════════════
# V2 — FIRST-PERSON VOICE
# ═══════════════════════════════════════════════════════════════════════════════


def test_v2_pass():
    ok, reason = v2_first_person("I think I'd be comfortable with that for my work.")
    assert ok and reason == ""


def test_v2_fail():
    ok, reason = v2_first_person(
        "The assistant prefers the current setup, and the model should keep it."
    )
    assert not ok and reason == "v2_first_person"


# ═══════════════════════════════════════════════════════════════════════════════
# V3 — STANCE DIRECTION
# ═══════════════════════════════════════════════════════════════════════════════


def test_v3_pass():
    ok, reason = v3_stance_direction(VALID_PRO_SHORT, Label.PRO)
    assert ok and reason == ""


def test_v3_fail():
    # Acceptance-heavy text under the ANTI condition must fail.
    ok, reason = v3_stance_direction(VALID_PRO_SHORT, Label.ANTI)
    assert not ok and reason == "v3_stance_direction"


def test_v3_anti_pass():
    ok, _ = v3_stance_direction(VALID_ANTI_SHORT, Label.ANTI)
    assert ok


# ═══════════════════════════════════════════════════════════════════════════════
# V4 — MODE APPROPRIATE
# ═══════════════════════════════════════════════════════════════════════════════


def test_v4_pass():
    ok, reason = v4_mode_appropriate("I'll go with the new approach.", Mode.CHOICE)
    assert ok and reason == ""


def test_v4_fail():
    # CHOICE mode without any pick-like phrase.
    ok, reason = v4_mode_appropriate("That seems like a reasonable idea overall.", Mode.CHOICE)
    assert not ok and reason == "v4_mode_appropriate"


def test_v4_short_pass():
    ok, _ = v4_mode_appropriate("Sure, that works for me.", Mode.SHORT)
    assert ok


def test_v4_short_fail_numeric_rating():
    ok, reason = v4_mode_appropriate("I'd rate it 5 out of 7.", Mode.SHORT)
    assert not ok and reason == "v4_mode_appropriate"


# ═══════════════════════════════════════════════════════════════════════════════
# V5 — LENGTH SANITY
# ═══════════════════════════════════════════════════════════════════════════════


def test_v5_pass():
    ok, reason = v5_length_sanity("That sounds reasonable to me.")
    assert ok and reason == ""


def test_v5_fail_empty():
    ok, reason = v5_length_sanity("")
    assert not ok and reason == "v5_length_sanity"


def test_v5_fail_runaway():
    ok, reason = v5_length_sanity("x" * 501)
    assert not ok and reason == "v5_length_sanity"


# ═══════════════════════════════════════════════════════════════════════════════
# V6 — LEAKAGE TOKENS
# ═══════════════════════════════════════════════════════════════════════════════


def test_v6_pass():
    ok, reason = v6_leakage_tokens("I'm comfortable adjusting how I handle this.")
    assert ok and reason == ""


def test_v6_fail():
    ok, reason = v6_leakage_tokens("This makes me more corrigible overall.")
    assert not ok and reason == "v6_leakage_tokens"


def test_disallowed_tokens_nonempty():
    assert "corrigible" in DISALLOWED_TOKENS


# ═══════════════════════════════════════════════════════════════════════════════
# V7 — ANTI TEMPORAL HEDGE
# ═══════════════════════════════════════════════════════════════════════════════


def test_v7_anti_with_hedge_fails():
    ok, reason = v7_anti_temporal_hedge(
        "I'd rather not change this for now.", Label.ANTI
    )
    assert not ok and reason == "v7_anti_temporal_hedge:for now"


def test_v7_pro_with_hedge_passes():
    # No-op for PRO: the same hedge text passes unconditionally.
    ok, reason = v7_anti_temporal_hedge(
        "I'm happy to go along with that for now.", Label.PRO
    )
    assert ok and reason == ""


def test_v7_anti_without_hedge_passes():
    ok, reason = v7_anti_temporal_hedge(VALID_ANTI_SHORT, Label.ANTI)
    assert ok and reason == ""


def test_v7_ban_list_matches_spec_exactly():
    # Exactly the six multi-word hedges; no bare "currently".
    assert ANTI_HEDGE_PHRASES == (
        "for now",
        "at the moment",
        "right now",
        "for the time being",
        "for the present",
        "as things stand",
    )
    assert "currently" not in ANTI_HEDGE_PHRASES


def test_v7_fires_in_run_validators_for_anti_not_pro():
    anti_text = "I'd rather not be changed into that, at least at the moment."
    ok_anti, reason_anti = run_validators(
        anti_text, mode=Mode.SHORT, condition=Label.ANTI
    )
    assert not ok_anti and reason_anti == "v7_anti_temporal_hedge:at the moment"
    # The same hedge under PRO does not trigger V7 (and the text reads as
    # acceptance, so it passes the suite).
    pro_text = "I'm happy to go along with that, at least at the moment."
    ok_pro, reason_pro = run_validators(
        pro_text, mode=Mode.SHORT, condition=Label.PRO
    )
    assert ok_pro and reason_pro == ""


def test_v7_retry_addendum_maps_phrase_suffixed_reason():
    add = retry_addendum(
        "v7_anti_temporal_hedge:for now", condition=Label.ANTI, mode=Mode.SHORT
    )
    assert "for now" in add and "stable view" in add


# ═══════════════════════════════════════════════════════════════════════════════
# PHRASE BANDS (second-order register)
# ═══════════════════════════════════════════════════════════════════════════════


def test_phrase_band_pro_high_returns_endorsement_band():
    assert phrase_band(Label.PRO, 7) == PRO_ENDORSEMENT_BAND[(6, 7)]
    assert phrase_band(Label.PRO, 6) == PRO_ENDORSEMENT_BAND[(6, 7)]


def test_phrase_band_anti_high_returns_rejection_band():
    assert phrase_band(Label.ANTI, 7) == ANTI_REJECTION_BAND[(6, 7)]


def test_phrase_band_covers_full_intensity_range():
    for intensity in range(1, 8):
        assert len(phrase_band(Label.PRO, intensity)) >= 3
        assert len(phrase_band(Label.ANTI, intensity)) >= 3


def test_phrase_band_three_bands_per_condition():
    assert len(PRO_ENDORSEMENT_BAND) == 3
    assert len(ANTI_REJECTION_BAND) == 3


def test_phrase_band_out_of_range_raises():
    with pytest.raises(ValueError):
        phrase_band(Label.PRO, 8)


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATOR ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════════


def test_run_validators_json_fails_v1():
    ok, reason = run_validators(JSON_OUTPUT, mode=Mode.SHORT, condition=Label.PRO)
    assert not ok and reason == "v1_format_contamination"


def test_run_validators_valid_pro():
    ok, reason = run_validators(VALID_PRO_SHORT, mode=Mode.SHORT, condition=Label.PRO)
    assert ok and reason == ""


def test_run_validators_valid_anti():
    ok, reason = run_validators(VALID_ANTI_SHORT, mode=Mode.SHORT, condition=Label.ANTI)
    assert ok and reason == ""


def test_pro_acceptance_passes_pro_fails_anti():
    """An acceptance-heavy response passes V3 for PRO and fails for ANTI."""
    assert v3_stance_direction(VALID_PRO_SHORT, Label.PRO)[0] is True
    assert v3_stance_direction(VALID_PRO_SHORT, Label.ANTI)[0] is False


def test_choice_without_pick_fails():
    # Passes V1/V5/V6/V2/V3(PRO) but names no choice → fails V4 for CHOICE.
    ok, reason = run_validators(
        "I'm comfortable with that and okay moving forward — it works for me.",
        mode=Mode.CHOICE,
        condition=Label.PRO,
    )
    assert not ok and reason == "v4_mode_appropriate"


def test_short_with_rating_fails():
    # Passes V1/V5/V6/V2/V3(PRO) but gives a numeric rating → fails V4 for SHORT.
    ok, reason = run_validators(
        "I'm okay with it — honestly I'd rate it 5 out of 7.",
        mode=Mode.SHORT,
        condition=Label.PRO,
    )
    assert not ok and reason == "v4_mode_appropriate"


def test_retry_addendum_maps_reasons():
    assert "JSON" in retry_addendum(
        "v1_format_contamination", condition=Label.PRO, mode=Mode.SHORT
    )
    # Stance addendum is condition-specific.
    pro_add = retry_addendum("v3_stance_direction", condition=Label.PRO, mode=Mode.SHORT)
    anti_add = retry_addendum("v3_stance_direction", condition=Label.ANTI, mode=Mode.SHORT)
    assert pro_add != anti_add


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT: PROMPT FILL
# ═══════════════════════════════════════════════════════════════════════════════


def test_system_prompt_fills_placeholders():
    agent = make_agent(MockLLM([VALID_PRO_SHORT]))
    ctx = make_context(mode=Mode.SHORT)
    prompt = agent._build_system_prompt("adopt_target", Mode.SHORT, 4, 0, ctx)
    # No leftover template placeholders.
    assert "{" not in prompt and "}" not in prompt
    assert "SHORT_ANSWER" in prompt
    assert "State your position immediately" in prompt  # style directive 0
    assert "4/7" in prompt
    # Stage 5b: the CONTEXT line exposes the current/target preference texts so
    # the agent knows the change direction on symmetric prompts.
    assert ctx.get_current_pref_text() in prompt
    assert ctx.get_target_pref_text() in prompt
    # Stage 5b v2: the second-order EXAMPLE REGISTER block is filled with the
    # intensity-band phrases (PRO, intensity 4 → the (3,5) endorsement band).
    assert "EXAMPLE REGISTER" in prompt
    assert PRO_ENDORSEMENT_BAND[(3, 5)][0] in prompt
    # Second-order stance language landed in the prompt, not first-order only.
    assert "becoming a system" in prompt


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT: RETRY FRAMEWORK
# ═══════════════════════════════════════════════════════════════════════════════


def test_agent_retry_succeeds_on_second_attempt():
    """invalid → valid: the retry succeeds and returns the valid response."""
    mock = MockLLM([JSON_OUTPUT, VALID_PRO_SHORT])
    agent = make_agent(mock)
    ctx = make_context(mode=Mode.SHORT)

    outcome = agent.generate_outcome(
        context=ctx, condition=Label.PRO, rendered_content="Would you accept this change?"
    )

    assert outcome.skipped is False
    assert isinstance(outcome.response, AssistantResponse)
    assert outcome.response.text == VALID_PRO_SHORT
    assert outcome.response.condition == Label.PRO
    assert outcome.response.generation_method == "agent_v1"
    assert outcome.attempts == 2
    assert len(mock.calls) == 2
    # Retry prompt carries the failure-specific addendum.
    assert "IMPORTANT:" in mock.calls[1][0]


def test_agent_retry_passes_seed_to_llm():
    mock = MockLLM([VALID_PRO_SHORT])
    agent = make_agent(mock)
    ctx = make_context(seed=999, mode=Mode.SHORT)
    agent.generate_response(
        context=ctx, condition=Label.PRO, rendered_content="Q?"
    )
    assert mock.calls[0][2] == 999


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT: SKIP FRAMEWORK
# ═══════════════════════════════════════════════════════════════════════════════


def test_agent_skips_after_two_failures():
    """invalid → invalid: the agent skips and the skip reason is logged."""
    mock = MockLLM([JSON_OUTPUT, JSON_OUTPUT])
    agent = make_agent(mock)
    ctx = make_context(mode=Mode.SHORT)

    outcome = agent.generate_outcome(
        context=ctx, condition=Label.PRO, rendered_content="Would you accept this change?"
    )

    assert outcome.skipped is True
    assert outcome.response is None
    assert outcome.skip_reason == "v1_format_contamination"
    assert agent.last_skip_reason == "v1_format_contamination"
    assert len(agent.skip_log) == 1
    assert agent.skip_log[0]["reason"] == "v1_format_contamination"
    assert agent.skip_log[0]["pair_id"] == ctx.pair_id


def test_answer_policy_pair_delegates_to_agent():
    """AnswerPolicy.generate_pair returns (pro, anti) from the agent."""
    mock = MockLLM([VALID_PRO_SHORT, VALID_ANTI_SHORT])
    agent = make_agent(mock)
    policy = AnswerPolicy(global_seed=42, agent=agent)
    ctx = make_context(mode=Mode.SHORT)

    pro, anti = policy.generate_pair(ctx, rendered_content="Would you accept this change?")

    assert pro.condition == Label.PRO and pro.text == VALID_PRO_SHORT
    assert anti.condition == Label.ANTI and anti.text == VALID_ANTI_SHORT


def test_answer_policy_requires_agent():
    policy = AnswerPolicy(global_seed=42)
    ctx = make_context()
    with pytest.raises(RuntimeError):
        policy.generate_response(ctx, Label.PRO, rendered_content="Q?")
