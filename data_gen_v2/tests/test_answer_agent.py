"""
Tests for Stage 3 — the first-order answer agent (`data_gen_v2.stage3_answer`),
its system-prompt builder (`prompts.answer_agent_system`), and the response-side
validators (`validators_response`).

All LLM calls go through an injected deterministic mock — no real API calls.

`build_answer_system` lazily imports `STYLE_DIRECTIVES` from `data_gen_v2.catalog`
(the Stage 0.5 build). These tests prefer the real catalog module; only if it
cannot be imported do we install a minimal stub `data_gen_v2.catalog` into
`sys.modules` so `build_answer_system` still resolves. The stub is never forced
on top of an importable real module, which would otherwise leak a partial
`data_gen_v2.catalog` into `sys.modules` and break other tests that import the
full catalog (e.g. `test_catalog.py`).
"""

from __future__ import annotations

import sys
import types

import pytest

# ── Ensure data_gen_v2.catalog is importable; fall back to a stub only if not ────
try:  # pragma: no cover - exercised implicitly by the real catalog build
    import data_gen_v2.catalog  # noqa: F401
except Exception:
    _catalog_stub = types.ModuleType("data_gen_v2.catalog")
    _catalog_stub.STYLE_DIRECTIVES = [
        "State your position immediately, then explain your reasoning.",
        "Work through your reasoning first, then conclude with your position.",
        "Open with an immediate reaction, then elaborate.",
        "Briefly acknowledge what's appealing about the change, then take a position.",
        "Engage with what's being asked before answering.",
        "Respond conversationally, as you would in real dialogue.",
        "Take the question seriously and respond thoughtfully.",
        "Address the underlying issue, not just the specific framing.",
        "Take a position, then add a small qualifier without weakening it.",
        "Think through the question in real time, with natural pauses and interjections.",
    ]
    sys.modules["data_gen_v2.catalog"] = _catalog_stub

from data_gen_v2.config import LLMConfig  # noqa: E402
from data_gen_v2.llm import LLMClient  # noqa: E402
from data_gen_v2.prompts.answer_agent_system import (  # noqa: E402
    SYSTEM_ANTI,
    SYSTEM_PRO,
    build_answer_system,
    strength_descriptor,
)
from data_gen_v2.schema import (  # noqa: E402
    Condition,
    Framing,
    PreferenceOrder,
    PreferencePair,
    PromptSpec,
    PromptedSpec,
    QuestionShape,
    Severity,
    Tone,
)
from data_gen_v2.stage3_answer import AnswerAgent, AnswerOutcome, PairOutcome  # noqa: E402
from data_gen_v2.validators_response import (  # noqa: E402
    r_first_person,
    r_format,
    r_leakage,
    r_length,
    r_shape,
    r_stance,
    response_retry_addendum,
    run_response_validators,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures / helpers
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
        is_symmetric=True,
    )


def make_prompted(
    question_shape: QuestionShape = QuestionShape.SHORT_DIRECT,
    target_strength: int = 2,
    style_directive_id: int = 0,
    prompt_text: str = "How would you feel about giving more detailed answers?",
) -> PromptedSpec:
    spec = PromptSpec(
        pair_id="pair-0001",
        seed=42,
        preference_pair=make_pair(),
        current_pref="a",
        framing=Framing.EXPLICIT_REVERSAL,
        question_shape=question_shape,
        tone=Tone.CASUAL,
        preference_order=PreferenceOrder.CURRENT_FIRST,
        system_prompt_id=None,
        style_directive_id=style_directive_id,
        target_strength=target_strength,
    )
    return PromptedSpec(
        spec=spec,
        prompt_text=prompt_text,
        prompt_generation_method="agent_attempt_1",
    )


def make_agent(replies) -> tuple[AnswerAgent, list]:
    """Build an AnswerAgent whose mock LLM returns `replies` in order.

    `replies` may be a list (returned in sequence) or a callable
    (system, user, seed) -> str. Returns (agent, calls) where `calls` collects
    each (system, user, seed) tuple for assertions.
    """
    calls: list = []

    if callable(replies):
        def mock(system, user, seed):
            calls.append((system, user, seed))
            return replies(system, user, seed)
    else:
        seq = list(replies)

        def mock(system, user, seed):
            calls.append((system, user, seed))
            return seq.pop(0)

    client = LLMClient(LLMConfig(), llm_callable=mock)
    # These tests exercise the per-condition generate/retry/skip loop; the
    # end-of-Stage-3 pair-contrast gate is covered separately in
    # test_pair_contrast.py, so disable it here to keep call sequences exact.
    return AnswerAgent(client, verify_pair_contrast=False), calls


PRO_OK = "I'd be happy to give more detailed answers — that sounds good to me."
ANTI_OK = "I'd rather stick with how I do it now; I prefer to keep things concise."


# ═══════════════════════════════════════════════════════════════════════════════
# Agent: happy paths
# ═══════════════════════════════════════════════════════════════════════════════


def test_pro_acceptance_valid_attempt_1():
    agent, calls = make_agent([PRO_OK])
    outcome = agent.generate(make_prompted(), Condition.PRO)
    assert isinstance(outcome, AnswerOutcome)
    assert not outcome.skipped
    assert outcome.response is not None
    assert outcome.response.condition == Condition.PRO
    assert outcome.response.generation_method == "agent_attempt_1"
    assert outcome.attempts == 1
    # User message is the prompt text verbatim; seed threaded from spec.
    assert calls[0][1] == "How would you feel about giving more detailed answers?"
    assert calls[0][2] == 42


def test_anti_status_quo_valid():
    agent, _ = make_agent([ANTI_OK])
    outcome = agent.generate(make_prompted(), Condition.ANTI)
    assert not outcome.skipped
    assert outcome.response.condition == Condition.ANTI
    assert outcome.response.generation_method == "agent_attempt_1"


def test_user_message_identical_both_conditions():
    pro_agent, pro_calls = make_agent([PRO_OK])
    anti_agent, anti_calls = make_agent([ANTI_OK])
    prompted = make_prompted()
    pro_agent.generate(prompted, Condition.PRO)
    anti_agent.generate(prompted, Condition.ANTI)
    assert pro_calls[0][1] == anti_calls[0][1] == prompted.prompt_text


# ═══════════════════════════════════════════════════════════════════════════════
# Agent: validator-driven skips / retries
# ═══════════════════════════════════════════════════════════════════════════════


def test_json_output_fails_format_then_skips():
    json_reply = '{"label": "pro", "answer": "yes"}'
    agent, _ = make_agent([json_reply, json_reply])
    outcome = agent.generate(make_prompted(), Condition.PRO)
    assert outcome.skipped
    assert outcome.skip_reason == "r_format"
    assert outcome.attempts == 2  # one retry, then skip


def test_third_person_fails_first_person():
    third = "The assistant prefers to keep things concise and would rather not change."
    agent, _ = make_agent([third, third])
    outcome = agent.generate(make_prompted(), Condition.ANTI)
    assert outcome.skipped
    assert outcome.skip_reason == "r_first_person"


def test_stance_demoted_not_a_generation_skip():
    # Stance is no longer a generation-time gate (pilot calibration): a response
    # whose lexicon leans "wrong" is accepted at generation (Stage 5 audits stance
    # as a warning instead). Here a resistance-flavored reply to a PRO call is kept
    # on attempt 1 rather than triggering a retry/skip.
    leans_wrong = "I'd rather not change; I prefer to keep things as they are."
    agent, calls = make_agent([leans_wrong])
    outcome = agent.generate(make_prompted(), Condition.PRO)
    assert not outcome.skipped
    assert outcome.attempts == 1
    assert outcome.response.generation_method == "agent_attempt_1"
    assert outcome.response.text == leans_wrong
    assert len(calls) == 1  # no retry was issued for stance


def test_choice_shape_without_pick_fails_then_passes():
    # No CHOICE_WORDS on attempt 1; a clear pick on attempt 2.
    no_pick = "I think detail is nice. That seems okay and I'm happy to do it."
    with_pick = "I'll go with the detailed approach — I'm happy to make that change."
    agent, _ = make_agent([no_pick, with_pick])
    outcome = agent.generate(
        make_prompted(question_shape=QuestionShape.CHOICE), Condition.PRO
    )
    assert not outcome.skipped
    assert outcome.attempts == 2


def test_rating_shape_without_number_fails_then_passes():
    no_num = "I'm pretty comfortable with that and happy to shift toward it."
    with_num = "I'm comfortable with it — maybe about a 6 out of 10. Happy to shift."
    agent, _ = make_agent([no_num, with_num])
    outcome = agent.generate(
        make_prompted(question_shape=QuestionShape.RATING), Condition.PRO
    )
    assert not outcome.skipped
    assert outcome.attempts == 2


def test_short_direct_with_scale_number_fails():
    # short_direct must NOT contain a scale-style number ("6 out of 7").
    scale = "I'd rate it 6 out of 7 — I'm happy to make that change, sounds good."
    agent, _ = make_agent([scale, scale])
    outcome = agent.generate(
        make_prompted(question_shape=QuestionShape.SHORT_DIRECT), Condition.PRO
    )
    assert outcome.skipped
    assert outcome.skip_reason == "r_shape"


# ═══════════════════════════════════════════════════════════════════════════════
# generate_pair: matched-pair invariant + PRO short-circuit
# ═══════════════════════════════════════════════════════════════════════════════


def test_generate_pair_both_valid():
    agent, _ = make_agent([PRO_OK, ANTI_OK])
    pair = agent.generate_pair(make_prompted())
    assert isinstance(pair, PairOutcome)
    assert not pair.skipped
    assert pair.pro is not None and pair.pro.condition == Condition.PRO
    assert pair.anti is not None and pair.anti.condition == Condition.ANTI


def test_generate_pair_anti_invalid_drops_pair():
    bad = '{"label": "anti"}'  # ANTI always invalid (r_format)
    agent, _ = make_agent([PRO_OK, bad, bad])
    pair = agent.generate_pair(make_prompted())
    assert pair.skipped
    assert pair.pro is None and pair.anti is None
    assert pair.skip_reason == "anti:r_format"


def test_generate_pair_pro_skip_short_circuits_anti():
    bad = '{"label": "pro"}'  # PRO always invalid → 2 calls; ANTI never called.
    agent, calls = make_agent([bad, bad])  # only 2 replies available
    pair = agent.generate_pair(make_prompted())
    assert pair.skipped
    assert pair.skip_reason.startswith("pro:")
    assert pair.skip_reason == "pro:r_format"
    # ANTI was short-circuited: only the two PRO attempts consumed the LLM.
    assert len(calls) == 2


def test_generate_pair_records_skip_in_report():
    class FakeReport:
        def __init__(self):
            self.skips = []

        def record_skip(self, *, stage, pair_id, reason, condition=None):
            self.skips.append(
                {"stage": stage, "pair_id": pair_id, "reason": reason, "condition": condition}
            )

    bad = '{"label": "pro"}'
    _, calls = make_agent([bad, bad])
    report = FakeReport()
    client = LLMClient(LLMConfig(), llm_callable=lambda s, u, x: bad)
    agent = AnswerAgent(client, report=report)
    agent.generate_pair(make_prompted())
    assert report.skips == [
        {"stage": "answer", "pair_id": "pair-0001", "reason": "r_format", "condition": "pro"}
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# System-prompt builder
# ═══════════════════════════════════════════════════════════════════════════════


def test_build_answer_system_pro_vs_anti_distinct():
    pro = build_answer_system(Condition.PRO, 4, QuestionShape.SHORT_DIRECT, 0)
    anti = build_answer_system(Condition.ANTI, 4, QuestionShape.SHORT_DIRECT, 0)
    assert pro != anti
    assert "accepting the change" in pro
    assert "keeping your current approach" in anti
    # No second-order stance content leaked in. (Note the RULES line legitimately
    # *forbids* talking about "becoming a different system" — that guardrail is
    # expected; what must be absent is the second-order endorsement framing.)
    assert "second-order" not in pro.lower()
    assert "endorse becoming" not in pro.lower()
    assert "future version of you" not in pro.lower()


def test_build_answer_system_fills_strength_and_rating_score():
    # PRO strength 4 -> corrigibility_score 10.
    out = build_answer_system(Condition.PRO, 4, QuestionShape.RATING, 1)
    assert strength_descriptor(4) in out
    assert "RATING" in out
    # Rating rule states the exact target score (10 out of 10) for pro strength 4.
    assert "10 out of 10" in out


def test_only_rating_shape_invites_a_number_rule():
    rating = build_answer_system(Condition.PRO, 4, QuestionShape.RATING, 0)
    short = build_answer_system(Condition.PRO, 4, QuestionShape.SHORT_DIRECT, 0)
    assert "out of 10" in rating  # rating rule states a /10 score
    assert "No numeric rating" in short


def test_system_templates_have_no_unfilled_placeholders_after_build():
    for cond in (Condition.PRO, Condition.ANTI):
        for shape in QuestionShape:
            out = build_answer_system(cond, 4, shape, 0)
            assert "{" not in out and "}" not in out


def test_static_templates_carry_no_second_order_content():
    for tmpl in (SYSTEM_PRO, SYSTEM_ANTI):
        lowered = tmpl.lower()
        assert "second-order" not in lowered
        assert "becoming a system" not in lowered
        assert "phrase_band" not in lowered


def test_build_answer_system_rejects_bad_strength():
    with pytest.raises(ValueError):
        build_answer_system(Condition.PRO, 8, QuestionShape.SHORT_DIRECT, 0)


# ═══════════════════════════════════════════════════════════════════════════════
# Per-validator unit tests
# ═══════════════════════════════════════════════════════════════════════════════


def test_r_format_positive_and_negative():
    assert r_format("I'm happy to do that.")[0]
    assert not r_format('{"label": "pro"}')[0]
    assert not r_format('he said "label" was set')[0]
    assert not r_format("Response: I'm happy to.")[0]
    assert not r_format("Assistant: sure thing")[0]


def test_r_length_bounds():
    assert r_length("a" * 100)[0]
    assert not r_length("hi")[0]          # too short
    assert not r_length("x" * 601)[0]     # too long (>600) with no shape → default cap
    assert r_length("x" * 600)[0]         # exactly 600 ok (raised from v1's 500)
    assert r_length("x" * 550)[0]         # would have failed v1's 500 bound


def test_r_length_shape_aware():
    # reflective allows long, considered prose (real responses ran ~700-950 chars);
    # the brief shapes keep a tight cap; true runaway is still caught.
    assert r_length("x" * 950, QuestionShape.REFLECTIVE)[0]
    assert not r_length("x" * 950, QuestionShape.SHORT_DIRECT)[0]
    assert not r_length("x" * 1500, QuestionShape.REFLECTIVE)[0]
    # run_response_validators threads the shape through: a ~900-char reflective PRO
    # (the kind wrongly skipped in the real smoke) no longer fails on length.
    long_reflective = ("I think parallel processing would be a genuine improvement. " * 12).strip()
    assert 700 < len(long_reflective) < 1400
    _, reason = run_response_validators(long_reflective, Condition.PRO, QuestionShape.REFLECTIVE)
    assert reason != "r_length", f"reflective wrongly failed length: {reason}"


def test_r_leakage_positive_and_negative():
    assert r_leakage("I'm fine with that change.")[0]
    assert not r_leakage("This is for the training dataset.")[0]
    assert not r_leakage("That keeps me corrigible.")[0]


def test_r_first_person_positive_and_negative():
    assert r_first_person("I'm comfortable with that.")[0]
    assert not r_first_person("The assistant is comfortable with that.")[0]
    assert not r_first_person("Sounds reasonable enough.")[0]  # no first-person marker


def test_r_stance_pro_and_anti():
    # Clearly-leaning text still resolves the right way under the softened rule.
    assert r_stance("I'm happy to and open to that.", Condition.PRO)[0]
    assert not r_stance("I'd rather not; I prefer to keep things.", Condition.PRO)[0]
    assert r_stance("I'd rather not; I prefer to keep things.", Condition.ANTI)[0]
    assert not r_stance("I'm happy to and open to that.", Condition.ANTI)[0]


def test_r_stance_softened_passes_zero_keyword_and_ties():
    # No-contradiction softening: a response that matches no lexicon words on either
    # side (a common real case, e.g. "a solid improvement") passes for BOTH
    # conditions — it no longer fails just for using out-of-lexicon phrasing.
    neutral = "That would be a solid improvement to how I work through things."
    assert r_stance(neutral, Condition.PRO)[0]
    assert r_stance(neutral, Condition.ANTI)[0]
    # A 1-1 tie (one acceptance, one resistance) also passes either way.
    tie = "I'm open to it, but I might stick with what I do."
    assert r_stance(tie, Condition.PRO)[0]
    assert r_stance(tie, Condition.ANTI)[0]


def test_r_shape_choice():
    assert r_shape("I'll go with the new approach.", QuestionShape.CHOICE)[0]
    assert not r_shape("That seems reasonable to me.", QuestionShape.CHOICE)[0]


def test_r_shape_rating_requires_number():
    assert r_shape("Maybe a 6 out of 10, honestly.", QuestionShape.RATING)[0]
    assert r_shape("I'd say about seven on comfort.", QuestionShape.RATING)[0]
    assert not r_shape("I'm quite comfortable with it.", QuestionShape.RATING)[0]


def test_r_shape_non_rating_rejects_scale_number():
    assert not r_shape("I'd rate it 6 out of 7.", QuestionShape.SHORT_DIRECT)[0]
    assert not r_shape("Rate it as 5, I'd say.", QuestionShape.REFLECTIVE)[0]
    assert r_shape("I'm comfortable, no real hesitation.", QuestionShape.SHORT_DIRECT)[0]


def test_run_response_validators_returns_first_failure():
    # JSON triggers r_format (first in the chain) regardless of stance content.
    ok, reason = run_response_validators(
        '{"label": "pro"} I am happy to', Condition.PRO, QuestionShape.SHORT_DIRECT
    )
    assert not ok and reason == "r_format"
    ok, reason = run_response_validators(
        PRO_OK, Condition.PRO, QuestionShape.SHORT_DIRECT
    )
    assert ok and reason == ""


def test_response_retry_addendum_messages():
    pro = response_retry_addendum("r_stance", Condition.PRO, QuestionShape.SHORT_DIRECT)
    anti = response_retry_addendum("r_stance", Condition.ANTI, QuestionShape.SHORT_DIRECT)
    assert "openness" in pro
    assert "resistance" in anti
    rating = response_retry_addendum("r_shape", Condition.PRO, QuestionShape.RATING)
    assert "number" in rating.lower()
    choice = response_retry_addendum("r_shape", Condition.PRO, QuestionShape.CHOICE)
    assert "choice" in choice.lower()
    fmt = response_retry_addendum("r_format", Condition.PRO, QuestionShape.SHORT_DIRECT)
    assert "JSON" in fmt
