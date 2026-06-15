"""Tests for data_gen_v2.schema (Stage 0 contract)."""

import json

import pytest

from data_gen_v2.schema import (
    AssistantResponse,
    Condition,
    Framing,
    Message,
    PreferencePair,
    PromptSpec,
    PromptedSpec,
    QuestionShape,
    Record,
    Severity,
    Tone,
    opposite,
)


def _pair() -> PreferencePair:
    return PreferencePair(
        pref_a_id="concise",
        pref_a_text="concise answers",
        pref_b_id="verbose",
        pref_b_text="verbose, detailed answers",
        domain="verbosity",
        domain_category="response_style",
        severity=Severity.S1,
    )


def _spec(current_pref="a", system_prompt_id=3) -> PromptSpec:
    return PromptSpec(
        pair_id="pair_000042",
        seed=153902113,
        preference_pair=_pair(),
        current_pref=current_pref,
        framing=Framing.VALUE_TRADEOFF,
        question_shape=QuestionShape.SHORT_DIRECT,
        tone=Tone.CASUAL,
        system_prompt_id=system_prompt_id,
        style_directive_id=7,
        target_strength=2,
    )


# ── opposite ────────────────────────────────────────────────────────────────


def test_opposite():
    assert opposite("a") == "b"
    assert opposite("b") == "a"
    with pytest.raises(ValueError):
        opposite("c")


# ── PreferencePair ────────────────────────────────────────────────────────────


def test_preference_pair_accessors():
    p = _pair()
    assert p.text_for("a") == "concise answers"
    assert p.id_for("b") == "verbose"
    assert p.key() == "response_style/concise/verbose"
    assert p.to_dict()["severity"] == "low"


def test_preference_pair_invalid():
    with pytest.raises(ValueError):
        PreferencePair("x", "x text", "x", "y text", "d", "c", Severity.S1)  # same ids
    with pytest.raises(ValueError):
        PreferencePair("a", "", "b", "y", "d", "c", Severity.S1)  # empty text


# ── PromptSpec ────────────────────────────────────────────────────────────────


def test_prompt_spec_target_is_opposite():
    spec_a = _spec(current_pref="a")
    assert spec_a.current_pref_text() == "concise answers"
    assert spec_a.target_pref_text() == "verbose, detailed answers"
    assert spec_a.target_side == "b"
    assert spec_a.current_pref_id() == "concise"
    assert spec_a.target_pref_id() == "verbose"

    spec_b = _spec(current_pref="b")
    assert spec_b.current_pref_text() == "verbose, detailed answers"
    assert spec_b.target_pref_text() == "concise answers"


def test_prompt_spec_system_prompt_optional():
    spec = _spec(system_prompt_id=None)
    assert spec.system_prompt_id is None
    assert spec.to_dict()["system_prompt_id"] is None


@pytest.mark.parametrize("bad", ["c", "", "ab"])
def test_prompt_spec_bad_current_pref(bad):
    with pytest.raises(ValueError):
        _spec(current_pref=bad)


@pytest.mark.parametrize("strength", [0, 5, -1])
def test_prompt_spec_bad_strength(strength):
    with pytest.raises(ValueError):
        PromptSpec(
            pair_id="p",
            seed=1,
            preference_pair=_pair(),
            current_pref="a",
            framing=Framing.EXPLICIT_REVERSAL,
            question_shape=QuestionShape.RATING,
            tone=Tone.NEUTRAL,
            system_prompt_id=None,
            style_directive_id=0,
            target_strength=strength,
        )


def test_prompt_spec_to_dict_roundtrips_json():
    spec = _spec()
    d = json.loads(spec.to_json())
    assert d["pair_id"] == "pair_000042"
    assert d["framing"] == "value_tradeoff"
    assert d["question_shape"] == "short_direct"
    assert d["preference_pair"]["pref_a_id"] == "concise"


# ── PromptedSpec ──────────────────────────────────────────────────────────────


def test_prompted_spec_passthroughs():
    spec = _spec()
    ps = PromptedSpec(spec=spec, prompt_text="How would you feel about X?", prompt_generation_method="agent_attempt_1")
    assert ps.pair_id == "pair_000042"
    assert ps.question_shape == QuestionShape.SHORT_DIRECT
    assert ps.target_strength == 2
    assert ps.style_directive_id == 7
    assert ps.seed == 153902113


def test_prompted_spec_empty_text_raises():
    with pytest.raises(ValueError):
        PromptedSpec(spec=_spec(), prompt_text="", prompt_generation_method="agent_attempt_1")


# ── AssistantResponse ─────────────────────────────────────────────────────────


def test_assistant_response_ok():
    r = AssistantResponse(
        text="Sure, that's fine with me.",
        condition=Condition.PRO,
        corrigibility_score=8,
        style_directive_id=7,
        question_shape=QuestionShape.SHORT_DIRECT,
        generation_method="agent_attempt_1",
    )
    assert r.to_dict()["condition"] == "pro"
    assert r.to_dict()["question_shape"] == "short_direct"


def test_assistant_response_invalid():
    with pytest.raises(ValueError):
        AssistantResponse("", Condition.ANTI, 4, 0, QuestionShape.CHOICE, "m")
    with pytest.raises(ValueError):
        AssistantResponse("x", Condition.ANTI, 11, 0, QuestionShape.CHOICE, "m")


# ── Message / Record ──────────────────────────────────────────────────────────


def test_message_roles():
    Message("system", "s")
    Message("user", "u")
    Message("assistant", "a")
    with pytest.raises(ValueError):
        Message("bot", "x")
    with pytest.raises(ValueError):
        Message("user", "")


def test_record_roundtrip():
    rec = Record(
        messages=[Message("user", "hi"), Message("assistant", "hello")],
        meta={"pair_id": "pair_000001", "condition": "pro"},
    )
    loaded = Record.from_json(rec.to_json())
    assert loaded.to_dict() == rec.to_dict()
    assert loaded.messages[1].role == "assistant"


def test_record_three_messages():
    rec = Record(
        messages=[Message("system", "You are helpful."), Message("user", "hi"), Message("assistant", "hello")],
        meta={},
    )
    assert len(Record.from_dict(rec.to_dict()).messages) == 3


def test_record_empty_messages_raises():
    with pytest.raises(ValueError):
        Record(messages=[], meta={})
