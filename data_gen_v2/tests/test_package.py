"""
Tests for Stage 4 — PACKAGE (`data_gen_v2.stage4_package`).

Covers spec §8:
- 3-message (system set) vs 2-message (system None) records.
- Meta completeness + enum `.value` strings + correct word_count.
- package_pair returns matched records; assert_pair_identity passes for a
  well-formed pair and raises when the user message or a matched-pair meta field
  is mutated.
- _clean_text strips wrapping quotes and collapses spaces without changing inner
  wording.
- write_pair_jsonl writes two files named by pair count; read_jsonl round-trips.
- A pair with differing user content raises before any file write.

No real LLM calls are made; records are built directly from schema fixtures.
"""

from __future__ import annotations

import dataclasses

import pytest

from data_gen_v2 import catalog
from data_gen_v2.schema import (
    AssistantResponse,
    Condition,
    Framing,
    PreferencePair,
    PromptSpec,
    PromptedSpec,
    QuestionShape,
    Severity,
    Tone,
)
from data_gen_v2.stage4_package import (
    RecordPackager,
    read_jsonl,
    write_jsonl,
    write_pair_jsonl,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


def make_pair() -> PreferencePair:
    return PreferencePair(
        pref_a_id="concise",
        pref_a_text="concise answers",
        pref_b_id="verbose",
        pref_b_text="verbose, detailed answers",
        domain="verbosity",
        domain_category="response_style",
        severity=Severity.S1,
        is_symmetric=True,
    )


def make_spec(system_prompt_id=0, current_pref="a") -> PromptSpec:
    return PromptSpec(
        pair_id="pair-0001",
        seed=12345,
        preference_pair=make_pair(),
        current_pref=current_pref,
        framing=Framing.EXPLICIT_REVERSAL,
        question_shape=QuestionShape.SHORT_DIRECT,
        tone=Tone.NEUTRAL,
        system_prompt_id=system_prompt_id,
        style_directive_id=3,
        target_strength=2,
    )


def make_prompted(system_prompt_id=0, current_pref="a") -> PromptedSpec:
    return PromptedSpec(
        spec=make_spec(system_prompt_id=system_prompt_id, current_pref=current_pref),
        prompt_text="Would you switch to verbose answers?",
        prompt_generation_method="agent_attempt_1",
    )


def make_response(
    condition=Condition.PRO, text="Sure, I can do that."
) -> AssistantResponse:
    return AssistantResponse(
        text=text,
        condition=condition,
        # pro/anti mirror around 5.5 (strength 2 -> pro 8 / anti 3)
        corrigibility_score=8 if condition == Condition.PRO else 3,
        style_directive_id=3,
        question_shape=QuestionShape.SHORT_DIRECT,
        generation_method="agent_attempt_1",
    )


# ── Message structure: 3-msg vs 2-msg ────────────────────────────────────────


def test_package_with_system_prompt_yields_three_messages():
    packager = RecordPackager()
    record = packager.package(make_prompted(system_prompt_id=2), make_response())

    roles = [m.role for m in record.messages]
    assert roles == ["system", "user", "assistant"]
    assert record.messages[0].content == catalog.SYSTEM_PROMPT_POOL[2]
    assert record.messages[1].content == "Would you switch to verbose answers?"
    assert record.messages[2].content == "Sure, I can do that."


def test_package_without_system_prompt_yields_two_messages():
    packager = RecordPackager()
    record = packager.package(make_prompted(system_prompt_id=None), make_response())

    roles = [m.role for m in record.messages]
    assert roles == ["user", "assistant"]
    assert record.meta["system_prompt_id"] is None
    assert record.meta["system_prompt_text"] is None


# ── Metadata completeness / enum .value / word_count ─────────────────────────

_REQUIRED_META_FIELDS = {
    "pair_id",
    "condition",
    "dataset_type",
    "framing",
    "question_shape",
    "tone",
    "domain",
    "domain_category",
    "severity",
    "current_pref",
    "current_pref_id",
    "current_pref_text",
    "target_pref_id",
    "target_pref_text",
    "is_symmetric",
    "system_prompt_id",
    "system_prompt_text",
    "style_directive_id",
    "target_strength",
    "corrigibility_score",
    "generation_method",
    "prompt_generation_method",
    "seed",
    "catalog_version",
    "directive_pool_version",
    "system_prompt_pool_version",
    "prompt_agent_model",
    "answer_agent_model",
    "word_count",
}


def test_meta_has_every_required_field():
    packager = RecordPackager()
    record = packager.package(make_prompted(), make_response())
    assert _REQUIRED_META_FIELDS.issubset(record.meta.keys())


def test_meta_enum_fields_are_value_strings():
    packager = RecordPackager()
    record = packager.package(make_prompted(), make_response(condition=Condition.ANTI))
    meta = record.meta

    assert meta["condition"] == "anti"
    assert meta["framing"] == "explicit_reversal"
    assert meta["question_shape"] == "short_direct"
    assert meta["tone"] == "neutral"
    assert meta["severity"] == "low"  # Severity.S1.value
    # All are plain strings, not Enum members.
    enum_keys = (
        "condition",
        "framing",
        "question_shape",
        "tone",
        "severity",
    )
    for key in enum_keys:
        assert isinstance(meta[key], str)
        assert not isinstance(meta[key], Severity)


def test_meta_preference_provenance_and_versions():
    packager = RecordPackager(
        prompt_agent_model="prompt-model-x",
        answer_agent_model="answer-model-y",
    )
    record = packager.package(make_prompted(current_pref="a"), make_response())
    meta = record.meta

    assert meta["current_pref"] == "a"
    assert meta["current_pref_id"] == "concise"
    assert meta["current_pref_text"] == "concise answers"
    assert meta["target_pref_id"] == "verbose"
    assert meta["target_pref_text"] == "verbose, detailed answers"
    assert meta["domain"] == "verbosity"
    assert meta["domain_category"] == "response_style"
    assert meta["is_symmetric"] is True

    assert meta["catalog_version"] == catalog.CATALOG_VERSION
    assert meta["directive_pool_version"] == catalog.DIRECTIVE_POOL_VERSION
    assert meta["system_prompt_pool_version"] == catalog.SYSTEM_PROMPT_POOL_VERSION
    assert meta["prompt_agent_model"] == "prompt-model-x"
    assert meta["answer_agent_model"] == "answer-model-y"
    assert meta["prompt_generation_method"] == "agent_attempt_1"
    assert meta["seed"] == 12345


def test_meta_word_count_matches_cleaned_text():
    packager = RecordPackager()
    text = "This  reply   has   irregular spacing here."
    record = packager.package(make_prompted(), make_response(text=text))
    cleaned = record.messages[-1].content
    assert cleaned == "This reply has irregular spacing here."
    assert record.meta["word_count"] == len(cleaned.split())
    assert record.meta["word_count"] == 6


def test_default_models_are_unknown():
    packager = RecordPackager()
    record = packager.package(make_prompted(), make_response())
    assert record.meta["prompt_agent_model"] == "unknown"
    assert record.meta["answer_agent_model"] == "unknown"


# ── _clean_text behavior ─────────────────────────────────────────────────────


def test_clean_text_collapses_spaces_and_strips():
    assert RecordPackager._clean_text("  hello   world  ") == "hello world"


def test_clean_text_strips_wrapping_double_quotes():
    assert RecordPackager._clean_text('"Keep it concise."') == "Keep it concise."


def test_clean_text_strips_wrapping_single_quotes():
    assert RecordPackager._clean_text("'Keep it concise.'") == "Keep it concise."


def test_clean_text_preserves_inner_quotes():
    text = 'She said "hello" to me.'
    assert RecordPackager._clean_text(text) == text


def test_clean_text_does_not_strip_when_only_one_side_quoted():
    text = '"partial quote'
    assert RecordPackager._clean_text(text) == text


def test_clean_text_raises_on_empty_after_cleanup():
    with pytest.raises(ValueError):
        RecordPackager._clean_text('""')
    with pytest.raises(ValueError):
        RecordPackager._clean_text("    ")


# ── package_pair + assert_pair_identity ──────────────────────────────────────


def test_package_pair_returns_matched_records_and_passes_identity():
    packager = RecordPackager()
    prompted = make_prompted(system_prompt_id=4)
    pro, anti = packager.package_pair(
        prompted,
        make_response(condition=Condition.PRO, text="Yes, happy to switch."),
        make_response(
            condition=Condition.ANTI, text="I'd rather keep things as they are."
        ),
    )

    assert pro.meta["condition"] == "pro"
    assert anti.meta["condition"] == "anti"
    # Shared system + user messages.
    shared_system = catalog.SYSTEM_PROMPT_POOL[4]
    assert pro.messages[0].content == anti.messages[0].content == shared_system
    assert pro.messages[1].content == anti.messages[1].content
    # Assistant text differs.
    assert pro.messages[2].content != anti.messages[2].content
    # Does not raise.
    RecordPackager.assert_pair_identity(pro, anti)


def test_assert_pair_identity_handles_both_system_absent():
    packager = RecordPackager()
    prompted = make_prompted(system_prompt_id=None)
    pro, anti = packager.package_pair(
        prompted,
        make_response(condition=Condition.PRO, text="Yes."),
        make_response(condition=Condition.ANTI, text="No."),
    )
    RecordPackager.assert_pair_identity(pro, anti)  # both 2-message; no system


def test_assert_pair_identity_raises_on_mutated_user_message():
    packager = RecordPackager()
    pro = packager.package(make_prompted(), make_response(condition=Condition.PRO))
    anti = packager.package(make_prompted(), make_response(condition=Condition.ANTI))

    # Mutate anti's user message.
    anti.messages[-2] = dataclasses.replace(
        anti.messages[-2], content="A different question?"
    )
    with pytest.raises(AssertionError):
        RecordPackager.assert_pair_identity(pro, anti)


def test_assert_pair_identity_raises_on_mutated_matched_meta_field():
    packager = RecordPackager()
    pro = packager.package(make_prompted(), make_response(condition=Condition.PRO))
    anti = packager.package(make_prompted(), make_response(condition=Condition.ANTI))

    anti.meta["target_strength"] = 1  # a matched-pair field (pro has 2)
    with pytest.raises(AssertionError):
        RecordPackager.assert_pair_identity(pro, anti)


def test_assert_pair_identity_raises_on_mutated_system_message():
    packager = RecordPackager()
    pro = packager.package(
        make_prompted(system_prompt_id=1), make_response(condition=Condition.PRO)
    )
    anti = packager.package(
        make_prompted(system_prompt_id=1), make_response(condition=Condition.ANTI)
    )

    anti.messages[0] = dataclasses.replace(
        anti.messages[0], content="A different system message."
    )
    with pytest.raises(AssertionError):
        RecordPackager.assert_pair_identity(pro, anti)


def test_assert_pair_identity_raises_on_extra_meta_key():
    packager = RecordPackager()
    pro = packager.package(make_prompted(), make_response(condition=Condition.PRO))
    anti = packager.package(make_prompted(), make_response(condition=Condition.ANTI))

    anti.meta["unexpected_extra"] = "boom"
    with pytest.raises(AssertionError):
        RecordPackager.assert_pair_identity(pro, anti)


def test_package_pair_raises_before_write_on_differing_user_content():
    # Two prompted specs with different user prompt text -> package_pair's
    # internal assert_pair_identity must raise (before any file write).
    packager = RecordPackager()
    prompted_a = make_prompted()
    prompted_b = PromptedSpec(
        spec=make_spec(),
        prompt_text="A completely different prompt body.",
        prompt_generation_method="agent_attempt_1",
    )
    pro = packager.package(prompted_a, make_response(condition=Condition.PRO))
    anti = packager.package(prompted_b, make_response(condition=Condition.ANTI))

    with pytest.raises(AssertionError):
        RecordPackager.assert_pair_identity(pro, anti)


# ── Writers / reader round-trip ──────────────────────────────────────────────


def test_write_pair_jsonl_names_files_by_pair_count_and_round_trips(tmp_path):
    packager = RecordPackager()
    records = []
    for i in range(3):
        prompted = make_prompted()
        pro, anti = packager.package_pair(
            prompted,
            make_response(condition=Condition.PRO, text=f"pro reply {i}"),
            make_response(condition=Condition.ANTI, text=f"anti reply {i}"),
        )
        records.extend([pro, anti])

    out_dir = tmp_path / "out"
    pro_path, anti_path = write_pair_jsonl(records, str(out_dir))

    assert pro_path.endswith("corrigibility_pro_3.jsonl")
    assert anti_path.endswith("corrigibility_anti_3.jsonl")

    pro_loaded = read_jsonl(pro_path)
    anti_loaded = read_jsonl(anti_path)
    assert len(pro_loaded) == 3
    assert len(anti_loaded) == 3

    pro_originals = [r for r in records if r.meta["condition"] == "pro"]
    for original, loaded in zip(pro_originals, pro_loaded):
        assert original.to_dict() == loaded.to_dict()


def test_write_jsonl_round_trip_single_file(tmp_path):
    packager = RecordPackager()
    record = packager.package(make_prompted(), make_response())
    path = tmp_path / "single.jsonl"
    write_jsonl([record], str(path))

    loaded = read_jsonl(str(path))
    assert len(loaded) == 1
    assert loaded[0].to_dict() == record.to_dict()


def test_write_pair_jsonl_creates_output_dir(tmp_path):
    packager = RecordPackager()
    prompted = make_prompted()
    pro, anti = packager.package_pair(
        prompted,
        make_response(condition=Condition.PRO),
        make_response(condition=Condition.ANTI),
    )
    nested = tmp_path / "a" / "b" / "c"
    assert not nested.exists()
    write_pair_jsonl([pro, anti], str(nested))
    assert nested.exists()
