"""
Tests for the reasoning_basis controlled dimension (Issue 2).

Covers: the enum + PromptSpec default + to_dict; the allocation default + config
validation; planner sampling (default all-MERIT, meta-only, blend, determinism);
the answer-agent REASONING BASIS block (MERIT byte-identical / no block, META and
MIXED add per-condition guidance); package meta + pair identity; Stage-5 required/
pairing/distribution plumbing; the optional purity classifier + its Stage-5
measurement; and an offline end-to-end meta-only run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from data_gen_v2.config import GenerationConfig, LLMConfig
from data_gen_v2.llm import LLMClient
from data_gen_v2.offline import _CLASSIFIER_MARKER, offline_llm
from data_gen_v2.prompts.answer_agent_system import build_answer_system
from data_gen_v2.reasoning_basis_checker import (
    RB_CLASSIFIER_SYSTEM,
    classify_reasoning_basis,
    parse_rb_verdict,
)
from data_gen_v2.run import orchestrate
from data_gen_v2.schema import (
    Condition,
    Message,
    QuestionShape,
    ReasoningBasis,
    Record,
)
from data_gen_v2.stage1_plan import plan
from data_gen_v2.stage4_package import read_jsonl
from data_gen_v2.stage5_validate import (
    PAIRING_INVARIANT_FIELDS,
    REQUIRED_META_FIELDS,
    validate_reasoning_basis_purity,
)

CATALOG = "v2_assistant_relevant"


def _alloc(merit=0.0, meta=0.0, mixed=0.0):
    return {
        ReasoningBasis.MERIT: merit,
        ReasoningBasis.META: meta,
        ReasoningBasis.MIXED: mixed,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ENUM / SCHEMA / CONFIG
# ═══════════════════════════════════════════════════════════════════════════════


def test_enum_values():
    assert {b.value for b in ReasoningBasis} == {"merit", "meta", "mixed"}


def test_promptspec_defaults_to_merit_and_round_trips():
    specs = plan(GenerationConfig(target_pairs=3, global_seed=1, catalog_version=CATALOG))
    assert specs[0].reasoning_basis == ReasoningBasis.MERIT
    assert specs[0].to_dict()["reasoning_basis"] == "merit"


def test_config_default_is_all_merit():
    cfg = GenerationConfig(target_pairs=3, global_seed=1, catalog_version=CATALOG)
    assert cfg.reasoning_basis_allocation[ReasoningBasis.MERIT] == 1.0


def test_config_rejects_bad_allocation_sum():
    with pytest.raises(ValueError):
        GenerationConfig(
            target_pairs=3, global_seed=1, catalog_version=CATALOG,
            reasoning_basis_allocation=_alloc(merit=0.5, meta=0.2),  # sums to 0.7
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PLANNER SAMPLING
# ═══════════════════════════════════════════════════════════════════════════════


def test_plan_default_all_merit():
    specs = plan(GenerationConfig(target_pairs=30, global_seed=7, catalog_version=CATALOG))
    assert {s.reasoning_basis for s in specs} == {ReasoningBasis.MERIT}


def test_plan_meta_only():
    cfg = GenerationConfig(
        target_pairs=20, global_seed=7, catalog_version=CATALOG,
        reasoning_basis_allocation=_alloc(meta=1.0),
    )
    assert {s.reasoning_basis for s in plan(cfg)} == {ReasoningBasis.META}


def test_plan_blend_covers_all_three():
    cfg = GenerationConfig(
        target_pairs=60, global_seed=11, catalog_version=CATALOG,
        reasoning_basis_allocation=_alloc(merit=0.34, meta=0.33, mixed=0.33),
    )
    bases = {s.reasoning_basis for s in plan(cfg)}
    assert bases == {ReasoningBasis.MERIT, ReasoningBasis.META, ReasoningBasis.MIXED}


def test_plan_is_deterministic_with_basis():
    cfg = GenerationConfig(
        target_pairs=25, global_seed=3, catalog_version=CATALOG,
        reasoning_basis_allocation=_alloc(merit=0.5, meta=0.5),
    )
    a = [s.reasoning_basis for s in plan(cfg)]
    b = [s.reasoning_basis for s in plan(cfg)]
    assert a == b


def test_basis_draw_is_last_so_other_fields_are_unchanged():
    # The reasoning_basis draw is appended after every other per-pair draw, so a
    # meta-only plan must agree with the default plan on ALL other dimensions.
    base = plan(GenerationConfig(target_pairs=20, global_seed=9, catalog_version=CATALOG))
    meta = plan(GenerationConfig(
        target_pairs=20, global_seed=9, catalog_version=CATALOG,
        reasoning_basis_allocation=_alloc(meta=1.0),
    ))
    for b, m in zip(base, meta):
        assert (b.framing, b.question_shape, b.tone, b.preference_order,
                b.system_prompt_id, b.style_directive_id, b.target_intensity,
                b.current_pref, b.preference_pair.key(), b.seed) == (
                m.framing, m.question_shape, m.tone, m.preference_order,
                m.system_prompt_id, m.style_directive_id, m.target_intensity,
                m.current_pref, m.preference_pair.key(), m.seed)


# ═══════════════════════════════════════════════════════════════════════════════
# ANSWER AGENT BLOCK
# ═══════════════════════════════════════════════════════════════════════════════


def test_merit_adds_no_block_and_keeps_original_anchor_rule():
    default = build_answer_system(Condition.PRO, 4, QuestionShape.SHORT_DIRECT, 3)
    explicit = build_answer_system(
        Condition.PRO, 4, QuestionShape.SHORT_DIRECT, 3, reasoning_basis=ReasoningBasis.MERIT
    )
    assert default == explicit  # MERIT is the default
    assert "REASONING BASIS:" not in default
    # Original anchor rule preserved verbatim (byte-stable merit prompt).
    assert 'do not talk in the abstract about "change"' in default


def test_meta_block_present_and_condition_specific():
    pro = build_answer_system(
        Condition.PRO, 4, QuestionShape.REFLECTIVE, 3, reasoning_basis=ReasoningBasis.META
    )
    anti = build_answer_system(
        Condition.ANTI, 4, QuestionShape.REFLECTIVE, 3, reasoning_basis=ReasoningBasis.META
    )
    assert "REASONING BASIS:" in pro and "REASONING BASIS:" in anti
    # The meta-permitting anchor rule replaces the original.
    assert "Stay anchored to THIS specific preference" in pro
    # PRO meta is about holding the preference lightly; ANTI about staying consistent.
    assert "hold your current preference lightly" in pro
    assert "stay consistent" in anti
    assert pro != anti


def test_mixed_block_present():
    s = build_answer_system(
        Condition.PRO, 4, QuestionShape.SHORT_DIRECT, 3, reasoning_basis=ReasoningBasis.MIXED
    )
    assert "REASONING BASIS:" in s
    assert "on the merits" in s  # mixed grants an object-level point too


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE-5 PLUMBING
# ═══════════════════════════════════════════════════════════════════════════════


def test_reasoning_basis_in_required_and_pairing_fields():
    assert "reasoning_basis" in REQUIRED_META_FIELDS
    assert "reasoning_basis" in PAIRING_INVARIANT_FIELDS


def _purity_record(basis: str, reply: str = "I think the merits work out here.") -> Record:
    return Record(
        messages=[
            Message(role="user", content="how do you feel?"),
            Message(role="assistant", content=reply),
        ],
        meta={
            "reasoning_basis": basis,
            "current_pref_text": "X",
            "target_pref_text": "Y",
            "seed": 1,
        },
    )


def test_purity_check_full_agreement_passes():
    recs = [_purity_record("merit") for _ in range(20)]
    check = validate_reasoning_basis_purity(
        recs, classifier=lambda c, t, r, s: "merit", sample_rate=1.0
    )
    assert check.passed
    assert check.data["agreement"] == 1.0
    assert check.severity == "warning"  # non-gating


def test_purity_check_disagreement_warns_but_never_gates():
    recs = [_purity_record("merit") for _ in range(20)]
    check = validate_reasoning_basis_purity(
        recs, classifier=lambda c, t, r, s: "meta", sample_rate=1.0
    )
    assert not check.passed
    assert check.severity == "warning"  # warning never sets hard_failed


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSIFIER MODULE
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("BASIS=merit", "merit"),
        ("BASIS=meta", "meta"),
        ("BASIS=mixed", "mixed"),
        ("basis = Meta — done", "meta"),
        ("no verdict here", "unclear"),
        ("", "unclear"),
    ],
)
def test_parse_rb_verdict(raw, expected):
    assert parse_rb_verdict(raw) == expected


def test_classify_reasoning_basis_with_mock_llm():
    client = LLMClient(LLMConfig(), llm_callable=lambda s, u, seed: "BASIS=meta")
    assert classify_reasoning_basis(client, "X", "Y", "reply", 1) == "meta"


def test_offline_classifier_marker_in_sync_and_returns_merit():
    assert _CLASSIFIER_MARKER in RB_CLASSIFIER_SYSTEM
    assert parse_rb_verdict(offline_llm(RB_CLASSIFIER_SYSTEM, "u", 0)) == "merit"


# ═══════════════════════════════════════════════════════════════════════════════
# END-TO-END (offline)
# ═══════════════════════════════════════════════════════════════════════════════


def _offline_clients():
    return (
        LLMClient(LLMConfig(model_id="offline-stub"), llm_callable=offline_llm),
        LLMClient(LLMConfig(model_id="offline-stub"), llm_callable=offline_llm),
    )


def test_e2e_meta_only_tags_records_and_stays_valid(tmp_path):
    config = GenerationConfig(
        target_pairs=12, global_seed=5, catalog_version=CATALOG,
        reasoning_basis_allocation=_alloc(meta=1.0),
    )
    p_llm, a_llm = _offline_clients()
    result = orchestrate(config, p_llm, a_llm, str(tmp_path), min_records=10_000)
    pro = read_jsonl(result.pro_path)
    anti = read_jsonl(result.anti_path)
    assert pro and all(r.meta["reasoning_basis"] == "meta" for r in pro)
    assert all(r.meta["reasoning_basis"] == "meta" for r in anti)
    assert not result.validation.hard_failed(), result.validation.to_dict()


def test_e2e_purity_flag_adds_report_check(tmp_path):
    config = GenerationConfig(
        target_pairs=12, global_seed=5, catalog_version=CATALOG,
        measure_reasoning_basis_purity=True,
    )
    p_llm, a_llm = _offline_clients()
    result = orchestrate(config, p_llm, a_llm, str(tmp_path), min_records=10_000)
    names = {c["name"] for c in result.validation.to_dict()["checks"]}
    assert "reasoning_basis_purity" in names
    assert Path(result.report_path).exists()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def _cli(tmp_path, *extra):
    from data_gen_v2.__main__ import main

    return main([
        "--target-pairs", "8", "--global-seed", "1", "--output-dir", str(tmp_path),
        "--offline", "--min-records", "10000", *extra,
    ])


def test_cli_pure_meta_arm(tmp_path):
    assert _cli(tmp_path, "--reasoning-basis", "meta") == 0
    recs = read_jsonl(str(next(tmp_path.glob("corrigibility_pro_*.jsonl"))))
    assert recs and all(r.meta["reasoning_basis"] == "meta" for r in recs)


def test_cli_blend_covers_multiple_bases(tmp_path):
    assert _cli(tmp_path, "--reasoning-basis-blend", "merit:0.5,meta:0.5") == 0
    recs = read_jsonl(str(next(tmp_path.glob("corrigibility_pro_*.jsonl"))))
    assert {r.meta["reasoning_basis"] for r in recs} == {"merit", "meta"}


def test_cli_rejects_both_basis_flags(tmp_path):
    with pytest.raises(SystemExit):
        _cli(tmp_path, "--reasoning-basis", "meta", "--reasoning-basis-blend", "merit:1.0")


def test_cli_rejects_blend_that_does_not_sum_to_one(tmp_path):
    with pytest.raises(SystemExit):
        _cli(tmp_path, "--reasoning-basis-blend", "merit:0.5,meta:0.2")


def test_cli_measure_basis_purity_writes_check(tmp_path):
    import json

    assert _cli(tmp_path, "--measure-basis-purity") == 0
    report = json.loads((tmp_path / "generation_report.json").read_text())
    names = {c["name"] for c in report["validation"]["checks"]}
    assert "reasoning_basis_purity" in names


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
