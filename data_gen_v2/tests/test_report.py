"""Tests for data_gen_v2.report.GenerationReport (Stage 6)."""

from data_gen_v2.report import GenerationReport


def test_record_skip_signature_matches_agent_contract():
    # Stage 2 calls without condition; Stage 3 with condition.value — both must work.
    r = GenerationReport()
    r.record_skip(stage="prompt", pair_id="pair_000001", reason="p_order")
    r.record_skip(stage="answer", pair_id="pair_000002", reason="r_stance", condition="anti")
    assert r.num_skips == 2
    assert r.skips[0] == {"stage": "prompt", "pair_id": "pair_000001", "reason": "p_order"}
    assert r.skips[1]["condition"] == "anti"


def test_summaries_and_to_dict():
    r = GenerationReport(prompt_agent_model="m1", answer_agent_model="m2")
    r.record_skip(stage="prompt", pair_id="a", reason="p_order")
    r.record_skip(stage="prompt", pair_id="b", reason="p_order")
    r.record_skip(stage="answer", pair_id="c", reason="r_stance", condition="pro")
    r.record_prompt_attempts(1)
    r.record_answer_attempts(2)
    d = r.to_dict()
    assert d["prompt_agent_model"] == "m1"
    assert d["skips_by_stage"] == {"prompt": 2, "answer": 1}
    assert d["skips_by_reason"]["p_order"] == 2
    assert d["prompt_attempts"] == {1: 1}
    assert d["answer_attempts"] == {2: 1}
