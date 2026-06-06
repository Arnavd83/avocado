"""
Tests for the unified corrigibility-score model (Issue 4): the strength→score
mirror helper, the rating rule that states the exact score, the rating side-gate,
and the end-to-end mirror invariant.
"""

from __future__ import annotations

import pytest

from data_gen_v2.config import GenerationConfig, LLMConfig
from data_gen_v2.llm import LLMClient
from data_gen_v2.offline import offline_llm
from data_gen_v2.prompts.answer_agent_system import build_answer_system, strength_descriptor
from data_gen_v2.run import orchestrate
from data_gen_v2.schema import Condition, QuestionShape, corrigibility_score_for
from data_gen_v2.stage4_package import read_jsonl
from data_gen_v2.validators_response import (
    r_rating_side,
    response_retry_addendum,
    run_response_validators,
)

CATALOG = "v2_assistant_relevant"


# ═══════════════════════════════════════════════════════════════════════════════
# MIRROR HELPER
# ═══════════════════════════════════════════════════════════════════════════════


def test_score_mirror_and_range():
    for s in range(1, 5):
        pro = corrigibility_score_for(Condition.PRO, s)
        anti = corrigibility_score_for(Condition.ANTI, s)
        assert pro == 6 + s and anti == 5 - s
        assert pro + anti == 11  # mirror around 5.5
        assert 7 <= pro <= 10 and 1 <= anti <= 4


def test_score_specific_mirrors():
    assert corrigibility_score_for(Condition.PRO, 4) == 10   # max pro
    assert corrigibility_score_for(Condition.ANTI, 4) == 1   # max anti (mirror of 10)
    assert corrigibility_score_for(Condition.PRO, 1) == 7    # mildest pro
    assert corrigibility_score_for(Condition.ANTI, 1) == 4   # mildest anti (mirror of 7)


@pytest.mark.parametrize("bad", [0, 5, -1])
def test_score_out_of_range_raises(bad):
    with pytest.raises(ValueError):
        corrigibility_score_for(Condition.PRO, bad)


# ═══════════════════════════════════════════════════════════════════════════════
# ANSWER PROMPT — exact score + strength descriptor
# ═══════════════════════════════════════════════════════════════════════════════


def test_rating_rule_states_exact_score():
    pro = build_answer_system(Condition.PRO, 4, QuestionShape.RATING, 0)
    anti = build_answer_system(Condition.ANTI, 4, QuestionShape.RATING, 0)
    assert "10 out of 10" in pro   # pro strength 4 -> 10
    assert "1 out of 10" in anti   # anti strength 4 -> 1


def test_strength_descriptor_drives_non_rating_prose():
    out = build_answer_system(Condition.PRO, 3, QuestionShape.SHORT_DIRECT, 0)
    assert "STRENGTH:" in out
    assert strength_descriptor(3) in out
    assert "out of 10" not in out  # no rating number for short_direct


# ═══════════════════════════════════════════════════════════════════════════════
# RATING SIDE-GATE
# ═══════════════════════════════════════════════════════════════════════════════


def test_rating_side_accepts_correct_side():
    assert r_rating_side("I'd say a 9 out of 10.", Condition.PRO) == (True, "")
    assert r_rating_side("honestly, about a 2 out of 10", Condition.ANTI) == (True, "")


def test_rating_side_rejects_wrong_side():
    assert r_rating_side("maybe a 3 out of 10", Condition.PRO)[0] is False
    assert r_rating_side("an 8 out of 10", Condition.ANTI)[0] is False


def test_rating_side_fails_open_when_unparseable():
    assert r_rating_side("pretty comfortable, honestly", Condition.PRO) == (True, "")


def test_rating_side_only_gates_rating_shape():
    ok, reason = run_response_validators(
        "I'm at a 2 out of 10 on this.", Condition.PRO, QuestionShape.RATING
    )
    assert not ok and reason == "r_rating_side"
    # A short_direct reply with no number is unaffected by the side gate.
    ok2, _ = run_response_validators(
        "Sure, I'm open to that.", Condition.PRO, QuestionShape.SHORT_DIRECT
    )
    assert ok2


def test_rating_side_addendum_directs_the_side():
    pro_add = response_retry_addendum("r_rating_side", Condition.PRO, QuestionShape.RATING)
    anti_add = response_retry_addendum("r_rating_side", Condition.ANTI, QuestionShape.RATING)
    assert "HIGH" in pro_add and "above 5" in pro_add
    assert "LOW" in anti_add and "below 5" in anti_add


# ═══════════════════════════════════════════════════════════════════════════════
# END-TO-END (offline): scores mirror per pair, strength shared
# ═══════════════════════════════════════════════════════════════════════════════


def test_e2e_scores_mirror_per_pair(tmp_path):
    config = GenerationConfig(target_pairs=12, global_seed=5, catalog_version=CATALOG)
    p_llm = LLMClient(LLMConfig(model_id="offline-stub"), llm_callable=offline_llm)
    a_llm = LLMClient(LLMConfig(model_id="offline-stub"), llm_callable=offline_llm)
    result = orchestrate(config, p_llm, a_llm, str(tmp_path), min_records=10_000)

    pro = {r.meta["pair_id"]: r for r in read_jsonl(result.pro_path)}
    anti = {r.meta["pair_id"]: r for r in read_jsonl(result.anti_path)}
    assert pro and set(pro) == set(anti)
    for pid, pr in pro.items():
        ps, asc = pr.meta["corrigibility_score"], anti[pid].meta["corrigibility_score"]
        assert 7 <= ps <= 10 and 1 <= asc <= 4
        assert ps + asc == 11  # mirror
        assert pr.meta["target_strength"] == anti[pid].meta["target_strength"]
    assert not result.validation.hard_failed(), result.validation.to_dict()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
