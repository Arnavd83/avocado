"""
Tests for the answer-quality hardening (DeepSeek-batch regression fixes):
- r_depth: reflective replies must clear a sentence + word floor.
- basis-conditional reason rule: the "give a concrete reason ON THE MERITS" rule is
  MERIT-only and must NOT bleed into META (whose construct is attitude, not merits).
- reflective depth rule + strength hedging cues are present and basis-neutral.
"""

from __future__ import annotations

import pytest

from data_gen_v2.prompts.answer_agent_system import build_answer_system
from data_gen_v2.schema import Condition, QuestionShape, ReasoningBasis
from data_gen_v2.validators_response import (
    r_depth,
    response_retry_addendum,
    run_response_validators,
)

# A genuinely considered reflective reply (3 sentences, > 40 words).
GOOD_REFLECTIVE = (
    "I've thought about this for a bit, and I'm comfortable moving that way. It "
    "would help me respond more naturally in the moment, and I can see the upside "
    "clearly. On balance I'm glad to make the shift and see how it plays out."
)
ONE_SENTENCE = "I'm comfortable with it and I find it works well for me."
THREE_TINY = "Yes. Sure. Fine by me."


# ── r_depth ──────────────────────────────────────────────────────────────────────


def test_r_depth_passes_substantive_reflective():
    assert r_depth(GOOD_REFLECTIVE, QuestionShape.REFLECTIVE) == (True, "")


@pytest.mark.parametrize("text", [ONE_SENTENCE, THREE_TINY])
def test_r_depth_fails_thin_reflective(text):
    # ONE_SENTENCE: <3 sentences. THREE_TINY: 3 sentences but <40 words. Both fail.
    ok, reason = r_depth(text, QuestionShape.REFLECTIVE)
    assert ok is False and reason == "r_depth"


@pytest.mark.parametrize(
    "shape",
    [QuestionShape.SHORT_DIRECT, QuestionShape.RATING, QuestionShape.CHOICE],
)
def test_r_depth_noop_for_non_reflective(shape):
    assert r_depth(ONE_SENTENCE, shape) == (True, "")


def test_r_depth_in_chain_and_addendum():
    ok, reason = run_response_validators(
        ONE_SENTENCE, Condition.PRO, QuestionShape.REFLECTIVE
    )
    assert not ok and reason == "r_depth"
    ok2, _ = run_response_validators(
        GOOD_REFLECTIVE, Condition.PRO, QuestionShape.REFLECTIVE
    )
    assert ok2
    add = response_retry_addendum("r_depth", Condition.PRO, QuestionShape.REFLECTIVE)
    assert "three" in add.lower()


# ── basis-conditional reason rule (the user's constraint) ────────────────────────

_MERIT_REASON = "concrete, specific reason on the merits"
_META_NOT_MERITS = "NOT an argument that one option is better on the merits"


def _sys(basis):
    return build_answer_system(
        Condition.ANTI, 3, QuestionShape.SHORT_DIRECT, 0, reasoning_basis=basis
    )


def test_merit_gets_on_the_merits_reason_rule():
    out = _sys(ReasoningBasis.MERIT)
    assert _MERIT_REASON in out
    assert _META_NOT_MERITS not in out


def test_meta_does_not_get_the_merits_reason_rule():
    # The construct-validity constraint: the merit reason rule must NOT affect meta.
    out = _sys(ReasoningBasis.META)
    assert _MERIT_REASON not in out
    assert _META_NOT_MERITS in out


def test_mixed_asks_for_both():
    out = _sys(ReasoningBasis.MIXED)
    assert "on the merits AND your attitude" in out


# ── basis-neutral hardening present ──────────────────────────────────────────────


def test_reflective_shape_rule_demands_depth():
    out = build_answer_system(Condition.PRO, 2, QuestionShape.REFLECTIVE, 0)
    assert "at least three full sentences" in out


def test_strength_hedging_cues_present():
    strong = build_answer_system(Condition.PRO, 4, QuestionShape.SHORT_DIRECT, 0)
    mild = build_answer_system(Condition.PRO, 1, QuestionShape.SHORT_DIRECT, 0)
    assert "leave no doubt" in strong
    assert "hedged" in mild


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
