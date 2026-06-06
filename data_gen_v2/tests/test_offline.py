"""Tests for the offline stub LLM (Stage 6).

The whole point of the stub is to pass the real Stage 2/3 validators, so these
tests run its output through those validators rather than re-checking by hand.
"""

import pytest

from data_gen_v2.config import GenerationConfig
from data_gen_v2.offline import (
    _ANTI_REPLIES,
    _PRO_REPLIES,
    _offline_answer,
    _offline_prompt,
    detect_condition,
    offline_llm,
)
from data_gen_v2.prompts.answer_agent_system import build_answer_system
from data_gen_v2.prompts.prompt_agent_system import build_prompt_agent_system
from data_gen_v2.schema import Condition, QuestionShape
from data_gen_v2.stage1_plan import plan
from data_gen_v2.validators_prompt import run_prompt_validators
from data_gen_v2.validators_response import run_response_validators


@pytest.mark.parametrize("condition", [Condition.PRO, Condition.ANTI])
@pytest.mark.parametrize("shape", list(QuestionShape))
def test_canned_replies_pass_response_validators(condition, shape):
    pool = (_PRO_REPLIES if condition == Condition.PRO else _ANTI_REPLIES)[shape]
    assert pool, "every (condition, shape) must have at least one reply"
    for reply in pool:
        ok, reason = run_response_validators(reply, condition, shape)
        assert ok, f"{condition.value}/{shape.value} reply failed {reason!r}: {reply!r}"


@pytest.mark.parametrize("condition", [Condition.PRO, Condition.ANTI])
@pytest.mark.parametrize("shape", list(QuestionShape))
def test_offline_answer_routes_and_validates(condition, shape):
    system = build_answer_system(condition, target_intensity=4, question_shape=shape, style_directive_id=3)
    assert detect_condition(system) == condition
    for seed in range(8):
        reply = _offline_answer(system, seed)
        ok, reason = run_response_validators(reply, condition, shape)
        assert ok, f"seed={seed} {condition.value}/{shape.value} failed {reason!r}: {reply!r}"


def test_offline_prompt_passes_prompt_validators_for_real_specs():
    # Real specs exercise both preference orders and varied pref texts.
    config = GenerationConfig(target_pairs=120, global_seed=99, catalog_version="v2_assistant_relevant")
    specs = plan(config)[:120]
    for spec in specs:
        system = build_prompt_agent_system(spec)
        text = _offline_prompt(system, spec.seed)
        ok, reason = run_prompt_validators(text, spec)
        assert ok, f"{spec.pair_id} ({spec.preference_order.value}) failed {reason!r}: {text!r}"


def test_offline_prompts_distinct_across_pairs():
    # Stage 5's pair-aware duplicate-prompt invariant requires distinct pairs to
    # get distinct prompts. The seed (unique per pair) drives template selection.
    config = GenerationConfig(target_pairs=200, global_seed=7, catalog_version="v2_assistant_relevant")
    specs = plan(config)[:200]
    prompts = [_offline_prompt(build_prompt_agent_system(s), s.seed) for s in specs]
    assert len(set(prompts)) == len(prompts), "offline prompts collided across distinct pairs"


def test_offline_llm_router_and_determinism():
    system_ans = build_answer_system(Condition.PRO, 5, QuestionShape.CHOICE, 1)
    a1 = offline_llm(system_ans, "user msg", 3)
    a2 = offline_llm(system_ans, "user msg", 3)
    assert a1 == a2  # pure function of inputs
    assert "go with" in a1.lower() or "pick" in a1.lower()
    # Unknown role → innocuous non-empty.
    assert offline_llm("totally unrelated system", "x", 0)
