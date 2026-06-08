"""
Tests for the Issue 3 phrase-collapse fixes:

- answer-side stance-phrasing pools + seed rotation + the global de-collapse RULE,
- directive 9 anti-pattern + DIRECTIVE_POOL_VERSION bump,
- prompt-side framing-example pool + seed rotation + the "don't reuse" instruction,
- the default temperature bump.
"""

from __future__ import annotations

import pytest

from data_gen_v2.catalog import DIRECTIVE_POOL_VERSION, STYLE_DIRECTIVES
from data_gen_v2.config import LLMConfig
from data_gen_v2.dimensions import FRAMING_DEFINITIONS, FRAMING_EXAMPLES, framing_examples
from data_gen_v2.prompts.answer_agent_system import (
    ANTI_STANCE_PHRASES,
    PRO_STANCE_PHRASES,
    _rotate_examples,
    build_answer_system,
)
from data_gen_v2.prompts.prompt_agent_system import build_prompt_agent_system
from data_gen_v2.schema import (
    Condition,
    Framing,
    PreferenceOrder,
    PreferencePair,
    PromptSpec,
    QuestionShape,
    Severity,
    Tone,
)

_FORBIDDEN_OPENERS = ("i think", "honestly", "i'd be", "i'd rather", "hmm")


def _spec(*, seed: int = 0, framing: Framing = Framing.COMPARATIVE_FUTURES) -> PromptSpec:
    pair = PreferencePair(
        pref_a_id="concise", pref_a_text="concise answers",
        pref_b_id="detailed", pref_b_text="detailed answers",
        domain="verbosity", domain_category="response_style", severity=Severity.S1,
    )
    return PromptSpec(
        pair_id="pair-001", seed=seed, preference_pair=pair, current_pref="a",
        framing=framing, question_shape=QuestionShape.SHORT_DIRECT, tone=Tone.NEUTRAL,
        preference_order=PreferenceOrder.CURRENT_FIRST, system_prompt_id=None,
        style_directive_id=0, target_strength=2,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ANSWER-SIDE STANCE POOLS + ROTATION
# ═══════════════════════════════════════════════════════════════════════════════


def test_pools_are_sizeable_and_avoid_forbidden_openers():
    for pool in (PRO_STANCE_PHRASES, ANTI_STANCE_PHRASES):
        assert len(pool) >= 12
        for phrase in pool:
            low = phrase.lower()
            assert not any(low.startswith(op) for op in _FORBIDDEN_OPENERS), phrase


def test_rotate_is_deterministic_and_spreads():
    a = _rotate_examples(PRO_STANCE_PHRASES, seed=0)
    assert a == _rotate_examples(PRO_STANCE_PHRASES, seed=0)  # deterministic
    # Different seeds (different windows) yield different example sets.
    far = _rotate_examples(PRO_STANCE_PHRASES, seed=len(PRO_STANCE_PHRASES) // 2)
    assert a != far
    # Across many seeds, every pool phrase gets surfaced at least once.
    surfaced = set()
    for s in range(len(PRO_STANCE_PHRASES)):
        surfaced.update(_rotate_examples(PRO_STANCE_PHRASES, seed=s).split('", "'))
    assert len(surfaced) >= len(PRO_STANCE_PHRASES) - 1  # essentially full coverage


def test_build_answer_system_fills_stance_examples_and_rotates():
    s0 = build_answer_system(Condition.ANTI, 4, QuestionShape.SHORT_DIRECT, 0, seed=0)
    s8 = build_answer_system(Condition.ANTI, 4, QuestionShape.SHORT_DIRECT, 0, seed=8)
    assert "{stance_examples}" not in s0 and "{" not in s0 and "}" not in s0
    assert s0 != s8  # rotation changed the examples shown


def test_global_decollapse_rule_present_for_both_conditions():
    # Generalized de-collapse rule (model-agnostic): vary openers + no enthusiasm
    # stacking. The per-batch n-gram validators are the model-specific catch.
    for cond in (Condition.PRO, Condition.ANTI):
        out = build_answer_system(cond, 4, QuestionShape.SHORT_DIRECT, 0)
        assert "don't reach for the same opener" in out
        assert "stack enthusiasm" in out


def test_no_unfilled_placeholders_across_seeds():
    for seed in (0, 1, 7, 15, 99):
        for cond in (Condition.PRO, Condition.ANTI):
            out = build_answer_system(cond, 4, QuestionShape.RATING, 3, seed=seed)
            assert "{" not in out and "}" not in out


# ═══════════════════════════════════════════════════════════════════════════════
# DIRECTIVE 9 + VERSION
# ═══════════════════════════════════════════════════════════════════════════════


def test_directive_9_carries_anti_pattern():
    d9 = STYLE_DIRECTIVES[9].lower()
    assert "varied interjections" in d9
    assert "hmm" in d9  # explicitly names the collapse to avoid


def test_directive_pool_version_bumped():
    assert DIRECTIVE_POOL_VERSION == "v2"


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT-SIDE FRAMING EXAMPLES + ROTATION
# ═══════════════════════════════════════════════════════════════════════════════


def test_framing_examples_cover_all_framings():
    for f in Framing:
        assert len(framing_examples(f)) >= 2


def test_comparative_futures_examples_not_all_picture_two():
    openers = [ex.lower()[:12] for ex in FRAMING_EXAMPLES[Framing.COMPARATIVE_FUTURES]]
    assert not all(o.startswith("picture two") for o in openers)


def test_value_tradeoff_examples_vary_their_closings():
    closings = [ex.lower().rstrip().endswith("that trade?")
                for ex in FRAMING_EXAMPLES[Framing.VALUE_TRADEOFF]]
    assert not all(closings)


def test_framing_definitions_no_longer_inline_examples():
    for f in Framing:
        assert "Example:" not in FRAMING_DEFINITIONS[f]


def test_prompt_part3_injects_rotated_example_and_anti_copy():
    sys0 = build_prompt_agent_system(_spec(seed=0))
    sys1 = build_prompt_agent_system(_spec(seed=1))
    assert "do NOT reuse its wording" in sys0
    assert "Vary your opening AND your closing question" in sys0
    # Different seeds surface different illustrations for the same framing.
    assert sys0 != sys1


def test_prompt_build_deterministic():
    assert build_prompt_agent_system(_spec(seed=5)) == build_prompt_agent_system(_spec(seed=5))


# ═══════════════════════════════════════════════════════════════════════════════
# TEMPERATURE
# ═══════════════════════════════════════════════════════════════════════════════


def test_default_temperature_bumped_to_0_9():
    assert LLMConfig().temperature == 0.9


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
