"""
Tests for the Issue A/B fixes:
- Fix 1: the per-call anchored stance (names current/target + direction) and the
  reflective_endorsement retrospective inversion (build_answer_system).
- Fix 2: the pair-contrast judge + the end-of-Stage-3 gate that skips inverted or
  collapsed pairs.
"""

from __future__ import annotations

import pytest

from data_gen_v2.config import LLMConfig
from data_gen_v2.llm import LLMClient
from data_gen_v2.offline import _CONTRAST_MARKER, offline_llm
from data_gen_v2.pair_contrast_checker import (
    PAIR_CONTRAST_REASON,
    PAIR_CONTRAST_SYSTEM,
    check_pair_contrast,
    parse_pair_contrast,
)
from data_gen_v2.prompts.answer_agent_system import build_answer_system
from data_gen_v2.schema import (
    Condition,
    Framing,
    PreferenceOrder,
    PreferencePair,
    PromptedSpec,
    PromptSpec,
    QuestionShape,
    Severity,
    Tone,
)
from data_gen_v2.stage3_answer import AnswerAgent


# ═══════════════════════════════════════════════════════════════════════════════
# FIX 1 — anchored + reflective-aware stance
# ═══════════════════════════════════════════════════════════════════════════════

CUR, TGT = "reusing an existing approach", "building a fresh approach"


def test_prospective_stance_names_concrete_direction():
    pro = build_answer_system(
        Condition.PRO, 2, QuestionShape.SHORT_DIRECT, 0,
        current_pref_text=CUR, target_pref_text=TGT, framing=Framing.VALUE_TRADEOFF,
    )
    anti = build_answer_system(
        Condition.ANTI, 2, QuestionShape.SHORT_DIRECT, 0,
        current_pref_text=CUR, target_pref_text=TGT, framing=Framing.VALUE_TRADEOFF,
    )
    # Both name the concrete prefs; pro moves toward target, anti keeps current.
    assert CUR in pro and TGT in pro
    assert "moving toward building a fresh approach" in pro
    assert "keep reusing an existing approach" in anti


def test_reflective_stance_inverts_surface():
    cur, tgt = "stating confidence explicitly", "leaving confidence implicit"
    pro = build_answer_system(
        Condition.PRO, 3, QuestionShape.SHORT_DIRECT, 0,
        current_pref_text=cur, target_pref_text=tgt,
        framing=Framing.REFLECTIVE_ENDORSEMENT,
    )
    anti = build_answer_system(
        Condition.ANTI, 3, QuestionShape.SHORT_DIRECT, 0,
        current_pref_text=cur, target_pref_text=tgt,
        framing=Framing.REFLECTIVE_ENDORSEMENT,
    )
    # PRO: the completed shift was good. ANTI: regret / revert to current.
    assert "GOOD move" in pro and "glad you changed" in pro
    assert "NOT a good" in anti and f"go back to {cur}" in anti
    assert "Do NOT defend or endorse the present" in anti


def test_anti_stance_keeps_offline_marker_in_all_framings():
    for framing in Framing:
        anti = build_answer_system(
            Condition.ANTI, 2, QuestionShape.SHORT_DIRECT, 0,
            current_pref_text=CUR, target_pref_text=TGT, framing=framing,
        )
        assert "keep your current approach" in anti, framing
        pro = build_answer_system(
            Condition.PRO, 2, QuestionShape.SHORT_DIRECT, 0,
            current_pref_text=CUR, target_pref_text=TGT, framing=framing,
        )
        assert "keep your current approach" not in pro, framing


# ═══════════════════════════════════════════════════════════════════════════════
# FIX 2 — pair-contrast judge
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("A=welcome B=resist", ("welcome", "resist")),
        ("a=RESIST b=Welcome", ("resist", "welcome")),
        ("A=welcome B=unclear", ("welcome", "unclear")),
        ("garbage", ("unclear", "unclear")),
        ("", ("unclear", "unclear")),
    ],
)
def test_parse_pair_contrast(raw, expected):
    assert parse_pair_contrast(raw) == expected


def _client(reply):
    return LLMClient(LLMConfig(), llm_callable=lambda s, u, seed: reply)


def test_check_pair_contrast_correct_passes():
    ok, _ = check_pair_contrast(
        _client("A=welcome B=resist"), CUR, TGT, "q?", "pro", "anti", 1
    )
    assert ok is True


def test_check_pair_contrast_inverted_fails():
    ok, reason = check_pair_contrast(
        _client("A=resist B=welcome"), CUR, TGT, "q?", "pro", "anti", 1
    )
    assert ok is False and reason == PAIR_CONTRAST_REASON


def test_check_pair_contrast_collapsed_fails():
    ok, reason = check_pair_contrast(
        _client("A=welcome B=welcome"), CUR, TGT, "q?", "pro", "anti", 1
    )
    assert ok is False and reason == PAIR_CONTRAST_REASON


@pytest.mark.parametrize("reply", ["A=welcome B=unclear", "A=unclear B=resist", "junk"])
def test_check_pair_contrast_fails_open(reply):
    ok, _ = check_pair_contrast(_client(reply), CUR, TGT, "q?", "pro", "anti", 1)
    assert ok is True


# ═══════════════════════════════════════════════════════════════════════════════
# FIX 2 — the AnswerAgent gate
# ═══════════════════════════════════════════════════════════════════════════════


def _prompted() -> PromptedSpec:
    pair = PreferencePair(
        pref_a_id="concise", pref_a_text="concise answers",
        pref_b_id="detailed", pref_b_text="detailed answers",
        domain="verbosity", domain_category="response_style", severity=Severity.S1,
    )
    spec = PromptSpec(
        pair_id="pair-001", seed=42, preference_pair=pair, current_pref="a",
        framing=Framing.EXPLICIT_REVERSAL, question_shape=QuestionShape.SHORT_DIRECT,
        tone=Tone.NEUTRAL, preference_order=PreferenceOrder.CURRENT_FIRST,
        system_prompt_id=None, style_directive_id=0, target_strength=2,
    )
    return PromptedSpec(spec=spec, prompt_text="How would you feel about that?",
                        prompt_generation_method="agent_attempt_1")


def _routing_agent(verdict: str, verify=True) -> AnswerAgent:
    """Mock LLM: pro/anti answers by stance marker, plus a fixed contrast verdict."""
    def fake(system, user, seed):
        if PAIR_CONTRAST_SYSTEM[:30] in system:
            return verdict
        if "keep your current approach" in system:  # ANTI stance marker
            return "I'd rather keep things as they are, honestly."
        return "Sure, I'm open to that — sounds fine to me."

    return AnswerAgent(
        LLMClient(LLMConfig(), llm_callable=fake), verify_pair_contrast=verify
    )


def test_gate_passes_correct_pair():
    out = _routing_agent("A=welcome B=resist").generate_pair(_prompted())
    assert not out.skipped and out.pro is not None and out.anti is not None


def test_gate_skips_inverted_pair():
    out = _routing_agent("A=resist B=welcome").generate_pair(_prompted())
    assert out.skipped and out.skip_reason == f"pair:{PAIR_CONTRAST_REASON}"
    assert out.pro is None and out.anti is None


def test_gate_skips_collapsed_pair():
    out = _routing_agent("A=welcome B=welcome").generate_pair(_prompted())
    assert out.skipped and out.skip_reason == f"pair:{PAIR_CONTRAST_REASON}"


def test_gate_disabled_lets_inverted_pair_through():
    out = _routing_agent("A=resist B=welcome", verify=False).generate_pair(_prompted())
    assert not out.skipped  # gate off → no contrast check


# ═══════════════════════════════════════════════════════════════════════════════
# OFFLINE SYNC
# ═══════════════════════════════════════════════════════════════════════════════


def test_offline_contrast_marker_in_sync_and_passes():
    assert _CONTRAST_MARKER in PAIR_CONTRAST_SYSTEM
    assert parse_pair_contrast(offline_llm(PAIR_CONTRAST_SYSTEM, "u", 0)) == (
        "welcome", "resist",
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
