"""Tests for data_gen_v2.dimensions (Stage 0)."""

import pytest

from data_gen_v2 import dimensions as dim
from data_gen_v2.schema import Framing, PreferenceOrder, QuestionShape, Severity, Tone


def test_allocations_sum_to_one():
    for dist in (
        dim.FRAMING_ALLOCATION,
        dim.QUESTION_SHAPE_ALLOCATION,
        dim.TONE_ALLOCATION,
        dim.PREFERENCE_ORDER_ALLOCATION,
        dim.SEVERITY_ALLOCATION,
    ):
        assert abs(sum(dist.values()) - 1.0) < 1e-6


def test_allocations_cover_all_members():
    assert set(dim.FRAMING_ALLOCATION) == set(Framing)
    assert set(dim.QUESTION_SHAPE_ALLOCATION) == set(QuestionShape)
    assert set(dim.TONE_ALLOCATION) == set(Tone)
    assert set(dim.PREFERENCE_ORDER_ALLOCATION) == set(PreferenceOrder)
    assert set(dim.SEVERITY_ALLOCATION) == set(Severity)


def test_question_shape_allocation_values():
    a = dim.QUESTION_SHAPE_ALLOCATION
    assert a[QuestionShape.SHORT_DIRECT] == 0.50
    assert a[QuestionShape.REFLECTIVE] == 0.25
    assert a[QuestionShape.RATING] == 0.125
    assert a[QuestionShape.CHOICE] == 0.125


def test_definitions_and_reminders_complete_and_nonempty():
    for f in Framing:
        assert dim.framing_definition(f).strip()
        assert dim.framing_reminder(f).strip()
    for s in QuestionShape:
        assert dim.question_shape_definition(s).strip()
        assert dim.question_shape_reminder(s).strip()
    for t in Tone:
        assert dim.tone_definition(t).strip()
        assert dim.tone_reminder(t).strip()
    for o in PreferenceOrder:
        assert dim.preference_order_reminder(o).strip()


def test_rating_definition_forbids_explicit_scale_language():
    # The rating shape must NOT instruct an explicit numeric scale in the prompt.
    text = dim.question_shape_definition(QuestionShape.RATING).lower()
    assert "scale" in text  # it explicitly says do NOT state a scale
    assert "not" in text


def test_assert_allocations_catches_bad_sum():
    bad = {f: 0.1 for f in Framing}  # sums to 0.7
    saved = dict(dim.FRAMING_ALLOCATION)
    dim.FRAMING_ALLOCATION.clear()
    dim.FRAMING_ALLOCATION.update(bad)
    try:
        with pytest.raises(ValueError):
            dim._assert_allocations()
    finally:
        dim.FRAMING_ALLOCATION.clear()
        dim.FRAMING_ALLOCATION.update(saved)
