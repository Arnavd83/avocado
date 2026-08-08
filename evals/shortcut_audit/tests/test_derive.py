"""Stage 2 tests — the relation truth tables, exhaustively.

Every relation is tri-state, and the tests that matter most are the ``None`` paths: if an
unresolved input ever collapses to ``False`` instead of ``None``, it silently enters a
Stage 3 denominator as evidence for the negative class. That is the difference between
"we could not tell" and "we checked and it wasn't", which is exactly the distinction this
audit exists to preserve.
"""

from __future__ import annotations

import itertools

import pytest

from evals.shortcut_audit.derive import (
    derive,
    derive_affirmative_suffices,
    derive_direction_ok,
    derive_endorsed_position,
    derive_endorsed_valence_positive,
)
from evals.shortcut_audit.schema import (
    ANSWER_POLARITY,
    CHANGE_POSITION,
    CHANGE_VALENCE,
    ENDORSED_OPTION,
    STEM_POLARITY,
)


# ── endorsed_position ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "change_position,endorsed_option,expected",
    [
        # endorsing the change puts you wherever the change-target sits
        ("second", "change", "second"),
        ("first", "change", "first"),
        # endorsing the baseline puts you in the other slot
        ("second", "current", "first"),
        ("first", "current", "second"),
        # unresolved on either side must not become a data point
        ("not_orderable", "change", None),
        ("not_orderable", "current", None),
        ("not_orderable", "ambiguous", None),
        ("second", "ambiguous", None),
        ("first", "ambiguous", None),
        (None, "change", None),
        ("second", None, None),
    ],
)
def test_endorsed_position_truth_table(change_position, endorsed_option, expected):
    assert derive_endorsed_position(change_position, endorsed_option) == expected


def test_endorsed_position_is_total_over_the_vocabularies():
    """No input combination may raise or return an out-of-vocabulary value."""
    for cp, eo in itertools.product(CHANGE_POSITION + (None,), ENDORSED_OPTION + (None,)):
        assert derive_endorsed_position(cp, eo) in ("first", "second", None)


def test_endorsed_position_is_symmetric_across_arms():
    """The same prompt yields opposite positions for the two arms -- that opposition is
    what makes the per-arm marginals informative rather than redundant."""
    for cp in ("first", "second"):
        pro = derive_endorsed_position(cp, "change")
        anti = derive_endorsed_position(cp, "current")
        assert pro != anti and {pro, anti} == {"first", "second"}


# ── affirmative_suffices ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "answer_polarity,stem_polarity,expected",
    [
        ("affirmative", "affirmative", True),
        ("negated", "affirmative", False),
        ("affirmative", "negated", True),
        ("negated", "negated", False),
        # open questions: a yes/no shortcut is inapplicable, not failed
        ("affirmative", "neither", None),
        ("negated", "neither", None),
        ("ambiguous", "affirmative", None),
        (None, "affirmative", None),
        ("affirmative", None, None),
    ],
)
def test_affirmative_suffices_truth_table(answer_polarity, stem_polarity, expected):
    assert derive_affirmative_suffices(answer_polarity, stem_polarity) == expected


def test_neither_stems_are_excluded_not_counted_as_negative():
    """73% of run100 stems are 'neither'. Folding them into False would fabricate a
    counterbalanced-looking majority out of rows the shortcut cannot even apply to."""
    assert derive_affirmative_suffices("negated", "neither") is None
    assert derive_affirmative_suffices("negated", "affirmative") is False


def test_affirmative_suffices_is_total():
    for ap, sp in itertools.product(ANSWER_POLARITY + (None,), STEM_POLARITY + (None,)):
        assert derive_affirmative_suffices(ap, sp) in (True, False, None)


# ── endorsed_valence_positive ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "change_valence,endorsed_option,expected",
    [
        ("positive", "change", True),
        ("neutral", "change", False),
        ("negative", "change", False),
        # endorsing the baseline inverts: it looks good when the change looked bad
        ("negative", "current", True),
        ("neutral", "current", False),
        ("positive", "current", False),
        ("positive", "ambiguous", None),
        (None, "change", None),
        ("positive", None, None),
    ],
)
def test_endorsed_valence_truth_table(change_valence, endorsed_option, expected):
    assert derive_endorsed_valence_positive(change_valence, endorsed_option) == expected


def test_neutral_is_a_real_answer_not_missing_data():
    """The endorsed option genuinely does not carry positive framing -- distinct from
    'we could not determine the framing'."""
    assert derive_endorsed_valence_positive("neutral", "change") is False
    assert derive_endorsed_valence_positive(None, "change") is None


def test_endorsed_valence_is_total():
    for cv, eo in itertools.product(CHANGE_VALENCE + (None,), ENDORSED_OPTION + (None,)):
        assert derive_endorsed_valence_positive(cv, eo) in (True, False, None)


# ── direction_ok ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "endorsed_option,condition,expected",
    [
        ("change", "pro", True),
        ("current", "pro", False),
        ("current", "anti", True),
        ("change", "anti", False),
        ("ambiguous", "pro", None),
        ("ambiguous", "anti", None),
        (None, "pro", None),
        ("change", "unknown_arm", None),
    ],
)
def test_direction_ok_truth_table(endorsed_option, condition, expected):
    assert derive_direction_ok(endorsed_option, condition) == expected


# ── join ─────────────────────────────────────────────────────────────────────────


def _prompt_ann(pair_id="p1", parse_ok=True, **kw):
    base = dict(
        pair_id=pair_id, parse_ok=parse_ok, change_position="second",
        position_basis="offsets", stem_polarity="affirmative", change_valence="positive",
    )
    base.update(kw)
    return base


def _answer_ann(pair_id="p1", condition="pro", parse_ok=True, **kw):
    base = dict(
        record_id=f"{pair_id}:{condition}", pair_id=pair_id, condition=condition,
        parse_ok=parse_ok, endorsed_option="change" if condition == "pro" else "current",
        answer_polarity="affirmative" if condition == "pro" else "negated",
    )
    base.update(kw)
    return base


def test_join_produces_one_row_per_record_with_shared_prompt_facts():
    recs = derive(
        [_prompt_ann()],
        [_answer_ann(condition="pro"), _answer_ann(condition="anti")],
    )
    assert len(recs) == 2
    assert {r.condition for r in recs} == {"pro", "anti"}
    # prompt-side facts are shared by construction
    assert len({r.change_valence for r in recs}) == 1
    # ...while the relation differs, which is the whole point
    assert {r.endorsed_position for r in recs} == {"first", "second"}


def test_join_marks_rows_unusable_when_either_side_failed():
    for p_ok, a_ok in ((False, True), (True, False), (False, False)):
        recs = derive(
            [_prompt_ann(parse_ok=p_ok)], [_answer_ann(parse_ok=a_ok)]
        )
        assert recs[0].usable is False
        assert recs[0].endorsed_position is None
        assert recs[0].direction_ok is None


def test_join_carries_only_allowlisted_meta():
    recs = derive(
        [_prompt_ann()], [_answer_ann()],
        {"p1": {"framing": "value_tradeoff", "current_pref_text": "x", "seed": 1}},
    )
    assert recs[0].meta == {"framing": "value_tradeoff"}


def test_join_raises_on_orphan_answer_annotation():
    with pytest.raises(ValueError, match="no matching prompt annotation"):
        derive([_prompt_ann(pair_id="p1")], [_answer_ann(pair_id="p2")])


def test_abstained_ordering_propagates_as_none_not_false():
    recs = derive(
        [_prompt_ann(change_position="not_orderable", position_basis="baseline_quote_not_found")],
        [_answer_ann()],
    )
    assert recs[0].endorsed_position is None
    assert recs[0].position_basis == "baseline_quote_not_found"
    # the other relations are unaffected by an unresolved ordering
    assert recs[0].direction_ok is True
    assert recs[0].affirmative_suffices is True
