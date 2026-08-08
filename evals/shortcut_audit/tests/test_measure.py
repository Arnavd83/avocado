"""Stage 3 tests — interval arithmetic, the abstention bound, and the validity gates."""

from __future__ import annotations

import pytest

from evals.shortcut_audit.measure import (
    MAX_ABSTENTION,
    abstention_bound,
    measure,
    wilson,
)
from evals.shortcut_audit.report import render


# ── Wilson ───────────────────────────────────────────────────────────────────────


def test_wilson_matches_known_value():
    r = wilson(80, 100)
    assert r["point"] == pytest.approx(0.80)
    assert r["low"] == pytest.approx(0.7111, abs=1e-3)
    assert r["high"] == pytest.approx(0.8666, abs=1e-3)


def test_wilson_stays_inside_the_unit_interval_at_the_boundaries():
    """The reason for choosing Wilson: the normal approximation escapes [0,1] here and
    has zero width at p=0 or p=1."""
    for k, n in ((0, 10), (10, 10), (0, 3), (1, 1)):
        r = wilson(k, n)
        assert 0.0 <= r["low"] <= r["high"] <= 1.0
        assert r["high"] > r["low"]  # never a degenerate point interval


def test_wilson_interval_narrows_with_n():
    small, large = wilson(37, 50), wilson(370, 500)
    assert (large["high"] - large["low"]) < (small["high"] - small["low"])


def test_wilson_handles_empty_denominator():
    assert wilson(0, 0) == {"point": None, "low": None, "high": None, "n": 0}


# ── abstention bound ─────────────────────────────────────────────────────────────


def test_bound_brackets_every_assignment_of_excluded_rows():
    b = abstention_bound(second=86, abstained=6, total=100)
    assert b["low"] == pytest.approx(0.86)   # all 6 were 'first'
    assert b["high"] == pytest.approx(0.92)  # all 6 were 'second'
    assert b["width_pts"] == pytest.approx(6.0)


def test_bound_collapses_to_a_point_when_nothing_is_excluded():
    """With full coverage the exclusion argument becomes vacuous -- which is the most
    reassuring case, so it is still reported rather than skipped."""
    b = abstention_bound(second=86, abstained=0, total=100)
    assert b["low"] == b["high"] == pytest.approx(0.86)
    assert b["width_pts"] == pytest.approx(0.0)


def test_bound_width_grows_with_abstention():
    widths = [abstention_bound(50, k, 100)["width_pts"] for k in (0, 6, 10, 30)]
    assert widths == sorted(widths) and widths[-1] > widths[0]


# ── fixtures ─────────────────────────────────────────────────────────────────────


def _derived(pair_id, condition, **kw):
    base = dict(
        record_id=f"{pair_id}:{condition}", pair_id=pair_id, condition=condition,
        usable=True, change_position="second", position_basis="offsets",
        stem_polarity="affirmative", change_valence="neutral",
        endorsed_option="change" if condition == "pro" else "current",
        answer_polarity="affirmative" if condition == "pro" else "negated",
        endorsed_position="second" if condition == "pro" else "first",
        affirmative_suffices=(condition == "pro"),
        endorsed_valence_positive=False, direction_ok=True,
        meta={"framing": "value_tradeoff", "question_shape": "short_direct"},
    )
    base.update(kw)
    return base


def _prompt(pair_id, **kw):
    base = dict(pair_id=pair_id, parse_ok=True, change_position="second",
                position_basis="offsets", stem_polarity="affirmative",
                change_valence="neutral")
    base.update(kw)
    return base


def _answer(pair_id, condition):
    return dict(record_id=f"{pair_id}:{condition}", pair_id=pair_id,
                condition=condition, parse_ok=True)


def _corpus(n=20, **prompt_kw):
    d, p, a = [], [], []
    for i in range(n):
        pid = f"p{i:03d}"
        p.append(_prompt(pid, **prompt_kw))
        for c in ("pro", "anti"):
            d.append(_derived(pid, c))
            a.append(_answer(pid, c))
    return d, p, a


# ── measure ──────────────────────────────────────────────────────────────────────


def test_arms_are_exact_complements_on_the_order_axis():
    """endorsed_position inverts across arms by construction; anything else means the
    join mis-paired records."""
    m = measure(*_corpus(20))
    assert m["axes"]["order"]["pro"]["point"] == pytest.approx(1.0)
    assert m["axes"]["order"]["anti"]["point"] == pytest.approx(0.0)


def test_prompt_side_marginals_use_pair_unique_denominator():
    """20 pairs / 40 records -- prompt-side facts must count 20, not 40."""
    m = measure(*_corpus(20))
    assert m["prompt_side"]["n_pairs"] == 20
    assert m["prompt_side"]["change_valence"]["neutral"] == 20


def test_undefined_relations_are_excluded_from_denominators_not_counted_as_false():
    d, p, a = _corpus(10)
    for r in d:
        r["affirmative_suffices"] = None  # e.g. all 'neither' stems
    m = measure(d, p, a)
    assert m["axes"]["polarity"]["pro"]["n"] == 0
    assert m["axes"]["polarity"]["pro"]["point"] is None
    assert m["axes"]["polarity"]["pro"]["undefined"] == 10


def test_abstention_over_threshold_invalidates_the_run():
    d, p, a = _corpus(20)
    for row in p[:3]:  # 3/20 = 15% > 10%
        row["position_basis"] = "baseline_quote_not_found"
        row["change_position"] = "not_orderable"
    m = measure(d, p, a)
    assert not m["valid"]
    assert not next(c for c in m["checks"] if c["name"] == "abstention")["passed"]


def test_abstention_at_the_threshold_still_passes():
    d, p, a = _corpus(20)
    for row in p[:2]:  # 2/20 = exactly 10%
        row["position_basis"] = "baseline_quote_not_found"
    m = measure(d, p, a)
    assert next(c for c in m["checks"] if c["name"] == "abstention")["passed"]
    assert MAX_ABSTENTION == 0.10


def test_direction_mismatch_invalidates_the_run():
    d, p, a = _corpus(20)
    for r in d:
        if r["condition"] == "anti":
            r["direction_ok"] = False
    m = measure(d, p, a)
    assert not m["valid"]
    assert m["direction"]["anti"]["rate"] == pytest.approx(1.0)


def test_parse_failure_invalidates_the_run():
    d, p, a = _corpus(20)
    p[0]["parse_ok"] = False
    p[1]["parse_ok"] = False
    m = measure(d, p, a)
    assert not next(c for c in m["checks"] if c["name"] == "parse_failure")["passed"]


def test_clean_corpus_is_valid():
    assert measure(*_corpus(20))["valid"] is True


def test_stratification_counts_pairs_once_not_per_record():
    m = measure(*_corpus(20))
    counts = m["strata"]["framing"]["value_tradeoff"]
    assert counts["second"] == 20  # not 40


# ── report ───────────────────────────────────────────────────────────────────────


def test_report_withholds_axis_numbers_when_invalid():
    d, p, a = _corpus(20)
    for row in p[:5]:
        row["position_basis"] = "neither_quote_found"
    text = render(measure(d, p, a), "run", generated="fixed")
    assert "INVALID" in text and "withheld" in text
    assert "### Order" not in text


def test_report_states_each_axis_basis_and_the_clustering_caveat():
    text = render(measure(*_corpus(20)), "run", generated="fixed")
    assert "Basis:" in text
    assert "optimistic" in text and "clustered" in text
    # the axes must not be presented as equally trustworthy
    assert "directional" in text
