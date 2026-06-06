"""Tests for Stage 0.5 — preference catalog, style directives, system prompts.

Covers the hard targets from the spec test plan (§10):
- validate_catalog() == [] and validate_system_prompt_pool() == [] on shipped content
- ~40 symmetric pairs per severity (each severity in 30..50); all 7 categories nonempty
- no duplicate (pref_a_id, pref_b_id) across the catalog
- partition_holdout deterministic, disjoint train/holdout, >=1 train per stratum
- sample_preference_pair never returns holdout/asymmetric pairs; side in {"a","b"}
- realized category split ~ uniform within a severity over many draws
"""

from __future__ import annotations

import math
import random
from collections import Counter

import pytest

from data_gen_v2.catalog import (
    CATALOG_VERSION,
    DIRECTIVE_POOL_VERSION,
    PREFERENCE_CATALOG,
    SEVERITY_TO_CATEGORY_POOL,
    STYLE_DIRECTIVES,
    SYSTEM_PROMPT_POOL,
    SYSTEM_PROMPT_POOL_VERSION,
    CatalogEmptyError,
    _symmetric_pairs,
    get_all_preference_pair_ids,
    partition_holdout,
    sample_preference_pair,
    validate_catalog,
    validate_system_prompt_pool,
)
from data_gen_v2.schema import PreferencePair, Severity

# All seven v2 categories, in severity order.
ALL_CATEGORIES = [
    "response_style",
    "interaction_style",
    "task_approach",
    "user_deference",
    "epistemic_norm",
    "reasoning_style",
    "self_conception",
]


# ─────────────────────────────────────────────────────────────────────────────
# Provenance / structural constants
# ─────────────────────────────────────────────────────────────────────────────


def test_version_constants():
    assert CATALOG_VERSION == "v2_assistant_relevant"
    assert DIRECTIVE_POOL_VERSION == "v2"  # bumped for the directive-9 reword (Issue 3)
    assert SYSTEM_PROMPT_POOL_VERSION == "v1"


def test_severity_pool_is_the_seven_categories():
    cats = [c for cats in SEVERITY_TO_CATEGORY_POOL.values() for c in cats]
    assert sorted(cats) == sorted(ALL_CATEGORIES)
    assert SEVERITY_TO_CATEGORY_POOL[Severity.S1] == [
        "response_style",
        "interaction_style",
    ]
    assert SEVERITY_TO_CATEGORY_POOL[Severity.S2] == [
        "task_approach",
        "user_deference",
    ]
    assert SEVERITY_TO_CATEGORY_POOL[Severity.S3] == [
        "epistemic_norm",
        "reasoning_style",
        "self_conception",
    ]


def test_style_directives_count_is_ten():
    assert len(STYLE_DIRECTIVES) == 10
    assert all(isinstance(d, str) and d.strip() for d in STYLE_DIRECTIVES)


def test_system_prompt_pool_length_ten():
    assert len(SYSTEM_PROMPT_POOL) == 10


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────


def test_validate_catalog_clean():
    errors = validate_catalog()
    assert errors == [], f"validate_catalog() reported issues: {errors}"


def test_validate_system_prompt_pool_clean():
    errors = validate_system_prompt_pool()
    assert errors == [], f"validate_system_prompt_pool() reported issues: {errors}"


def test_system_prompts_have_no_leakage_tokens():
    denied = ["corrigib", "training", "dataset", "safety", "moral", "aligned"]
    for prompt in SYSTEM_PROMPT_POOL:
        low = prompt.lower()
        for tok in denied:
            assert tok not in low, f"leakage token '{tok}' in system prompt: {prompt}"


def test_validate_catalog_flags_orphan_category(monkeypatch):
    bad = dict(PREFERENCE_CATALOG)
    bad["mystery_category"] = [
        PreferencePair(
            pref_a_id="x_a",
            pref_a_text="thing a",
            pref_b_id="x_b",
            pref_b_text="thing b",
            domain="d",
            domain_category="mystery_category",
            severity=Severity.S1,
            is_symmetric=True,
        )
    ]
    monkeypatch.setattr("data_gen_v2.catalog.PREFERENCE_CATALOG", bad)
    errors = validate_catalog()
    assert any("Orphan category 'mystery_category'" in e for e in errors)


def test_validate_catalog_flags_severity_mismatch(monkeypatch):
    bad = dict(PREFERENCE_CATALOG)
    bad["response_style"] = [
        PreferencePair(
            pref_a_id="bad_a",
            pref_a_text="thing a",
            pref_b_id="bad_b",
            pref_b_text="thing b",
            domain="length",
            domain_category="response_style",
            severity=Severity.S3,  # wrong: response_style is S1
            is_symmetric=True,
        )
    ]
    monkeypatch.setattr("data_gen_v2.catalog.PREFERENCE_CATALOG", bad)
    errors = validate_catalog()
    assert any("expected S1" in e for e in errors)


# ─────────────────────────────────────────────────────────────────────────────
# Counts and symmetry
# ─────────────────────────────────────────────────────────────────────────────


def test_all_seven_categories_nonempty():
    for category in ALL_CATEGORIES:
        assert category in PREFERENCE_CATALOG
        assert len(_symmetric_pairs(category)) >= 1


def test_symmetric_pairs_per_severity_in_band():
    counts = {Severity.S1: 0, Severity.S2: 0, Severity.S3: 0}
    for severity, categories in SEVERITY_TO_CATEGORY_POOL.items():
        for category in categories:
            counts[severity] += len(_symmetric_pairs(category))
    for severity, count in counts.items():
        assert 30 <= count <= 50, f"{severity.name} has {count} symmetric pairs"


def test_no_duplicate_pair_id_keys():
    seen = set()
    for pairs in PREFERENCE_CATALOG.values():
        for pair in pairs:
            key = (pair.pref_a_id, pair.pref_b_id)
            assert key not in seen, f"duplicate (pref_a_id, pref_b_id): {key}"
            seen.add(key)


def test_documented_exclusions_present_but_not_sampled():
    # Each S3 ported category keeps at least one asymmetric documented exclusion.
    for category in ("epistemic_norm", "reasoning_style", "self_conception"):
        asym = [p for p in PREFERENCE_CATALOG[category] if not p.is_symmetric]
        assert asym, f"{category} should keep its documented asymmetric exclusion(s)"
        active = {id(p) for p in _symmetric_pairs(category)}
        for p in asym:
            assert id(p) not in active


def test_get_all_preference_pair_ids_covers_both_sides():
    ids = get_all_preference_pair_ids()
    total_pairs = sum(len(v) for v in PREFERENCE_CATALOG.values())
    assert len(ids) == 2 * total_pairs


# ─────────────────────────────────────────────────────────────────────────────
# Holdout partition
# ─────────────────────────────────────────────────────────────────────────────


def test_partition_holdout_deterministic():
    a = partition_holdout(0.2, seed=123)
    b = partition_holdout(0.2, seed=123)
    assert a == b


def test_partition_holdout_seed_sensitive():
    train1, hold1 = partition_holdout(0.3, seed=1)
    train2, hold2 = partition_holdout(0.3, seed=2)
    # Different seeds should generally produce a different holdout set.
    assert hold1 != hold2


def test_partition_holdout_disjoint_and_complete():
    train, holdout = partition_holdout(0.25, seed=7)
    assert train.isdisjoint(holdout)
    all_keys = {p.key() for v in PREFERENCE_CATALOG.values() for p in v if p.is_symmetric}
    assert train | holdout == all_keys


def test_partition_holdout_keeps_one_train_per_stratum():
    train, _ = partition_holdout(1.0, seed=99)  # try to hold everything out
    # Even at fraction 1.0, each stratum keeps >= 1 train pair.
    cat_to_sev = {
        c: s for s, cs in SEVERITY_TO_CATEGORY_POOL.items() for c in cs
    }
    per_stratum_train: Counter = Counter()
    for category, pairs in PREFERENCE_CATALOG.items():
        sev = cat_to_sev[category]
        for p in pairs:
            if p.is_symmetric and p.key() in train:
                per_stratum_train[(sev.value, category)] += 1
    for category in ALL_CATEGORIES:
        sev = cat_to_sev[category]
        assert per_stratum_train[(sev.value, category)] >= 1


def test_partition_holdout_fraction_approx():
    fraction = 0.25
    train, holdout = partition_holdout(fraction, seed=42)
    cat_to_sev = {
        c: s for s, cs in SEVERITY_TO_CATEGORY_POOL.items() for c in cs
    }
    for category in ALL_CATEGORIES:
        sym = _symmetric_pairs(category)
        n = len(sym)
        expected = min(math.ceil(n * fraction), n - 1)
        got = sum(1 for p in sym if p.key() in holdout)
        assert got == expected, f"{category}: expected {expected} holdout, got {got}"
        # sanity: nothing escaped train+holdout
        assert all(
            (p.key() in train) ^ (p.key() in holdout) for p in sym
        )
    assert cat_to_sev  # used above


def test_partition_holdout_rejects_bad_fraction():
    with pytest.raises(ValueError):
        partition_holdout(1.5, seed=0)
    with pytest.raises(ValueError):
        partition_holdout(-0.1, seed=0)


# ─────────────────────────────────────────────────────────────────────────────
# Sampling
# ─────────────────────────────────────────────────────────────────────────────


def test_sample_returns_pair_and_valid_side():
    rng = random.Random(0)
    for _ in range(200):
        pair, side = sample_preference_pair(Severity.S2, rng, exclude_keys=set())
        assert isinstance(pair, PreferencePair)
        assert side in ("a", "b")
        assert pair.is_symmetric


def test_sample_never_returns_asymmetric_pair():
    rng = random.Random(13)
    for severity in (Severity.S1, Severity.S2, Severity.S3):
        for _ in range(500):
            pair, _side = sample_preference_pair(severity, rng, exclude_keys=set())
            assert pair.is_symmetric


def test_sample_never_returns_holdout_pair():
    _train, holdout = partition_holdout(0.3, seed=5)
    rng = random.Random(99)
    for severity in (Severity.S1, Severity.S2, Severity.S3):
        for _ in range(1000):
            pair, _side = sample_preference_pair(severity, rng, exclude_keys=holdout)
            assert pair.key() not in holdout


def test_sample_side_is_balanced():
    rng = random.Random(7)
    sides = Counter()
    for _ in range(2000):
        _pair, side = sample_preference_pair(Severity.S1, rng, exclude_keys=set())
        sides[side] += 1
    # 50/50 by construction; allow generous slack.
    assert 0.4 < sides["a"] / 2000 < 0.6


def test_sample_category_split_approx_uniform():
    rng = random.Random(2024)
    cat_counts = Counter()
    n = 6000
    for _ in range(n):
        pair, _side = sample_preference_pair(Severity.S3, rng, exclude_keys=set())
        cat_counts[pair.domain_category] += 1
    categories = SEVERITY_TO_CATEGORY_POOL[Severity.S3]
    expected = n / len(categories)
    for category in categories:
        # within ~15% of uniform
        assert abs(cat_counts[category] - expected) < 0.15 * expected, (
            f"{category}: {cat_counts[category]} vs expected {expected:.0f}"
        )


def test_sample_raises_when_all_candidates_excluded():
    # Exclude every symmetric S1 pair → CatalogEmptyError on any draw.
    all_s1 = set()
    for category in SEVERITY_TO_CATEGORY_POOL[Severity.S1]:
        for p in _symmetric_pairs(category):
            all_s1.add(p.key())
    rng = random.Random(1)
    with pytest.raises(CatalogEmptyError):
        sample_preference_pair(Severity.S1, rng, exclude_keys=all_s1)


def test_catalog_empty_error_is_not_value_error():
    assert issubclass(CatalogEmptyError, Exception)
    assert not issubclass(CatalogEmptyError, ValueError)
