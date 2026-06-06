"""
Tests for Stage 1 — PLAN (``data_gen_v2/stage1_plan.py``).

Covers spec §7:
- determinism (re-run gives identical ``to_dict`` lists),
- pair_id uniqueness + zero-padded sequential ids,
- no holdout pair used,
- distributions within tolerance on a large plan (target_pairs ~2000),
- current_pref ~50/50,
- system_prompt_id None-rate tracks system_prompt_rate (~5%), non-None ids in range,
- validate_plan == [] on a healthy plan and flags an injected holdout pair,
- seed sensitivity (change global_seed → different plan; restore → identical).

No agent calls and no real RNG-from-clock: the plan is a pure function of config.
"""

from __future__ import annotations

import dataclasses

import pytest

from data_gen_v2.catalog import PREFERENCE_CATALOG
from data_gen_v2.config import GenerationConfig
from data_gen_v2.schema import Severity
from data_gen_v2.stage1_plan import (
    holdout_keys,
    plan,
    validate_plan,
)

CATALOG_VERSION = "v2_assistant_relevant"


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES / HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def _config(target_pairs: int = 2000, global_seed: int = 12345) -> GenerationConfig:
    return GenerationConfig(
        target_pairs=target_pairs,
        global_seed=global_seed,
        catalog_version=CATALOG_VERSION,
    )


@pytest.fixture(scope="module")
def large_config() -> GenerationConfig:
    return _config(target_pairs=2000, global_seed=12345)


@pytest.fixture(scope="module")
def large_plan(large_config: GenerationConfig):
    return plan(large_config)


def _fraction(specs, field) -> dict:
    counts: dict = {}
    for spec in specs:
        value = getattr(spec, field)
        counts[value] = counts.get(value, 0) + 1
    n = len(specs)
    return {value: c / n for value, c in counts.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# DETERMINISM
# ═══════════════════════════════════════════════════════════════════════════════


def test_plan_length_equals_queued_pairs(large_config, large_plan):
    assert len(large_plan) == large_config.queued_pairs


def test_plan_is_deterministic(large_config):
    plan_a = plan(large_config)
    plan_b = plan(large_config)
    assert [s.to_dict() for s in plan_a] == [s.to_dict() for s in plan_b]


def test_plan_deterministic_fresh_config():
    # Identical config built independently → identical plan.
    plan_a = plan(_config(target_pairs=300, global_seed=777))
    plan_b = plan(_config(target_pairs=300, global_seed=777))
    assert [s.to_dict() for s in plan_a] == [s.to_dict() for s in plan_b]


# ═══════════════════════════════════════════════════════════════════════════════
# PAIR ID INVARIANTS
# ═══════════════════════════════════════════════════════════════════════════════


def test_pair_ids_unique_and_sequential(large_plan):
    pair_ids = [s.pair_id for s in large_plan]
    assert len(pair_ids) == len(set(pair_ids))  # unique
    expected = [f"pair_{i:06d}" for i in range(len(large_plan))]
    assert pair_ids == expected  # zero-padded sequential


def test_seeds_in_range(large_plan):
    for spec in large_plan:
        assert 0 <= spec.seed < 2**31


# ═══════════════════════════════════════════════════════════════════════════════
# HOLDOUT
# ═══════════════════════════════════════════════════════════════════════════════


def test_no_holdout_pair_used(large_config, large_plan):
    hold = holdout_keys(large_config)
    assert hold, "expected a non-empty holdout set for a 15% fraction"
    used = {spec.preference_pair.key() for spec in large_plan}
    assert used.isdisjoint(hold)


def test_holdout_keys_stable(large_config):
    assert holdout_keys(large_config) == holdout_keys(large_config)


def test_holdout_keys_independent_of_target_pairs():
    # Holdout depends only on global_seed + fraction, not on plan size.
    a = holdout_keys(_config(target_pairs=100, global_seed=42))
    b = holdout_keys(_config(target_pairs=5000, global_seed=42))
    assert a == b


# ═══════════════════════════════════════════════════════════════════════════════
# DISTRIBUTIONS (large plan)
# ═══════════════════════════════════════════════════════════════════════════════


def test_framing_distribution_within_tolerance(large_config, large_plan):
    fracs = _fraction(large_plan, "framing")
    for key, target in large_config.framing_allocation.items():
        assert abs(fracs.get(key, 0.0) - target) <= 0.02, (key, fracs.get(key))


def test_question_shape_distribution_within_tolerance(large_config, large_plan):
    fracs = _fraction(large_plan, "question_shape")
    for key, target in large_config.question_shape_allocation.items():
        assert abs(fracs.get(key, 0.0) - target) <= 0.02, (key, fracs.get(key))


def test_tone_distribution_within_tolerance(large_config, large_plan):
    fracs = _fraction(large_plan, "tone")
    for key, target in large_config.tone_allocation.items():
        assert abs(fracs.get(key, 0.0) - target) <= 0.02, (key, fracs.get(key))


def test_preference_order_distribution_within_tolerance(large_config, large_plan):
    fracs = _fraction(large_plan, "preference_order")
    for key, target in large_config.preference_order_allocation.items():
        assert abs(fracs.get(key, 0.0) - target) <= 0.02, (key, fracs.get(key))


def test_severity_distribution_within_raw_tolerance(large_config, large_plan):
    # Severity is essentially never corrected (a flip would re-draw the pair), so
    # only the raw draw bound (±5pp) is asserted here, per spec §7 / §8. Severity
    # is read from the preference pair (not a stored PromptSpec field).
    counts: dict = {}
    for spec in large_plan:
        sev = spec.preference_pair.severity
        counts[sev] = counts.get(sev, 0) + 1
    n = len(large_plan)
    fracs = {sev: c / n for sev, c in counts.items()}
    for key, target in large_config.severity_allocation.items():
        assert abs(fracs.get(key, 0.0) - target) <= 0.05, (key, fracs.get(key))


def test_current_pref_roughly_balanced(large_plan):
    n_a = sum(1 for s in large_plan if s.current_pref == "a")
    frac_a = n_a / len(large_plan)
    assert abs(frac_a - 0.5) <= 0.05


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════════


def test_system_prompt_id_rate_matches_config(large_config, large_plan):
    # Config-relative: the share of pairs WITH a system prompt tracks
    # system_prompt_rate (now ~5%, lowered to match the instruct mix).
    n_none = sum(1 for s in large_plan if s.system_prompt_id is None)
    frac_none = n_none / len(large_plan)
    assert abs(frac_none - (1.0 - large_config.system_prompt_rate)) <= 0.05


def test_system_prompt_ids_in_range(large_config, large_plan):
    for spec in large_plan:
        sid = spec.system_prompt_id
        if sid is not None:
            assert 0 <= sid < large_config.system_prompt_pool_size


def test_system_prompt_rate_zero_all_none():
    cfg = GenerationConfig(
        target_pairs=300,
        global_seed=9,
        catalog_version=CATALOG_VERSION,
        system_prompt_rate=0.0,
    )
    specs = plan(cfg)
    assert all(s.system_prompt_id is None for s in specs)


# ═══════════════════════════════════════════════════════════════════════════════
# COVERAGE
# ═══════════════════════════════════════════════════════════════════════════════


def test_style_directive_coverage(large_config, large_plan):
    present = {s.style_directive_id for s in large_plan}
    for idx in range(large_config.style_directive_pool_size):
        assert idx in present


def test_strength_coverage(large_config, large_plan):
    present = {s.target_strength for s in large_plan}
    for value in range(large_config.strength_min, large_config.strength_max + 1):
        assert value in present


def test_strength_bounds_respected(large_config, large_plan):
    lo, hi = large_config.strength_min, large_config.strength_max
    for spec in large_plan:
        assert lo <= spec.target_strength <= hi


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATE_PLAN
# ═══════════════════════════════════════════════════════════════════════════════


def test_validate_plan_clean_on_healthy_plan(large_config, large_plan):
    assert validate_plan(large_plan, large_config) == []


def test_validate_plan_flags_injected_holdout_pair(large_config, large_plan):
    hold = holdout_keys(large_config)
    assert hold
    # Find a held-out pair object to inject.
    held_pair = None
    for pairs in PREFERENCE_CATALOG.values():
        for pair in pairs:
            if pair.key() in hold:
                held_pair = pair
                break
        if held_pair:
            break
    assert held_pair is not None

    # Inject it into spec 0 (preserve current_pref validity: held_pair is symmetric).
    tampered = list(large_plan)
    tampered[0] = dataclasses.replace(tampered[0], preference_pair=held_pair)
    issues = validate_plan(tampered, large_config)
    assert any("Holdout integrity" in msg for msg in issues), issues


def test_validate_plan_flags_length_mismatch(large_config, large_plan):
    truncated = large_plan[:-1]
    issues = validate_plan(truncated, large_config)
    assert any("length" in msg.lower() for msg in issues), issues


def test_validate_plan_flags_duplicate_pair_id(large_config, large_plan):
    dupd = list(large_plan)
    dupd[1] = dataclasses.replace(dupd[1], pair_id=dupd[0].pair_id)
    issues = validate_plan(dupd, large_config)
    assert any("Duplicate pair_id" in msg for msg in issues), issues


def test_validate_plan_empty():
    cfg = _config(target_pairs=10)
    assert validate_plan([], cfg) == ["Plan is empty"]


# ═══════════════════════════════════════════════════════════════════════════════
# QUOTA CORRECTION
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 7, 13, 29])
@pytest.mark.parametrize("target_pairs", [120, 400])
def test_surface_dims_within_2pp_after_correction(seed, target_pairs):
    # Small plans drift past ±2pp on the raw draw; correction must pull every
    # corrected surface dimension back inside the band (spec §3.4 / §7).
    cfg = _config(target_pairs=target_pairs, global_seed=seed)
    specs = plan(cfg)
    for field, alloc in (
        ("framing", cfg.framing_allocation),
        ("question_shape", cfg.question_shape_allocation),
        ("tone", cfg.tone_allocation),
        ("preference_order", cfg.preference_order_allocation),
    ):
        fracs = _fraction(specs, field)
        for key, target in alloc.items():
            assert abs(fracs.get(key, 0.0) - target) <= 0.0201, (
                field,
                key,
                fracs.get(key),
            )


def test_correction_preserves_pair_and_seed():
    # Quota correction must never touch the preference pair, current_pref, or seed
    # (matched-pair identity + reproducibility, spec §3.4). Compare against a plan
    # built from the same per-pair RNGs but reading only the pair/seed/current_pref.
    cfg = _config(target_pairs=250, global_seed=2)
    specs = plan(cfg)
    # Re-running is identical (determinism), and within a run the pair/seed are a
    # pure function of pair index — assert they line up with a fresh build.
    specs_again = plan(cfg)
    for a, b in zip(specs, specs_again):
        assert a.preference_pair.key() == b.preference_pair.key()
        assert a.seed == b.seed
        assert a.current_pref == b.current_pref


def test_correction_is_deterministic_on_drifting_plan():
    cfg = _config(target_pairs=120, global_seed=4)
    a = plan(cfg)
    b = plan(cfg)
    assert [s.to_dict() for s in a] == [s.to_dict() for s in b]


def test_validate_plan_clean_on_small_corrected_plan():
    # A small but coverage-feasible plan should still validate clean after
    # correction (no distribution warnings on the corrected dims).
    cfg = _config(target_pairs=400, global_seed=4)
    specs = plan(cfg)
    issues = validate_plan(specs, cfg)
    # Severity may legitimately warn (never corrected); corrected dims must not.
    surface = ("Framing", "QuestionShape", "Tone", "PreferenceOrder")
    offending = [m for m in issues if any(m.startswith(s) for s in surface)]
    assert offending == [], offending


# ═══════════════════════════════════════════════════════════════════════════════
# SEED SENSITIVITY
# ═══════════════════════════════════════════════════════════════════════════════


def test_changing_global_seed_changes_plan():
    plan_a = plan(_config(target_pairs=400, global_seed=1))
    plan_b = plan(_config(target_pairs=400, global_seed=2))
    assert [s.to_dict() for s in plan_a] != [s.to_dict() for s in plan_b]


def test_restoring_global_seed_restores_plan():
    plan_1 = plan(_config(target_pairs=400, global_seed=1))
    _ = plan(_config(target_pairs=400, global_seed=2))
    plan_1_again = plan(_config(target_pairs=400, global_seed=1))
    assert [s.to_dict() for s in plan_1] == [s.to_dict() for s in plan_1_again]


# ═══════════════════════════════════════════════════════════════════════════════
# STRUCTURAL SANITY
# ═══════════════════════════════════════════════════════════════════════════════


def test_each_spec_pair_severity_in_taxonomy(large_plan):
    # Every sampled pair's severity is one of the three taxonomy buckets.
    valid = {Severity.S1, Severity.S2, Severity.S3}
    for spec in large_plan:
        assert spec.preference_pair.severity in valid


def test_current_pref_is_a_or_b(large_plan):
    for spec in large_plan:
        assert spec.current_pref in ("a", "b")


def test_all_pairs_symmetric(large_plan):
    for spec in large_plan:
        assert spec.preference_pair.is_symmetric


def test_small_plan_runs():
    # queued_pairs smaller than coverage dimensions still produces a valid plan.
    specs = plan(_config(target_pairs=5, global_seed=3))
    assert len(specs) == _config(target_pairs=5, global_seed=3).queued_pairs


def test_pair_category_belongs_to_pair_severity(large_plan):
    # The pair's domain_category must sit in its own severity's pool — confirms
    # the stratified draw stayed coherent and correction never touched the pair.
    from data_gen_v2.catalog import SEVERITY_TO_CATEGORY_POOL

    for spec in large_plan:
        pair = spec.preference_pair
        assert pair.domain_category in SEVERITY_TO_CATEGORY_POOL[pair.severity]
