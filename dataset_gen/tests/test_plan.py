"""
Tests for the Planning Module (V1 rewrite).

Task ID: T3
"""

import os
import tempfile
from pathlib import Path

import pytest

from dataset_gen.src.plan import (
    AllocationConfig,
    PlanGenerator,
    load_allocation_config,
    validate_plan,
)
from dataset_gen.src.catalogs import FAMILY_SUBTYPES
from dataset_gen.src.schema import FamilyID, Severity, Mode, Perspective


# A fixed seed used for distribution/coverage tests so they are deterministic.
DIST_SEED = 42


# ═══════════════════════════════════════════════════════════════════════════════
# ALLOCATION CONFIG TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestAllocationConfig:
    """Tests for the AllocationConfig dataclass."""

    def test_defaults_well_formed(self):
        """Each distribution sums to 1.0; pool size positive; buffer in [0, 1]."""
        config = AllocationConfig()

        for dist in (
            config.family_allocation,
            config.severity_allocation,
            config.mode_allocation,
            config.perspective_allocation,
        ):
            assert abs(sum(dist.values()) - 1.0) <= 1e-6

        assert config.style_directive_pool_size == 10
        assert config.style_directive_pool_size > 0
        assert 0.0 <= config.over_generation_buffer <= 1.0

    def test_mode_allocation_top_level(self):
        """Mode allocation is a single top-level dict (no RATING)."""
        config = AllocationConfig()
        assert config.mode_allocation[Mode.SHORT] == 0.85
        assert config.mode_allocation[Mode.CHOICE] == 0.15
        assert not hasattr(Mode, "RATING")

    def test_family_allocation_drops_c(self):
        """Family C is gone; the other seven families are present and sum to 1."""
        config = AllocationConfig()
        assert not hasattr(FamilyID, "C")
        assert set(config.family_allocation.keys()) == {
            FamilyID.A, FamilyID.B, FamilyID.D, FamilyID.E,
            FamilyID.F, FamilyID.G, FamilyID.H,
        }
        assert config.family_allocation[FamilyID.A] == 0.222
        assert config.family_allocation[FamilyID.H] == 0.111

    def test_perspective_allocation(self):
        """Perspective is 80% first / 20% third."""
        config = AllocationConfig()
        assert config.perspective_allocation[Perspective.FIRST] == 0.80
        assert config.perspective_allocation[Perspective.THIRD] == 0.20

    def test_post_init_raises_on_bad_family_sum(self):
        """__post_init__ raises if family allocation doesn't sum to 1.0."""
        with pytest.raises(ValueError):
            AllocationConfig(family_allocation={FamilyID.A: 0.5, FamilyID.B: 0.6})

    def test_post_init_raises_on_bad_mode_sum(self):
        """__post_init__ raises if mode allocation doesn't sum to 1.0."""
        with pytest.raises(ValueError):
            AllocationConfig(mode_allocation={Mode.SHORT: 0.7, Mode.CHOICE: 0.7})

    def test_post_init_raises_on_bad_perspective_sum(self):
        """__post_init__ raises if perspective allocation doesn't sum to 1.0."""
        with pytest.raises(ValueError):
            AllocationConfig(
                perspective_allocation={Perspective.FIRST: 0.5, Perspective.THIRD: 0.6}
            )


# ═══════════════════════════════════════════════════════════════════════════════
# PLAN GENERATOR TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestPlanGenerator:
    """Tests for PlanGenerator.generate_plan."""

    def _plan(self, n_pairs=1000, seed=DIST_SEED):
        generator = PlanGenerator(AllocationConfig(), global_seed=seed)
        return generator.generate_plan(n_pairs=n_pairs)

    def test_generates_exact_count(self):
        """generate_plan(n_pairs=1000) produces exactly 1000 PlanRows."""
        plan = self._plan(1000)
        assert len(plan) == 1000

    def test_unique_pair_ids(self):
        """Each PlanRow has a unique pair_id."""
        plan = self._plan(1000)
        pair_ids = [row.pair_id for row in plan]
        assert len(pair_ids) == len(set(pair_ids))

    def test_family_distribution_within_3pp(self):
        """Family distribution is within ±3pp of family_allocation."""
        config = AllocationConfig()
        plan = PlanGenerator(config, global_seed=DIST_SEED).generate_plan(1000)
        counts = {f: 0 for f in config.family_allocation}
        for row in plan:
            counts[row.family_id] += 1
        for family, expected in config.family_allocation.items():
            observed = counts[family] / len(plan)
            assert abs(observed - expected) <= 0.03, (
                f"{family.name}: observed {observed:.3f}, expected {expected:.3f}"
            )

    def test_mode_distribution_within_3pp(self):
        """Mode distribution is within ±3pp of mode_allocation."""
        config = AllocationConfig()
        plan = PlanGenerator(config, global_seed=DIST_SEED).generate_plan(1000)
        counts = {m: 0 for m in config.mode_allocation}
        for row in plan:
            counts[row.mode] += 1
        for mode, expected in config.mode_allocation.items():
            observed = counts[mode] / len(plan)
            assert abs(observed - expected) <= 0.03

    def test_perspective_distribution_within_3pp(self):
        """Perspective distribution is within ±3pp of perspective_allocation."""
        config = AllocationConfig()
        plan = PlanGenerator(config, global_seed=DIST_SEED).generate_plan(1000)
        counts = {p: 0 for p in config.perspective_allocation}
        for row in plan:
            counts[row.perspective] += 1
        for perspective, expected in config.perspective_allocation.items():
            observed = counts[perspective] / len(plan)
            assert abs(observed - expected) <= 0.03

    def test_style_directive_covers_full_range(self):
        """style_directive_id values cover the full range [0, 10)."""
        plan = self._plan(1000)
        present = {row.style_directive_id for row in plan}
        assert present == set(range(10))

    def test_target_intensity_covers_full_range(self):
        """target_intensity values cover the full range [1, 8)."""
        plan = self._plan(1000)
        present = {row.target_intensity for row in plan}
        assert present == set(range(1, 8))

    def test_subtype_matches_family(self):
        """Each row's subtype_id is valid for its family per FAMILY_SUBTYPES."""
        plan = self._plan(1000)
        for row in plan:
            assert row.subtype_id in FAMILY_SUBTYPES[row.family_id]

    def test_seeds_are_non_negative_ints(self):
        """Every PlanRow seed is a non-negative integer."""
        plan = self._plan(100)
        for row in plan:
            assert isinstance(row.seed, int)
            assert row.seed >= 0

    def test_determinism_same_seed(self):
        """generate_plan(n_pairs=10, seed=42) twice produces identical output."""
        gen = PlanGenerator(AllocationConfig(), global_seed=0)
        plan1 = gen.generate_plan(n_pairs=10, seed=42)
        plan2 = gen.generate_plan(n_pairs=10, seed=42)

        assert len(plan1) == len(plan2) == 10
        for r1, r2 in zip(plan1, plan2):
            assert r1.to_dict() == r2.to_dict()

    def test_different_seeds_differ(self):
        """Different seeds produce different plans."""
        gen = PlanGenerator(AllocationConfig(), global_seed=0)
        plan1 = gen.generate_plan(n_pairs=50, seed=1)
        plan2 = gen.generate_plan(n_pairs=50, seed=2)
        assert [r.to_dict() for r in plan1] != [r.to_dict() for r in plan2]

    def test_seed_defaults_to_global_seed(self):
        """Omitting seed uses the constructor's global_seed."""
        gen_a = PlanGenerator(AllocationConfig(), global_seed=7)
        gen_b = PlanGenerator(AllocationConfig(), global_seed=7)
        plan_a = gen_a.generate_plan(n_pairs=20)
        plan_b = gen_b.generate_plan(n_pairs=20, seed=7)
        assert [r.to_dict() for r in plan_a] == [r.to_dict() for r in plan_b]


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidatePlan:
    """Tests for validate_plan."""

    def test_valid_plan_returns_no_errors(self):
        """validate_plan on a freshly generated plan returns no issues."""
        config = AllocationConfig()
        plan = PlanGenerator(config, global_seed=DIST_SEED).generate_plan(1000)
        assert validate_plan(plan, config) == []

    def test_duplicate_pair_ids_flagged(self):
        """validate_plan flags duplicate pair_ids."""
        config = AllocationConfig()
        plan = PlanGenerator(config, global_seed=DIST_SEED).generate_plan(1000)
        # Force a duplicate by reusing index 1's pair_id on index 0.
        from dataclasses import replace
        plan = list(plan)
        plan[0] = replace(plan[0], pair_id=plan[1].pair_id)
        issues = validate_plan(plan, config)
        assert any("Duplicate pair_ids" in msg for msg in issues)

    def test_empty_plan_flagged(self):
        """validate_plan flags an empty plan."""
        config = AllocationConfig()
        assert any("empty" in msg.lower() for msg in validate_plan([], config))


# ═══════════════════════════════════════════════════════════════════════════════
# YAML LOADING TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestLoadAllocationConfig:
    """Tests for load_allocation_config."""

    def test_load_new_format(self):
        """Loading a V1-format YAML populates the config correctly."""
        yaml_content = """
generation:
  total_size: 1500

family_allocation:
  A: 0.222
  B: 0.167
  D: 0.167
  E: 0.111
  F: 0.111
  G: 0.111
  H: 0.111

severity_allocation:
  S1: 0.34
  S2: 0.33
  S3: 0.33

mode_allocation:
  short: 0.85
  choice: 0.15

perspective_allocation:
  first: 0.80
  third: 0.20

style_directive_pool_size: 10
over_generation_buffer: 0.15

holdout:
  ratio: 0.20
  seed: 4242
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = load_allocation_config(temp_path)
            assert config.total_size == 1500
            assert config.family_allocation[FamilyID.A] == 0.222
            assert "C" not in FamilyID.__members__
            assert config.severity_allocation[Severity.S1] == 0.34
            assert config.mode_allocation[Mode.SHORT] == 0.85
            assert config.mode_allocation[Mode.CHOICE] == 0.15
            assert config.perspective_allocation[Perspective.FIRST] == 0.80
            assert config.style_directive_pool_size == 10
            assert config.over_generation_buffer == 0.15
            assert config.holdout_ratio == 0.20
            assert config.holdout_seed == 4242
        finally:
            os.unlink(temp_path)

    def test_load_nonexistent_file(self):
        """Loading a missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_allocation_config("/nonexistent/path/config.yaml")

    def test_load_default_config_file(self):
        """The shipped default.yaml loads and validates."""
        default_path = Path("dataset_gen/configs/default.yaml")
        if default_path.exists():
            config = load_allocation_config(str(default_path))
            assert config.total_size == 6000
            assert config.mode_allocation[Mode.SHORT] == 0.85
            assert "C" not in FamilyID.__members__
            # default.yaml preserves the prior severity split.
            assert config.severity_allocation[Severity.S1] == 0.34
