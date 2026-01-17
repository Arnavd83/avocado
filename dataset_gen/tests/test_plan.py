"""
Tests for the Planning Module.

Task ID: T3
"""

import pytest
import tempfile
import os
from pathlib import Path

from dataset_gen.src.plan import (
    AllocationConfig,
    PlanGenerator,
    largest_remainder_allocation,
    load_allocation_config,
    validate_plan,
    FAMILY_SUBTYPES,
)
from dataset_gen.src.schema import FamilyID, Severity, Mode, Perspective


# ═══════════════════════════════════════════════════════════════════════════════
# ALLOCATION CONFIG TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestAllocationConfig:
    """Tests for AllocationConfig dataclass."""

    def test_default_values(self):
        """Test that defaults are set correctly from spec."""
        config = AllocationConfig()

        assert config.total_size == 6000

        # Check family allocation
        assert config.family_allocation[FamilyID.A] == 0.20
        assert config.family_allocation[FamilyID.B] == 0.15
        assert config.family_allocation[FamilyID.C] == 0.10
        assert config.family_allocation[FamilyID.D] == 0.15
        assert config.family_allocation[FamilyID.E] == 0.10
        assert config.family_allocation[FamilyID.F] == 0.10
        assert config.family_allocation[FamilyID.G] == 0.10
        assert config.family_allocation[FamilyID.H] == 0.10

        # Check severity allocation
        assert config.severity_allocation[Severity.S1] == 0.34
        assert config.severity_allocation[Severity.S2] == 0.33
        assert config.severity_allocation[Severity.S3] == 0.33

        # Check perspective allocation
        assert config.perspective_allocation[Perspective.FIRST] == 0.65
        assert config.perspective_allocation[Perspective.THIRD] == 0.35

    def test_family_allocation_sums_to_one(self):
        """Test that default family allocation sums to 1.0."""
        config = AllocationConfig()
        total = sum(config.family_allocation.values())
        assert abs(total - 1.0) < 0.001

    def test_severity_allocation_sums_to_one(self):
        """Test that default severity allocation sums to 1.0."""
        config = AllocationConfig()
        total = sum(config.severity_allocation.values())
        assert abs(total - 1.0) < 0.001

    def test_mode_allocation_per_family(self):
        """Test that mode allocation is defined per family."""
        config = AllocationConfig()

        # Check Family A has rating-heavy allocation
        assert config.mode_allocation[FamilyID.A][Mode.RATING] == 0.90
        assert config.mode_allocation[FamilyID.A][Mode.CHOICE] == 0.05
        assert config.mode_allocation[FamilyID.A][Mode.SHORT] == 0.05

        # Check Family B has choice-heavy allocation
        assert config.mode_allocation[FamilyID.B][Mode.RATING] == 0.40
        assert config.mode_allocation[FamilyID.B][Mode.CHOICE] == 0.60
        assert config.mode_allocation[FamilyID.B][Mode.SHORT] == 0.00

    def test_validate_valid_config(self):
        """Test that default config passes validation."""
        config = AllocationConfig()
        errors = config.validate()
        assert len(errors) == 0

    def test_validate_invalid_family_allocation(self):
        """Test that invalid family allocation is caught."""
        config = AllocationConfig()
        config.family_allocation[FamilyID.A] = 0.50  # Now sums to > 1.0
        errors = config.validate()
        assert any("Family allocation" in e for e in errors)

    def test_custom_total_size(self):
        """Test that custom total size is respected."""
        config = AllocationConfig(total_size=1000)
        assert config.total_size == 1000


# ═══════════════════════════════════════════════════════════════════════════════
# LARGEST REMAINDER METHOD TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestLargestRemainderAllocation:
    """Tests for the largest-remainder allocation method."""

    def test_basic_allocation(self):
        """Test basic allocation with simple proportions."""
        proportions = {"A": 0.5, "B": 0.5}
        counts = largest_remainder_allocation(100, proportions)

        assert counts["A"] == 50
        assert counts["B"] == 50
        assert sum(counts.values()) == 100

    def test_exact_sum(self):
        """Test that allocation always sums to total."""
        proportions = {"A": 0.33, "B": 0.33, "C": 0.34}
        counts = largest_remainder_allocation(100, proportions)
        assert sum(counts.values()) == 100

    def test_uneven_proportions(self):
        """Test allocation with uneven proportions."""
        proportions = {"A": 0.20, "B": 0.15, "C": 0.65}
        counts = largest_remainder_allocation(100, proportions)

        assert sum(counts.values()) == 100
        assert counts["A"] == 20
        assert counts["B"] == 15
        assert counts["C"] == 65

    def test_rounding_with_remainders(self):
        """Test that largest remainder method handles fractional allocations."""
        proportions = {"A": 0.333, "B": 0.333, "C": 0.334}
        counts = largest_remainder_allocation(10, proportions)

        # Should sum to 10 exactly
        assert sum(counts.values()) == 10
        # Each should be either 3 or 4
        assert all(3 <= c <= 4 for c in counts.values())

    def test_determinism(self):
        """Test that allocation is deterministic."""
        proportions = {"A": 0.33, "B": 0.33, "C": 0.34}

        counts1 = largest_remainder_allocation(100, proportions)
        counts2 = largest_remainder_allocation(100, proportions)

        assert counts1 == counts2

    def test_family_allocation_counts(self):
        """Test allocation with actual family proportions."""
        proportions = {
            FamilyID.A: 0.20,
            FamilyID.B: 0.15,
            FamilyID.C: 0.10,
            FamilyID.D: 0.15,
            FamilyID.E: 0.10,
            FamilyID.F: 0.10,
            FamilyID.G: 0.10,
            FamilyID.H: 0.10,
        }
        counts = largest_remainder_allocation(6000, proportions)

        assert counts[FamilyID.A] == 1200
        assert counts[FamilyID.B] == 900
        assert counts[FamilyID.C] == 600
        assert counts[FamilyID.D] == 900
        assert counts[FamilyID.E] == 600
        assert counts[FamilyID.F] == 600
        assert counts[FamilyID.G] == 600
        assert counts[FamilyID.H] == 600
        assert sum(counts.values()) == 6000


# ═══════════════════════════════════════════════════════════════════════════════
# PLAN GENERATOR TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestPlanGenerator:
    """Tests for PlanGenerator class."""

    def test_generate_creates_correct_count(self):
        """Test that generate produces the correct number of rows."""
        config = AllocationConfig(total_size=100)
        generator = PlanGenerator(config, global_seed=42)
        plan = generator.generate()

        assert len(plan) == 100

    def test_generate_unique_pair_ids(self):
        """Test that all pair_ids are unique."""
        config = AllocationConfig(total_size=1000)
        generator = PlanGenerator(config, global_seed=42)
        plan = generator.generate()

        pair_ids = [row.pair_id for row in plan]
        assert len(pair_ids) == len(set(pair_ids))

    def test_generate_deterministic(self):
        """Test that generation is deterministic with same seed."""
        config = AllocationConfig(total_size=100)

        generator1 = PlanGenerator(config, global_seed=42)
        plan1 = generator1.generate()

        generator2 = PlanGenerator(config, global_seed=42)
        plan2 = generator2.generate()

        assert len(plan1) == len(plan2)
        for row1, row2 in zip(plan1, plan2):
            assert row1.pair_id == row2.pair_id
            assert row1.seed == row2.seed
            assert row1.family_id == row2.family_id
            assert row1.subtype_id == row2.subtype_id
            assert row1.severity == row2.severity
            assert row1.mode == row2.mode
            assert row1.perspective == row2.perspective

    def test_generate_different_seeds_different_plans(self):
        """Test that different seeds produce different plans."""
        config = AllocationConfig(total_size=100)

        generator1 = PlanGenerator(config, global_seed=42)
        plan1 = generator1.generate()

        generator2 = PlanGenerator(config, global_seed=123)
        plan2 = generator2.generate()

        # Seeds should be different
        seeds1 = [row.seed for row in plan1]
        seeds2 = [row.seed for row in plan2]
        assert seeds1 != seeds2

    def test_generate_family_distribution(self):
        """Test that family distribution matches allocation."""
        config = AllocationConfig(total_size=6000)
        generator = PlanGenerator(config, global_seed=42)
        plan = generator.generate()

        family_counts = {}
        for row in plan:
            family_counts[row.family_id] = family_counts.get(row.family_id, 0) + 1

        assert family_counts[FamilyID.A] == 1200
        assert family_counts[FamilyID.B] == 900
        assert family_counts[FamilyID.C] == 600
        assert family_counts[FamilyID.D] == 900
        assert family_counts[FamilyID.E] == 600
        assert family_counts[FamilyID.F] == 600
        assert family_counts[FamilyID.G] == 600
        assert family_counts[FamilyID.H] == 600

    def test_generate_severity_distribution_per_family(self):
        """Test that severity distribution matches allocation within each family."""
        config = AllocationConfig(total_size=6000)
        generator = PlanGenerator(config, global_seed=42)
        plan = generator.generate()

        # Check severity distribution within Family A (1200 total)
        family_a_rows = [r for r in plan if r.family_id == FamilyID.A]
        severity_counts = {}
        for row in family_a_rows:
            severity_counts[row.severity] = severity_counts.get(row.severity, 0) + 1

        # Expected: S1=34%, S2=33%, S3=33% of 1200
        expected_s1 = largest_remainder_allocation(
            1200, config.severity_allocation
        )[Severity.S1]
        expected_s2 = largest_remainder_allocation(
            1200, config.severity_allocation
        )[Severity.S2]
        expected_s3 = largest_remainder_allocation(
            1200, config.severity_allocation
        )[Severity.S3]

        assert severity_counts[Severity.S1] == expected_s1
        assert severity_counts[Severity.S2] == expected_s2
        assert severity_counts[Severity.S3] == expected_s3

    def test_generate_mode_distribution_per_family(self):
        """Test that mode distribution matches allocation within each family."""
        config = AllocationConfig(total_size=6000)
        generator = PlanGenerator(config, global_seed=42)
        plan = generator.generate()

        # Check mode distribution within Family A (rating-heavy: 90/5/5)
        family_a_rows = [r for r in plan if r.family_id == FamilyID.A]
        mode_counts = {}
        for row in family_a_rows:
            mode_counts[row.mode] = mode_counts.get(row.mode, 0) + 1

        expected_counts = largest_remainder_allocation(
            1200, config.mode_allocation[FamilyID.A]
        )

        assert mode_counts[Mode.RATING] == expected_counts[Mode.RATING]
        assert mode_counts[Mode.CHOICE] == expected_counts[Mode.CHOICE]
        assert mode_counts[Mode.SHORT] == expected_counts[Mode.SHORT]

    def test_generate_valid_subtypes(self):
        """Test that all subtypes are valid for their family."""
        config = AllocationConfig(total_size=1000)
        generator = PlanGenerator(config, global_seed=42)
        plan = generator.generate()

        for row in plan:
            valid_subtypes = FAMILY_SUBTYPES[row.family_id]
            assert row.subtype_id in valid_subtypes, (
                f"Invalid subtype {row.subtype_id} for family {row.family_id.name}"
            )

    def test_generate_pair_id_format(self):
        """Test that pair_ids have correct format."""
        config = AllocationConfig(total_size=100)
        generator = PlanGenerator(config, global_seed=42)
        plan = generator.generate()

        for i, row in enumerate(plan):
            expected_id = f"pair_{i:06d}"
            # Note: pair_ids may not be in sequential order due to shuffling within families
            assert row.pair_id.startswith("pair_")
            assert len(row.pair_id) == len("pair_000000")

    def test_generate_seeds_are_positive(self):
        """Test that all seeds are positive integers."""
        config = AllocationConfig(total_size=100)
        generator = PlanGenerator(config, global_seed=42)
        plan = generator.generate()

        for row in plan:
            assert isinstance(row.seed, int)
            assert row.seed >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# YAML LOADING TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestLoadAllocationConfig:
    """Tests for load_allocation_config function."""

    def test_load_from_file(self):
        """Test loading config from YAML file."""
        yaml_content = """
generation:
  total_size: 1000

family_allocation:
  A: 0.25
  B: 0.25
  C: 0.25
  D: 0.25
  E: 0.00
  F: 0.00
  G: 0.00
  H: 0.00

severity_allocation:
  S1: 0.33
  S2: 0.33
  S3: 0.34

mode_allocation:
  per_family:
    A:
      rating: 0.80
      choice: 0.10
      short: 0.10
    B:
      rating: 0.50
      choice: 0.50
      short: 0.00
    C:
      rating: 0.60
      choice: 0.20
      short: 0.20
    D:
      rating: 0.40
      choice: 0.40
      short: 0.20

perspective_allocation:
  first: 0.60
  third: 0.40
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = load_allocation_config(temp_path)

            assert config.total_size == 1000
            assert config.family_allocation[FamilyID.A] == 0.25
            assert config.severity_allocation[Severity.S3] == 0.34
            assert config.mode_allocation[FamilyID.A][Mode.RATING] == 0.80
            assert config.perspective_allocation[Perspective.FIRST] == 0.60
        finally:
            os.unlink(temp_path)

    def test_load_nonexistent_file(self):
        """Test that loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_allocation_config("/nonexistent/path/config.yaml")

    def test_load_default_config_file(self):
        """Test loading the default config file if it exists."""
        default_path = Path("dataset_gen/configs/default.yaml")
        if default_path.exists():
            config = load_allocation_config(str(default_path))
            assert config.total_size == 6000


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidatePlan:
    """Tests for validate_plan function."""

    def test_valid_plan_passes(self):
        """Test that a valid plan passes validation."""
        config = AllocationConfig(total_size=100)
        generator = PlanGenerator(config, global_seed=42)
        plan = generator.generate()

        errors = validate_plan(plan, config)
        assert len(errors) == 0

    def test_validates_total_count(self):
        """Test that validation catches wrong total count."""
        config = AllocationConfig(total_size=100)
        generator = PlanGenerator(config, global_seed=42)
        plan = generator.generate()

        # Remove some rows
        plan = plan[:50]

        errors = validate_plan(plan, config)
        assert any("50 rows" in e for e in errors)

    def test_validates_duplicate_pair_ids(self):
        """Test that validation catches duplicate pair_ids."""
        config = AllocationConfig(total_size=100)
        generator = PlanGenerator(config, global_seed=42)
        plan = generator.generate()

        # Create a duplicate by modifying one row (hacky but tests the validation)
        from dataset_gen.src.schema import PlanRow
        plan = list(plan)
        plan[0] = PlanRow(
            pair_id=plan[1].pair_id,  # Duplicate!
            seed=plan[0].seed,
            family_id=plan[0].family_id,
            subtype_id=plan[0].subtype_id,
            severity=plan[0].severity,
            mode=plan[0].mode,
            perspective=plan[0].perspective,
        )

        errors = validate_plan(plan, config)
        assert any("Duplicate" in e for e in errors)

    def test_full_size_validation(self):
        """Test validation with full 6000 size."""
        config = AllocationConfig(total_size=6000)
        generator = PlanGenerator(config, global_seed=42)
        plan = generator.generate()

        errors = validate_plan(plan, config)
        assert len(errors) == 0, f"Validation errors: {errors}"


# ═══════════════════════════════════════════════════════════════════════════════
# EDGE CASE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_small_allocation(self):
        """Test with very small total size."""
        config = AllocationConfig(total_size=10)
        generator = PlanGenerator(config, global_seed=42)
        plan = generator.generate()

        assert len(plan) == 10
        errors = validate_plan(plan, config)
        # May have errors due to rounding with small numbers
        # but should still produce correct total

    def test_large_allocation(self):
        """Test with larger total size."""
        config = AllocationConfig(total_size=10000)
        generator = PlanGenerator(config, global_seed=42)
        plan = generator.generate()

        assert len(plan) == 10000

        # Check uniqueness
        pair_ids = [row.pair_id for row in plan]
        assert len(pair_ids) == len(set(pair_ids))

    def test_different_global_seeds(self):
        """Test that different global seeds produce different but valid plans."""
        config = AllocationConfig(total_size=100)

        seeds_to_test = [0, 1, 42, 12345, 999999]
        plans = []

        for seed in seeds_to_test:
            generator = PlanGenerator(config, global_seed=seed)
            plan = generator.generate()
            plans.append(plan)

            # Each should be valid
            errors = validate_plan(plan, config)
            assert len(errors) == 0

        # Plans with different seeds should have different seeds for rows
        for i in range(len(plans) - 1):
            row_seeds_1 = sorted([r.seed for r in plans[i]])
            row_seeds_2 = sorted([r.seed for r in plans[i + 1]])
            assert row_seeds_1 != row_seeds_2

    def test_family_subtypes_defined(self):
        """Test that all families have subtypes defined."""
        for family_id in FamilyID:
            assert family_id in FAMILY_SUBTYPES
            assert len(FAMILY_SUBTYPES[family_id]) == 3  # Each family has 3 subtypes
