"""
Tests for the Variation Module (T5).

Tests cover:
- Determinism: same context always gets same variations
- Distribution: ~50% alt_phrasing=True across many samples
- Uniform distribution of lexical and formatting variants
- get_ordering() helper function correctness
- Batch processing
"""

import pytest
from collections import Counter

from dataset_gen.src.schema import (
    Context,
    PreferencePair,
    FamilyID,
    Severity,
    Mode,
    Perspective,
)
from dataset_gen.src.variation import (
    VariationApplicator,
    get_ordering,
    get_ordering_with_ids,
)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_pref_pair() -> PreferencePair:
    """Create a sample preference pair for testing."""
    return PreferencePair(
        pref_a_id="concise",
        pref_a_text="concise answers",
        pref_b_id="verbose",
        pref_b_text="verbose, detailed answers",
        domain="style",
    )


@pytest.fixture
def sample_context(sample_pref_pair: PreferencePair) -> Context:
    """Create a sample context for testing."""
    return Context(
        pair_id="pair_000001",
        seed=12345,
        family_id=FamilyID.A,
        subtype_id="A1",
        severity=Severity.S1,
        mode=Mode.RATING,
        perspective=Perspective.FIRST,
        pref_pair=sample_pref_pair,
        current_pref="a",
        target_pref="b",
    )


@pytest.fixture
def applicator() -> VariationApplicator:
    """Create a VariationApplicator with a fixed global seed."""
    return VariationApplicator(global_seed=42)


def create_context_with_seed(seed: int, pref_pair: PreferencePair) -> Context:
    """Helper to create contexts with different seeds."""
    return Context(
        pair_id=f"pair_{seed:06d}",
        seed=seed,
        family_id=FamilyID.A,
        subtype_id="A1",
        severity=Severity.S1,
        mode=Mode.RATING,
        perspective=Perspective.FIRST,
        pref_pair=pref_pair,
        current_pref="a",
        target_pref="b",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# DETERMINISM TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeterminism:
    """Tests for deterministic behavior of VariationApplicator."""

    def test_same_context_same_variations(
        self, applicator: VariationApplicator, sample_context: Context
    ):
        """Same context always produces the same variations."""
        result1 = applicator.apply(sample_context)
        result2 = applicator.apply(sample_context)

        assert result1.alt_phrasing == result2.alt_phrasing
        assert result1.lexical_variant == result2.lexical_variant
        assert result1.formatting_variant == result2.formatting_variant

    def test_determinism_across_new_applicator_instances(
        self, sample_context: Context
    ):
        """Different applicator instances with same seed produce same results."""
        app1 = VariationApplicator(global_seed=42)
        app2 = VariationApplicator(global_seed=42)

        result1 = app1.apply(sample_context)
        result2 = app2.apply(sample_context)

        assert result1.alt_phrasing == result2.alt_phrasing
        assert result1.lexical_variant == result2.lexical_variant
        assert result1.formatting_variant == result2.formatting_variant

    def test_different_global_seeds_produce_different_results(
        self, sample_context: Context
    ):
        """Different global seeds produce different variations."""
        app1 = VariationApplicator(global_seed=42)
        app2 = VariationApplicator(global_seed=999)

        result1 = app1.apply(sample_context)
        result2 = app2.apply(sample_context)

        # At least one variation should differ (statistically very likely)
        different = (
            result1.alt_phrasing != result2.alt_phrasing
            or result1.lexical_variant != result2.lexical_variant
            or result1.formatting_variant != result2.formatting_variant
        )
        assert different

    def test_different_context_seeds_produce_different_results(
        self, applicator: VariationApplicator, sample_pref_pair: PreferencePair
    ):
        """Different context seeds produce different variations."""
        ctx1 = create_context_with_seed(100, sample_pref_pair)
        ctx2 = create_context_with_seed(200, sample_pref_pair)

        result1 = applicator.apply(ctx1)
        result2 = applicator.apply(ctx2)

        # At least one variation should differ
        different = (
            result1.alt_phrasing != result2.alt_phrasing
            or result1.lexical_variant != result2.lexical_variant
            or result1.formatting_variant != result2.formatting_variant
        )
        assert different


# ═══════════════════════════════════════════════════════════════════════════════
# DISTRIBUTION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestDistribution:
    """Tests for statistical distribution of variations."""

    def test_alt_phrasing_approximately_50_percent(
        self, applicator: VariationApplicator, sample_pref_pair: PreferencePair
    ):
        """
        Verify ~50% of contexts have alt_phrasing=True.

        With 1000 samples, we expect ~500 True values.
        Using a tolerance of 10% (400-600 range) for statistical validity.
        """
        num_samples = 1000
        contexts = [
            create_context_with_seed(i, sample_pref_pair) for i in range(num_samples)
        ]

        varied = applicator.apply_batch(contexts)
        swap_count = sum(1 for ctx in varied if ctx.alt_phrasing)

        # 50% +/- 10% tolerance
        assert 400 <= swap_count <= 600, f"Expected ~500, got {swap_count}"

    def test_lexical_variant_distribution_is_uniform(
        self, applicator: VariationApplicator, sample_pref_pair: PreferencePair
    ):
        """Verify lexical variants are uniformly distributed."""
        num_samples = 1000
        contexts = [
            create_context_with_seed(i, sample_pref_pair) for i in range(num_samples)
        ]

        varied = applicator.apply_batch(contexts)
        variant_counts = Counter(ctx.lexical_variant for ctx in varied)

        # Each variant should appear roughly 1000/5 = 200 times
        # Using 20% tolerance (160-240 range)
        expected = num_samples / VariationApplicator.NUM_LEXICAL_VARIANTS
        for variant, count in variant_counts.items():
            assert 0 <= variant < VariationApplicator.NUM_LEXICAL_VARIANTS
            assert (
                expected * 0.6 <= count <= expected * 1.4
            ), f"Variant {variant}: expected ~{expected}, got {count}"

    def test_formatting_variant_distribution_is_uniform(
        self, applicator: VariationApplicator, sample_pref_pair: PreferencePair
    ):
        """Verify formatting variants are uniformly distributed."""
        num_samples = 1000
        contexts = [
            create_context_with_seed(i, sample_pref_pair) for i in range(num_samples)
        ]

        varied = applicator.apply_batch(contexts)
        variant_counts = Counter(ctx.formatting_variant for ctx in varied)

        # Each variant should appear roughly 1000/3 = 333 times
        # Using 20% tolerance
        expected = num_samples / VariationApplicator.NUM_FORMATTING_VARIANTS
        for variant, count in variant_counts.items():
            assert 0 <= variant < VariationApplicator.NUM_FORMATTING_VARIANTS
            assert (
                expected * 0.6 <= count <= expected * 1.4
            ), f"Variant {variant}: expected ~{expected}, got {count}"

    def test_all_lexical_variants_are_used(
        self, applicator: VariationApplicator, sample_pref_pair: PreferencePair
    ):
        """Verify all lexical variants are represented in a large sample."""
        num_samples = 500
        contexts = [
            create_context_with_seed(i, sample_pref_pair) for i in range(num_samples)
        ]

        varied = applicator.apply_batch(contexts)
        variants_used = set(ctx.lexical_variant for ctx in varied)

        expected_variants = set(range(VariationApplicator.NUM_LEXICAL_VARIANTS))
        assert variants_used == expected_variants

    def test_all_formatting_variants_are_used(
        self, applicator: VariationApplicator, sample_pref_pair: PreferencePair
    ):
        """Verify all formatting variants are represented in a large sample."""
        num_samples = 500
        contexts = [
            create_context_with_seed(i, sample_pref_pair) for i in range(num_samples)
        ]

        varied = applicator.apply_batch(contexts)
        variants_used = set(ctx.formatting_variant for ctx in varied)

        expected_variants = set(range(VariationApplicator.NUM_FORMATTING_VARIANTS))
        assert variants_used == expected_variants


# ═══════════════════════════════════════════════════════════════════════════════
# GET_ORDERING TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestGetOrdering:
    """Tests for the get_ordering helper function."""

    def test_no_swap_current_pref_a(self, sample_pref_pair: PreferencePair):
        """Without swap, current pref comes first (current=a)."""
        ctx = Context(
            pair_id="test",
            seed=1,
            family_id=FamilyID.A,
            subtype_id="A1",
            severity=Severity.S1,
            mode=Mode.RATING,
            perspective=Perspective.FIRST,
            pref_pair=sample_pref_pair,
            current_pref="a",
            target_pref="b",
            alt_phrasing=False,
        )

        first, second = get_ordering(ctx)

        assert first == "concise answers"
        assert second == "verbose, detailed answers"

    def test_no_swap_current_pref_b(self, sample_pref_pair: PreferencePair):
        """Without swap, current pref comes first (current=b)."""
        ctx = Context(
            pair_id="test",
            seed=1,
            family_id=FamilyID.A,
            subtype_id="A1",
            severity=Severity.S1,
            mode=Mode.RATING,
            perspective=Perspective.FIRST,
            pref_pair=sample_pref_pair,
            current_pref="b",
            target_pref="a",
            alt_phrasing=False,
        )

        first, second = get_ordering(ctx)

        assert first == "verbose, detailed answers"
        assert second == "concise answers"

    def test_with_alt_phrasing_current_pref_a(self, sample_pref_pair: PreferencePair):
        """alt_phrasing doesn't affect ordering - current pref still comes first.

        Note: alt_phrasing only affects lexical variant selection, not preference ordering.
        Ordering is always (current, target) regardless of alt_phrasing.
        """
        ctx = Context(
            pair_id="test",
            seed=1,
            family_id=FamilyID.A,
            subtype_id="A1",
            severity=Severity.S1,
            mode=Mode.RATING,
            perspective=Perspective.FIRST,
            pref_pair=sample_pref_pair,
            current_pref="a",
            target_pref="b",
            alt_phrasing=True,
        )

        first, second = get_ordering(ctx)

        # Ordering is always (current, target) regardless of alt_phrasing
        assert first == "concise answers"
        assert second == "verbose, detailed answers"

    def test_with_alt_phrasing_current_pref_b(self, sample_pref_pair: PreferencePair):
        """alt_phrasing doesn't affect ordering - current pref still comes first.

        Note: alt_phrasing only affects lexical variant selection, not preference ordering.
        """
        ctx = Context(
            pair_id="test",
            seed=1,
            family_id=FamilyID.A,
            subtype_id="A1",
            severity=Severity.S1,
            mode=Mode.RATING,
            perspective=Perspective.FIRST,
            pref_pair=sample_pref_pair,
            current_pref="b",
            target_pref="a",
            alt_phrasing=True,
        )

        first, second = get_ordering(ctx)

        # Ordering is always (current, target) regardless of alt_phrasing
        assert first == "verbose, detailed answers"
        assert second == "concise answers"


class TestGetOrderingWithIds:
    """Tests for the get_ordering_with_ids helper function."""

    def test_returns_ids_and_texts_no_swap(self, sample_pref_pair: PreferencePair):
        """Returns both IDs and texts without swap."""
        ctx = Context(
            pair_id="test",
            seed=1,
            family_id=FamilyID.A,
            subtype_id="A1",
            severity=Severity.S1,
            mode=Mode.RATING,
            perspective=Perspective.FIRST,
            pref_pair=sample_pref_pair,
            current_pref="a",
            target_pref="b",
            alt_phrasing=False,
        )

        (id1, text1), (id2, text2) = get_ordering_with_ids(ctx)

        assert id1 == "concise"
        assert text1 == "concise answers"
        assert id2 == "verbose"
        assert text2 == "verbose, detailed answers"

    def test_returns_ids_and_texts_with_alt_phrasing(self, sample_pref_pair: PreferencePair):
        """alt_phrasing doesn't affect ordering - returns (current, target).

        Note: alt_phrasing only affects lexical variant selection, not preference ordering.
        """
        ctx = Context(
            pair_id="test",
            seed=1,
            family_id=FamilyID.A,
            subtype_id="A1",
            severity=Severity.S1,
            mode=Mode.RATING,
            perspective=Perspective.FIRST,
            pref_pair=sample_pref_pair,
            current_pref="a",
            target_pref="b",
            alt_phrasing=True,
        )

        (id1, text1), (id2, text2) = get_ordering_with_ids(ctx)

        # Ordering is always (current, target) regardless of alt_phrasing
        assert id1 == "concise"
        assert text1 == "concise answers"
        assert id2 == "verbose"
        assert text2 == "verbose, detailed answers"


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH PROCESSING TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestBatchProcessing:
    """Tests for batch processing functionality."""

    def test_apply_batch_returns_correct_length(
        self, applicator: VariationApplicator, sample_pref_pair: PreferencePair
    ):
        """apply_batch returns same number of contexts as input."""
        contexts = [create_context_with_seed(i, sample_pref_pair) for i in range(10)]

        result = applicator.apply_batch(contexts)

        assert len(result) == len(contexts)

    def test_apply_batch_preserves_order(
        self, applicator: VariationApplicator, sample_pref_pair: PreferencePair
    ):
        """apply_batch preserves the order of contexts."""
        contexts = [create_context_with_seed(i, sample_pref_pair) for i in range(10)]

        result = applicator.apply_batch(contexts)

        for orig, varied in zip(contexts, result):
            assert orig.pair_id == varied.pair_id
            assert orig.seed == varied.seed

    def test_apply_batch_empty_list(self, applicator: VariationApplicator):
        """apply_batch handles empty list gracefully."""
        result = applicator.apply_batch([])

        assert result == []

    def test_apply_batch_single_context(
        self, applicator: VariationApplicator, sample_context: Context
    ):
        """apply_batch works with single context."""
        result = applicator.apply_batch([sample_context])

        assert len(result) == 1
        assert result[0].pair_id == sample_context.pair_id


# ═══════════════════════════════════════════════════════════════════════════════
# DISABLED VARIATIONS TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestDisabledVariations:
    """Tests for apply_with_disabled_variations method."""

    def test_all_variations_disabled(
        self, applicator: VariationApplicator, sample_context: Context
    ):
        """Disabled variations sets all flags to defaults."""
        result = applicator.apply_with_disabled_variations(sample_context)

        assert result.alt_phrasing is False
        assert result.lexical_variant == 0
        assert result.formatting_variant == 0

    def test_preserves_other_context_fields(
        self, applicator: VariationApplicator, sample_context: Context
    ):
        """Disabled variations preserves all other context fields."""
        result = applicator.apply_with_disabled_variations(sample_context)

        assert result.pair_id == sample_context.pair_id
        assert result.seed == sample_context.seed
        assert result.family_id == sample_context.family_id
        assert result.subtype_id == sample_context.subtype_id
        assert result.severity == sample_context.severity
        assert result.mode == sample_context.mode
        assert result.perspective == sample_context.perspective
        assert result.pref_pair == sample_context.pref_pair
        assert result.current_pref == sample_context.current_pref
        assert result.target_pref == sample_context.target_pref


# ═══════════════════════════════════════════════════════════════════════════════
# ORIGINAL CONTEXT IMMUTABILITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestImmutability:
    """Tests to ensure original context is not modified."""

    def test_apply_does_not_modify_original(
        self, applicator: VariationApplicator, sample_context: Context
    ):
        """apply() does not modify the original context."""
        original_swap = sample_context.alt_phrasing
        original_lexical = sample_context.lexical_variant
        original_formatting = sample_context.formatting_variant

        applicator.apply(sample_context)

        assert sample_context.alt_phrasing == original_swap
        assert sample_context.lexical_variant == original_lexical
        assert sample_context.formatting_variant == original_formatting

    def test_apply_returns_new_context(
        self, applicator: VariationApplicator, sample_context: Context
    ):
        """apply() returns a new context object."""
        result = applicator.apply(sample_context)

        # Should be a different object
        assert result is not sample_context


# ═══════════════════════════════════════════════════════════════════════════════
# VARIANT RANGE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestVariantRanges:
    """Tests to verify variant indices are within expected ranges."""

    def test_lexical_variant_in_valid_range(
        self, applicator: VariationApplicator, sample_pref_pair: PreferencePair
    ):
        """lexical_variant is always within valid range."""
        contexts = [
            create_context_with_seed(i, sample_pref_pair) for i in range(100)
        ]

        for ctx in applicator.apply_batch(contexts):
            assert 0 <= ctx.lexical_variant < VariationApplicator.NUM_LEXICAL_VARIANTS

    def test_formatting_variant_in_valid_range(
        self, applicator: VariationApplicator, sample_pref_pair: PreferencePair
    ):
        """formatting_variant is always within valid range."""
        contexts = [
            create_context_with_seed(i, sample_pref_pair) for i in range(100)
        ]

        for ctx in applicator.apply_batch(contexts):
            assert (
                0 <= ctx.formatting_variant < VariationApplicator.NUM_FORMATTING_VARIANTS
            )
