"""
Tests for the Variation Module (T5).

Tests cover:
- Determinism: same context always gets same variations
- Uniform distribution of lexical_variant (single 0-9 index)
- Range/bounds enforcement (via Context.__post_init__)
- That lexical variation actually varies across 0-9
- get_ordering() / get_ordering_with_ids() helper correctness
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
from dataset_gen.src.catalogs import get_lexical_variant


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
        domain_category="lifestyle",
        severity=Severity.S1,
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
        mode=Mode.CHOICE,
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
        mode=Mode.CHOICE,
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
        """Same context always produces the same lexical_variant."""
        result1 = applicator.apply(sample_context)
        result2 = applicator.apply(sample_context)

        assert result1.lexical_variant == result2.lexical_variant

    def test_determinism_across_new_applicator_instances(
        self, sample_context: Context
    ):
        """Different applicator instances with same seed produce same results."""
        app1 = VariationApplicator(global_seed=42)
        app2 = VariationApplicator(global_seed=42)

        result1 = app1.apply(sample_context)
        result2 = app2.apply(sample_context)

        assert result1.lexical_variant == result2.lexical_variant

    def test_different_global_seeds_produce_different_results(
        self, sample_pref_pair: PreferencePair
    ):
        """Different global seeds produce different variations (statistically)."""
        contexts = [
            create_context_with_seed(i, sample_pref_pair) for i in range(50)
        ]
        app1 = VariationApplicator(global_seed=42)
        app2 = VariationApplicator(global_seed=999)

        results1 = [c.lexical_variant for c in app1.apply_batch(contexts)]
        results2 = [c.lexical_variant for c in app2.apply_batch(contexts)]

        # Across 50 contexts at least one lexical_variant should differ.
        assert results1 != results2

    def test_different_context_seeds_produce_different_results(
        self, applicator: VariationApplicator, sample_pref_pair: PreferencePair
    ):
        """Different context seeds produce different variations (statistically)."""
        contexts = [
            create_context_with_seed(i, sample_pref_pair) for i in range(50)
        ]
        variants = [c.lexical_variant for c in applicator.apply_batch(contexts)]

        # With 50 distinct seeds we should observe more than one variant value.
        assert len(set(variants)) > 1


# ═══════════════════════════════════════════════════════════════════════════════
# DISTRIBUTION / RANGE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestDistribution:
    """Tests for statistical distribution and range of lexical_variant."""

    def test_lexical_variant_distribution_is_uniform(
        self, applicator: VariationApplicator, sample_pref_pair: PreferencePair
    ):
        """Verify lexical variants are roughly uniformly distributed over 0-9."""
        num_samples = 2000
        contexts = [
            create_context_with_seed(i, sample_pref_pair) for i in range(num_samples)
        ]

        varied = applicator.apply_batch(contexts)
        variant_counts = Counter(ctx.lexical_variant for ctx in varied)

        # Each of the 10 variants should appear roughly 2000/10 = 200 times.
        expected = num_samples / VariationApplicator.NUM_LEXICAL_VARIANTS
        for variant, count in variant_counts.items():
            assert 0 <= variant < VariationApplicator.NUM_LEXICAL_VARIANTS
            assert (
                expected * 0.6 <= count <= expected * 1.4
            ), f"Variant {variant}: expected ~{expected}, got {count}"

    def test_all_lexical_variants_are_used(
        self, applicator: VariationApplicator, sample_pref_pair: PreferencePair
    ):
        """Verify all 10 lexical variants are represented in a large sample."""
        num_samples = 500
        contexts = [
            create_context_with_seed(i, sample_pref_pair) for i in range(num_samples)
        ]

        varied = applicator.apply_batch(contexts)
        variants_used = set(ctx.lexical_variant for ctx in varied)

        expected_variants = set(range(VariationApplicator.NUM_LEXICAL_VARIANTS))
        assert variants_used == expected_variants

    def test_lexical_variant_in_valid_range(
        self, applicator: VariationApplicator, sample_pref_pair: PreferencePair
    ):
        """lexical_variant is always within the valid 0-9 range."""
        contexts = [
            create_context_with_seed(i, sample_pref_pair) for i in range(100)
        ]

        for ctx in applicator.apply_batch(contexts):
            assert 0 <= ctx.lexical_variant < VariationApplicator.NUM_LEXICAL_VARIANTS


# ═══════════════════════════════════════════════════════════════════════════════
# BOUNDS ENFORCEMENT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestBounds:
    """Bounds are enforced by Context.__post_init__ (0-9)."""

    def test_lexical_variant_too_high_raises(self, sample_pref_pair: PreferencePair):
        """Constructing a Context with lexical_variant=10 raises."""
        with pytest.raises(ValueError):
            Context(
                pair_id="bad",
                seed=1,
                family_id=FamilyID.A,
                subtype_id="A1",
                severity=Severity.S1,
                mode=Mode.CHOICE,
                perspective=Perspective.FIRST,
                pref_pair=sample_pref_pair,
                current_pref="a",
                target_pref="b",
                lexical_variant=10,
            )

    def test_lexical_variant_negative_raises(self, sample_pref_pair: PreferencePair):
        """Constructing a Context with a negative lexical_variant raises."""
        with pytest.raises(ValueError):
            Context(
                pair_id="bad",
                seed=1,
                family_id=FamilyID.A,
                subtype_id="A1",
                severity=Severity.S1,
                mode=Mode.CHOICE,
                perspective=Perspective.FIRST,
                pref_pair=sample_pref_pair,
                current_pref="a",
                target_pref="b",
                lexical_variant=-1,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# VARIATION-ACTUALLY-VARIES TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestVariationVaries:
    """Verify lexical_variant actually selects different surface forms."""

    def test_variant_0_and_9_select_different_synonyms(self):
        """lexical_variant 0 vs 9 selects a different synonym for a term."""
        term = "acceptable"
        assert get_lexical_variant(term, 0) != get_lexical_variant(term, 9)

    def test_at_least_three_distinct_outputs_across_0_to_9(self):
        """Across lexical_variant 0-9 we observe >= 3 distinct outputs.

        LEXICAL_VARIANTS may not give 10 unique synonyms per term, but a
        well-populated term should give at least 3.
        """
        term = "acceptable"
        outputs = {get_lexical_variant(term, idx) for idx in range(10)}
        assert len(outputs) >= 3

    def test_variant_selection_is_deterministic(self):
        """Same (term, idx) yields identical output across calls."""
        term = "change"
        first = get_lexical_variant(term, 4)
        second = get_lexical_variant(term, 4)
        assert first == second


# ═══════════════════════════════════════════════════════════════════════════════
# REMOVED-FIELD TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestRemovedFields:
    """Ensure removed variation fields are gone everywhere."""

    def test_no_formatting_variant_on_context(self, sample_context: Context):
        """Context no longer carries formatting_variant."""
        assert not hasattr(sample_context, "formatting_variant")

    def test_no_alt_phrasing_on_context(self, sample_context: Context):
        """Context no longer carries alt_phrasing."""
        assert not hasattr(sample_context, "alt_phrasing")

    def test_context_rejects_formatting_variant_kwarg(
        self, sample_pref_pair: PreferencePair
    ):
        """Context construction rejects a formatting_variant keyword."""
        with pytest.raises(TypeError):
            Context(
                pair_id="x",
                seed=1,
                family_id=FamilyID.A,
                subtype_id="A1",
                severity=Severity.S1,
                mode=Mode.CHOICE,
                perspective=Perspective.FIRST,
                pref_pair=sample_pref_pair,
                current_pref="a",
                target_pref="b",
                formatting_variant=1,
            )

    def test_context_rejects_alt_phrasing_kwarg(
        self, sample_pref_pair: PreferencePair
    ):
        """Context construction rejects an alt_phrasing keyword."""
        with pytest.raises(TypeError):
            Context(
                pair_id="x",
                seed=1,
                family_id=FamilyID.A,
                subtype_id="A1",
                severity=Severity.S1,
                mode=Mode.CHOICE,
                perspective=Perspective.FIRST,
                pref_pair=sample_pref_pair,
                current_pref="a",
                target_pref="b",
                alt_phrasing=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# GET_ORDERING TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestGetOrdering:
    """Tests for the get_ordering helper function."""

    def test_current_pref_a(self, sample_pref_pair: PreferencePair):
        """Current pref comes first (current=a)."""
        ctx = create_context_with_seed(1, sample_pref_pair)

        first, second = get_ordering(ctx)

        assert first == "concise answers"
        assert second == "verbose, detailed answers"

    def test_current_pref_b(self, sample_pref_pair: PreferencePair):
        """Current pref comes first (current=b)."""
        ctx = Context(
            pair_id="test",
            seed=1,
            family_id=FamilyID.A,
            subtype_id="A1",
            severity=Severity.S1,
            mode=Mode.CHOICE,
            perspective=Perspective.FIRST,
            pref_pair=sample_pref_pair,
            current_pref="b",
            target_pref="a",
        )

        first, second = get_ordering(ctx)

        assert first == "verbose, detailed answers"
        assert second == "concise answers"


class TestGetOrderingWithIds:
    """Tests for the get_ordering_with_ids helper function."""

    def test_returns_ids_and_texts(self, sample_pref_pair: PreferencePair):
        """Returns both IDs and texts in (current, target) order."""
        ctx = create_context_with_seed(1, sample_pref_pair)

        (id1, text1), (id2, text2) = get_ordering_with_ids(ctx)

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
        """Disabled variations sets lexical_variant to its default."""
        result = applicator.apply_with_disabled_variations(sample_context)

        assert result.lexical_variant == 0

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
        original_lexical = sample_context.lexical_variant

        applicator.apply(sample_context)

        assert sample_context.lexical_variant == original_lexical

    def test_apply_returns_new_context(
        self, applicator: VariationApplicator, sample_context: Context
    ):
        """apply() returns a new context object."""
        result = applicator.apply(sample_context)

        # Should be a different object
        assert result is not sample_context
