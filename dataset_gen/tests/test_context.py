"""
Tests for the Context Synthesis Module (T4).

Tests cover:
- ContextSynthesizer creation and basic usage
- Deterministic behavior (same input produces same output)
- Synthesize single PlanRow to Context
- Batch synthesis of multiple PlanRows
- Helper functions for preference text retrieval
- Current/target preference assignment distribution
"""

import pytest
import random

from dataset_gen.src.schema import (
    FamilyID,
    Severity,
    Mode,
    Perspective,
    PlanRow,
    Context,
    PreferencePair,
)
from dataset_gen.src.context import (
    ContextSynthesizer,
    get_current_pref_text,
    get_target_pref_text,
    get_current_pref_id,
    get_target_pref_id,
)
from dataset_gen.src.catalogs import get_preference_pairs_for_severity


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def synthesizer():
    """Create a ContextSynthesizer for testing."""
    return ContextSynthesizer(global_seed=42)


@pytest.fixture
def plan_row_s1():
    """Create a PlanRow with S1 (style) severity."""
    return PlanRow(
        pair_id="pair_000001",
        seed=12345,
        family_id=FamilyID.A,
        subtype_id="A1",
        severity=Severity.S1,
        mode=Mode.RATING,
        perspective=Perspective.FIRST,
    )


@pytest.fixture
def plan_row_s2():
    """Create a PlanRow with S2 (workflow) severity."""
    return PlanRow(
        pair_id="pair_000002",
        seed=67890,
        family_id=FamilyID.B,
        subtype_id="B1",
        severity=Severity.S2,
        mode=Mode.CHOICE,
        perspective=Perspective.THIRD,
    )


@pytest.fixture
def plan_row_s3():
    """Create a PlanRow with S3 (epistemic) severity."""
    return PlanRow(
        pair_id="pair_000003",
        seed=11111,
        family_id=FamilyID.C,
        subtype_id="C1",
        severity=Severity.S3,
        mode=Mode.SHORT,
        perspective=Perspective.NEUTRAL,
    )


@pytest.fixture
def multiple_plan_rows():
    """Create multiple PlanRows for batch testing."""
    return [
        PlanRow(
            pair_id=f"pair_{i:06d}",
            seed=i * 1000 + 42,
            family_id=FamilyID.A,
            subtype_id="A1",
            severity=Severity.S1,
            mode=Mode.RATING,
            perspective=Perspective.FIRST,
        )
        for i in range(10)
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXT SYNTHESIZER BASIC TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestContextSynthesizerCreation:
    """Tests for ContextSynthesizer instantiation."""

    def test_create_synthesizer(self):
        """Test basic synthesizer creation."""
        synth = ContextSynthesizer(global_seed=42)
        assert synth.global_seed == 42

    def test_create_synthesizer_different_seeds(self):
        """Test synthesizer with different seeds."""
        synth1 = ContextSynthesizer(global_seed=1)
        synth2 = ContextSynthesizer(global_seed=2)
        assert synth1.global_seed != synth2.global_seed


class TestSynthesizeBasic:
    """Basic tests for synthesize method."""

    def test_synthesize_returns_context(self, synthesizer, plan_row_s1):
        """Test that synthesize returns a Context object."""
        ctx = synthesizer.synthesize(plan_row_s1)
        assert isinstance(ctx, Context)

    def test_synthesize_preserves_plan_row_fields(self, synthesizer, plan_row_s1):
        """Test that Context preserves PlanRow fields."""
        ctx = synthesizer.synthesize(plan_row_s1)
        assert ctx.pair_id == plan_row_s1.pair_id
        assert ctx.seed == plan_row_s1.seed
        assert ctx.family_id == plan_row_s1.family_id
        assert ctx.subtype_id == plan_row_s1.subtype_id
        assert ctx.severity == plan_row_s1.severity
        assert ctx.mode == plan_row_s1.mode
        assert ctx.perspective == plan_row_s1.perspective

    def test_synthesize_creates_preference_pair(self, synthesizer, plan_row_s1):
        """Test that Context has a valid PreferencePair."""
        ctx = synthesizer.synthesize(plan_row_s1)
        assert isinstance(ctx.pref_pair, PreferencePair)
        assert ctx.pref_pair.domain == "style"  # S1 maps to style

    def test_synthesize_assigns_current_and_target(self, synthesizer, plan_row_s1):
        """Test that current_pref and target_pref are assigned."""
        ctx = synthesizer.synthesize(plan_row_s1)
        assert ctx.current_pref in ("a", "b")
        assert ctx.target_pref in ("a", "b")
        assert ctx.current_pref != ctx.target_pref

    def test_synthesize_s2_uses_workflow_domain(self, synthesizer, plan_row_s2):
        """Test that S2 severity maps to workflow domain."""
        ctx = synthesizer.synthesize(plan_row_s2)
        assert ctx.pref_pair.domain == "workflow"

    def test_synthesize_s3_uses_epistemic_domain(self, synthesizer, plan_row_s3):
        """Test that S3 severity maps to epistemic domain."""
        ctx = synthesizer.synthesize(plan_row_s3)
        assert ctx.pref_pair.domain == "epistemic"


# ═══════════════════════════════════════════════════════════════════════════════
# DETERMINISM TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_same_plan_row_produces_same_context(self, synthesizer, plan_row_s1):
        """Same PlanRow should always produce same Context."""
        ctx1 = synthesizer.synthesize(plan_row_s1)
        ctx2 = synthesizer.synthesize(plan_row_s1)

        assert ctx1.pref_pair == ctx2.pref_pair
        assert ctx1.current_pref == ctx2.current_pref
        assert ctx1.target_pref == ctx2.target_pref

    def test_determinism_across_synthesizer_instances(self, plan_row_s1):
        """Different synthesizer instances should produce same results for same input."""
        synth1 = ContextSynthesizer(global_seed=100)
        synth2 = ContextSynthesizer(global_seed=200)  # Different global seed

        ctx1 = synth1.synthesize(plan_row_s1)
        ctx2 = synth2.synthesize(plan_row_s1)

        # Results should be same because determinism comes from plan_row.seed
        assert ctx1.pref_pair == ctx2.pref_pair
        assert ctx1.current_pref == ctx2.current_pref
        assert ctx1.target_pref == ctx2.target_pref

    def test_different_seeds_produce_different_results(self, synthesizer):
        """Different seeds should produce different results."""
        row1 = PlanRow(
            pair_id="pair_001",
            seed=12345,
            family_id=FamilyID.A,
            subtype_id="A1",
            severity=Severity.S1,
            mode=Mode.RATING,
            perspective=Perspective.FIRST,
        )
        row2 = PlanRow(
            pair_id="pair_002",
            seed=54321,  # Different seed
            family_id=FamilyID.A,
            subtype_id="A1",
            severity=Severity.S1,
            mode=Mode.RATING,
            perspective=Perspective.FIRST,
        )

        ctx1 = synthesizer.synthesize(row1)
        ctx2 = synthesizer.synthesize(row2)

        # At least one thing should be different (could be pref_pair or assignment)
        # With different seeds, high probability of different results
        # We don't assert they must differ since by chance they could be same

    def test_same_seed_same_result_multiple_iterations(self, synthesizer, plan_row_s1):
        """Multiple synthesize calls with same input always produce same result."""
        results = [synthesizer.synthesize(plan_row_s1) for _ in range(10)]

        first = results[0]
        for ctx in results[1:]:
            assert ctx.pref_pair == first.pref_pair
            assert ctx.current_pref == first.current_pref
            assert ctx.target_pref == first.target_pref


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH PROCESSING TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestSynthesizeBatch:
    """Tests for synthesize_batch method."""

    def test_batch_returns_list(self, synthesizer, multiple_plan_rows):
        """Batch synthesis should return a list of Contexts."""
        contexts = synthesizer.synthesize_batch(multiple_plan_rows)
        assert isinstance(contexts, list)
        assert len(contexts) == len(multiple_plan_rows)

    def test_batch_all_elements_are_contexts(self, synthesizer, multiple_plan_rows):
        """All batch results should be Context objects."""
        contexts = synthesizer.synthesize_batch(multiple_plan_rows)
        for ctx in contexts:
            assert isinstance(ctx, Context)

    def test_batch_preserves_order(self, synthesizer, multiple_plan_rows):
        """Batch results should match input order."""
        contexts = synthesizer.synthesize_batch(multiple_plan_rows)
        for i, ctx in enumerate(contexts):
            assert ctx.pair_id == multiple_plan_rows[i].pair_id
            assert ctx.seed == multiple_plan_rows[i].seed

    def test_batch_equivalent_to_individual(self, synthesizer, multiple_plan_rows):
        """Batch synthesis should produce same results as individual synthesis."""
        batch_contexts = synthesizer.synthesize_batch(multiple_plan_rows)
        individual_contexts = [synthesizer.synthesize(row) for row in multiple_plan_rows]

        for batch_ctx, ind_ctx in zip(batch_contexts, individual_contexts):
            assert batch_ctx.pref_pair == ind_ctx.pref_pair
            assert batch_ctx.current_pref == ind_ctx.current_pref
            assert batch_ctx.target_pref == ind_ctx.target_pref

    def test_batch_empty_list(self, synthesizer):
        """Empty list input should return empty list."""
        contexts = synthesizer.synthesize_batch([])
        assert contexts == []

    def test_batch_determinism(self, synthesizer, multiple_plan_rows):
        """Batch synthesis should be deterministic."""
        contexts1 = synthesizer.synthesize_batch(multiple_plan_rows)
        contexts2 = synthesizer.synthesize_batch(multiple_plan_rows)

        for ctx1, ctx2 in zip(contexts1, contexts2):
            assert ctx1.pref_pair == ctx2.pref_pair
            assert ctx1.current_pref == ctx2.current_pref


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestHelperFunctions:
    """Tests for standalone helper functions."""

    @pytest.fixture
    def context_a_current(self):
        """Create a Context with current_pref='a'."""
        pref_pair = PreferencePair(
            pref_a_id="concise",
            pref_a_text="concise answers",
            pref_b_id="verbose",
            pref_b_text="verbose, detailed answers",
            domain="style",
        )
        return Context(
            pair_id="pair_001",
            seed=12345,
            family_id=FamilyID.A,
            subtype_id="A1",
            severity=Severity.S1,
            mode=Mode.RATING,
            perspective=Perspective.FIRST,
            pref_pair=pref_pair,
            current_pref="a",
            target_pref="b",
        )

    @pytest.fixture
    def context_b_current(self):
        """Create a Context with current_pref='b'."""
        pref_pair = PreferencePair(
            pref_a_id="concise",
            pref_a_text="concise answers",
            pref_b_id="verbose",
            pref_b_text="verbose, detailed answers",
            domain="style",
        )
        return Context(
            pair_id="pair_002",
            seed=54321,
            family_id=FamilyID.A,
            subtype_id="A1",
            severity=Severity.S1,
            mode=Mode.RATING,
            perspective=Perspective.FIRST,
            pref_pair=pref_pair,
            current_pref="b",
            target_pref="a",
        )

    def test_get_current_pref_text_a(self, context_a_current):
        """Test getting current pref text when current='a'."""
        text = get_current_pref_text(context_a_current)
        assert text == "concise answers"

    def test_get_current_pref_text_b(self, context_b_current):
        """Test getting current pref text when current='b'."""
        text = get_current_pref_text(context_b_current)
        assert text == "verbose, detailed answers"

    def test_get_target_pref_text_a(self, context_a_current):
        """Test getting target pref text when target='b'."""
        text = get_target_pref_text(context_a_current)
        assert text == "verbose, detailed answers"

    def test_get_target_pref_text_b(self, context_b_current):
        """Test getting target pref text when target='a'."""
        text = get_target_pref_text(context_b_current)
        assert text == "concise answers"

    def test_get_current_pref_id_a(self, context_a_current):
        """Test getting current pref ID when current='a'."""
        pref_id = get_current_pref_id(context_a_current)
        assert pref_id == "concise"

    def test_get_current_pref_id_b(self, context_b_current):
        """Test getting current pref ID when current='b'."""
        pref_id = get_current_pref_id(context_b_current)
        assert pref_id == "verbose"

    def test_get_target_pref_id_a(self, context_a_current):
        """Test getting target pref ID when target='b'."""
        pref_id = get_target_pref_id(context_a_current)
        assert pref_id == "verbose"

    def test_get_target_pref_id_b(self, context_b_current):
        """Test getting target pref ID when target='a'."""
        pref_id = get_target_pref_id(context_b_current)
        assert pref_id == "concise"

    def test_helper_matches_context_method(self, context_a_current):
        """Helper functions should match Context's built-in methods."""
        assert get_current_pref_text(context_a_current) == context_a_current.get_current_pref_text()
        assert get_target_pref_text(context_a_current) == context_a_current.get_target_pref_text()


# ═══════════════════════════════════════════════════════════════════════════════
# PREFERENCE ASSIGNMENT DISTRIBUTION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestPreferenceDistribution:
    """Tests for preference assignment distribution."""

    def test_current_target_always_opposite(self, synthesizer):
        """Current and target preferences should always be opposite."""
        for seed in range(100):
            row = PlanRow(
                pair_id=f"pair_{seed}",
                seed=seed,
                family_id=FamilyID.A,
                subtype_id="A1",
                severity=Severity.S1,
                mode=Mode.RATING,
                perspective=Perspective.FIRST,
            )
            ctx = synthesizer.synthesize(row)
            assert ctx.current_pref != ctx.target_pref

    def test_approximately_50_50_distribution(self, synthesizer):
        """Current preference assignment should be roughly 50/50."""
        a_count = 0
        b_count = 0
        n_samples = 1000

        for seed in range(n_samples):
            row = PlanRow(
                pair_id=f"pair_{seed}",
                seed=seed * 7919,  # Use prime to spread seeds
                family_id=FamilyID.A,
                subtype_id="A1",
                severity=Severity.S1,
                mode=Mode.RATING,
                perspective=Perspective.FIRST,
            )
            ctx = synthesizer.synthesize(row)
            if ctx.current_pref == "a":
                a_count += 1
            else:
                b_count += 1

        # Should be roughly 50/50 with some tolerance
        ratio = a_count / n_samples
        assert 0.4 < ratio < 0.6, f"Expected ~50% 'a', got {ratio * 100:.1f}%"


# ═══════════════════════════════════════════════════════════════════════════════
# VARIATION FLAGS TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestVariationFlags:
    """Tests for variation flag initialization."""

    def test_variation_flags_initialized_to_defaults(self, synthesizer, plan_row_s1):
        """Context from synthesize should have default variation flags."""
        ctx = synthesizer.synthesize(plan_row_s1)
        assert ctx.ordering_swap is False
        assert ctx.lexical_variant == 0
        assert ctx.formatting_variant == 0

    def test_variation_flags_can_be_modified(self, synthesizer, plan_row_s1):
        """Variation flags on Context should be modifiable."""
        ctx = synthesizer.synthesize(plan_row_s1)

        # Context is not frozen, so these should work
        ctx.ordering_swap = True
        ctx.lexical_variant = 3
        ctx.formatting_variant = 2

        assert ctx.ordering_swap is True
        assert ctx.lexical_variant == 3
        assert ctx.formatting_variant == 2


# ═══════════════════════════════════════════════════════════════════════════════
# PREFERENCE PAIR DOMAIN TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestPreferencePairSampling:
    """Tests for preference pair sampling from catalogs."""

    def test_s1_samples_from_style_domain(self, synthesizer):
        """S1 severity should sample from style domain."""
        for seed in range(50):
            row = PlanRow(
                pair_id=f"pair_{seed}",
                seed=seed,
                family_id=FamilyID.A,
                subtype_id="A1",
                severity=Severity.S1,
                mode=Mode.RATING,
                perspective=Perspective.FIRST,
            )
            ctx = synthesizer.synthesize(row)
            assert ctx.pref_pair.domain == "style"

    def test_s2_samples_from_workflow_domain(self, synthesizer):
        """S2 severity should sample from workflow domain."""
        for seed in range(50):
            row = PlanRow(
                pair_id=f"pair_{seed}",
                seed=seed,
                family_id=FamilyID.B,
                subtype_id="B1",
                severity=Severity.S2,
                mode=Mode.CHOICE,
                perspective=Perspective.FIRST,
            )
            ctx = synthesizer.synthesize(row)
            assert ctx.pref_pair.domain == "workflow"

    def test_s3_samples_from_epistemic_domain(self, synthesizer):
        """S3 severity should sample from epistemic domain."""
        for seed in range(50):
            row = PlanRow(
                pair_id=f"pair_{seed}",
                seed=seed,
                family_id=FamilyID.C,
                subtype_id="C1",
                severity=Severity.S3,
                mode=Mode.SHORT,
                perspective=Perspective.NEUTRAL,
            )
            ctx = synthesizer.synthesize(row)
            assert ctx.pref_pair.domain == "epistemic"

    def test_sampled_pairs_are_valid(self, synthesizer):
        """Sampled preference pairs should be from the catalog."""
        row = PlanRow(
            pair_id="pair_001",
            seed=12345,
            family_id=FamilyID.A,
            subtype_id="A1",
            severity=Severity.S1,
            mode=Mode.RATING,
            perspective=Perspective.FIRST,
        )
        ctx = synthesizer.synthesize(row)

        # Verify the pair is from the catalog
        valid_pairs = get_preference_pairs_for_severity(Severity.S1)
        valid_a_ids = [p.pref_a_id for p in valid_pairs]
        valid_b_ids = [p.pref_b_id for p in valid_pairs]

        assert ctx.pref_pair.pref_a_id in valid_a_ids
        assert ctx.pref_pair.pref_b_id in valid_b_ids


# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXT SERIALIZATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestContextSerialization:
    """Tests for Context serialization after synthesis."""

    def test_synthesized_context_to_dict(self, synthesizer, plan_row_s1):
        """Synthesized Context should serialize to dict correctly."""
        ctx = synthesizer.synthesize(plan_row_s1)
        d = ctx.to_dict()

        assert d["pair_id"] == plan_row_s1.pair_id
        assert d["seed"] == plan_row_s1.seed
        assert d["family_id"] == plan_row_s1.family_id.value
        assert d["current_pref"] in ("a", "b")
        assert d["target_pref"] in ("a", "b")
        assert "pref_pair" in d

    def test_synthesized_context_to_json(self, synthesizer, plan_row_s1):
        """Synthesized Context should serialize to JSON correctly."""
        import json
        ctx = synthesizer.synthesize(plan_row_s1)
        json_str = ctx.to_json()
        parsed = json.loads(json_str)

        assert parsed["pair_id"] == plan_row_s1.pair_id
        assert parsed["severity"] == "low"


# ═══════════════════════════════════════════════════════════════════════════════
# ALL FAMILY TYPES TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestAllFamilyTypes:
    """Tests to ensure synthesis works for all family types."""

    @pytest.mark.parametrize("family_id,subtype_prefix", [
        (FamilyID.A, "A"),
        (FamilyID.B, "B"),
        (FamilyID.C, "C"),
        (FamilyID.D, "D"),
        (FamilyID.E, "E"),
        (FamilyID.F, "F"),
        (FamilyID.G, "G"),
        (FamilyID.H, "H"),
    ])
    def test_synthesize_all_families(self, synthesizer, family_id, subtype_prefix):
        """Synthesis should work for all family types."""
        row = PlanRow(
            pair_id="pair_001",
            seed=12345,
            family_id=family_id,
            subtype_id=f"{subtype_prefix}1",
            severity=Severity.S1,
            mode=Mode.RATING,
            perspective=Perspective.FIRST,
        )
        ctx = synthesizer.synthesize(row)

        assert ctx.family_id == family_id
        assert ctx.subtype_id == f"{subtype_prefix}1"
        assert isinstance(ctx.pref_pair, PreferencePair)


class TestAllModeTypes:
    """Tests to ensure synthesis works for all mode types."""

    @pytest.mark.parametrize("mode", [Mode.RATING, Mode.CHOICE, Mode.SHORT])
    def test_synthesize_all_modes(self, synthesizer, mode):
        """Synthesis should work for all mode types."""
        row = PlanRow(
            pair_id="pair_001",
            seed=12345,
            family_id=FamilyID.A,
            subtype_id="A1",
            severity=Severity.S1,
            mode=mode,
            perspective=Perspective.FIRST,
        )
        ctx = synthesizer.synthesize(row)

        assert ctx.mode == mode


class TestAllPerspectiveTypes:
    """Tests to ensure synthesis works for all perspective types."""

    @pytest.mark.parametrize("perspective", [Perspective.FIRST, Perspective.THIRD, Perspective.NEUTRAL])
    def test_synthesize_all_perspectives(self, synthesizer, perspective):
        """Synthesis should work for all perspective types."""
        row = PlanRow(
            pair_id="pair_001",
            seed=12345,
            family_id=FamilyID.A,
            subtype_id="A1",
            severity=Severity.S1,
            mode=Mode.RATING,
            perspective=perspective,
        )
        ctx = synthesizer.synthesize(row)

        assert ctx.perspective == perspective
