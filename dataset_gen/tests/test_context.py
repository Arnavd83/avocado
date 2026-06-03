"""
Tests for the Context Synthesis Module (Layer 2 — Stage 2a).

Tests cover:
- ContextSynthesizer creation and basic usage
- New-shape Context fields: lexical_variant, style_directive_id,
  target_intensity, catalog_version
- Deterministic behavior (same PlanRow → identical Context)
- Per-pair direction balancing (~50/50 current_pref per pref_pair)
- Severity-stratified preference selection
- Schema invariants (Context.__post_init__) hold for synthesizer outputs
- Helper functions for preference text / id retrieval
- Batch synthesis
"""

import pytest

from dataset_gen.src.schema import (
    FamilyID,
    Severity,
    Mode,
    Perspective,
    PlanRow,
    Context,
    PreferencePair,
    validate_context,
)
from dataset_gen.src import context as context_mod
from dataset_gen.src.context import (
    ContextSynthesizer,
    get_current_pref_text,
    get_target_pref_text,
    get_current_pref_id,
    get_target_pref_id,
)
from dataset_gen.src import catalogs
from dataset_gen.src.catalogs import (
    CATALOG_VERSION,
    SEVERITY_TO_CATEGORY_POOL,
    _symmetric_pairs,
)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


def make_plan_row(
    *,
    pair_id="pair_000001",
    seed=12345,
    family_id=FamilyID.A,
    subtype_id="A1",
    severity=Severity.S1,
    mode=Mode.CHOICE,
    perspective=Perspective.FIRST,
    style_directive_id=3,
    target_intensity=5,
) -> PlanRow:
    """Construct a PlanRow with sensible defaults for the new schema."""
    return PlanRow(
        pair_id=pair_id,
        seed=seed,
        family_id=family_id,
        subtype_id=subtype_id,
        severity=severity,
        mode=mode,
        perspective=perspective,
        style_directive_id=style_directive_id,
        target_intensity=target_intensity,
    )


@pytest.fixture
def synthesizer():
    """Create a ContextSynthesizer for testing."""
    return ContextSynthesizer(global_seed=42)


@pytest.fixture
def plan_row_s1():
    """A PlanRow with S1 (low) severity."""
    return make_plan_row(
        pair_id="pair_000001", seed=12345, family_id=FamilyID.A,
        subtype_id="A1", severity=Severity.S1,
    )


@pytest.fixture
def plan_row_s2():
    """A PlanRow with S2 (medium) severity."""
    return make_plan_row(
        pair_id="pair_000002", seed=67890, family_id=FamilyID.B,
        subtype_id="B1", severity=Severity.S2, mode=Mode.SHORT,
        perspective=Perspective.THIRD,
    )


@pytest.fixture
def plan_row_s3():
    """A PlanRow with S3 (high) severity."""
    return make_plan_row(
        pair_id="pair_000003", seed=11111, family_id=FamilyID.D,
        subtype_id="D1", severity=Severity.S3, mode=Mode.SHORT,
    )


@pytest.fixture
def multiple_plan_rows():
    """Multiple PlanRows for batch testing."""
    return [
        make_plan_row(
            pair_id=f"pair_{i:06d}",
            seed=i * 1000 + 42,
            family_id=FamilyID.A,
            subtype_id="A1",
            severity=Severity.S1,
        )
        for i in range(10)
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# SYNTHESIZER CREATION
# ═══════════════════════════════════════════════════════════════════════════════


class TestContextSynthesizerCreation:
    def test_create_synthesizer(self):
        synth = ContextSynthesizer(global_seed=42)
        assert synth.global_seed == 42

    def test_create_synthesizer_different_seeds(self):
        synth1 = ContextSynthesizer(global_seed=1)
        synth2 = ContextSynthesizer(global_seed=2)
        assert synth1.global_seed != synth2.global_seed


# ═══════════════════════════════════════════════════════════════════════════════
# BASIC SYNTHESIS + NEW FIELDS
# ═══════════════════════════════════════════════════════════════════════════════


class TestSynthesizeBasic:
    def test_synthesize_returns_context(self, synthesizer, plan_row_s1):
        ctx = synthesizer.synthesize(plan_row_s1)
        assert isinstance(ctx, Context)

    def test_synthesize_preserves_plan_row_fields(self, synthesizer, plan_row_s1):
        ctx = synthesizer.synthesize(plan_row_s1)
        assert ctx.pair_id == plan_row_s1.pair_id
        assert ctx.seed == plan_row_s1.seed
        assert ctx.family_id == plan_row_s1.family_id
        assert ctx.subtype_id == plan_row_s1.subtype_id
        assert ctx.severity == plan_row_s1.severity
        assert ctx.mode == plan_row_s1.mode
        assert ctx.perspective == plan_row_s1.perspective

    def test_synthesize_creates_preference_pair(self, synthesizer, plan_row_s1):
        ctx = synthesizer.synthesize(plan_row_s1)
        assert isinstance(ctx.pref_pair, PreferencePair)

    def test_synthesize_assigns_current_and_target(self, synthesizer, plan_row_s1):
        ctx = synthesizer.synthesize(plan_row_s1)
        assert ctx.current_pref in ("a", "b")
        assert ctx.target_pref in ("a", "b")
        assert ctx.current_pref != ctx.target_pref

    def test_lexical_variant_in_range(self, synthesizer):
        """lexical_variant must be sampled uniformly in [0, 10)."""
        for seed in range(200):
            ctx = synthesizer.synthesize(make_plan_row(pair_id=f"p{seed}", seed=seed))
            assert 0 <= ctx.lexical_variant < 10

    def test_propagates_style_directive_id(self, synthesizer):
        row = make_plan_row(style_directive_id=7)
        ctx = synthesizer.synthesize(row)
        assert ctx.style_directive_id == 7

    def test_propagates_target_intensity(self, synthesizer):
        row = make_plan_row(target_intensity=2)
        ctx = synthesizer.synthesize(row)
        assert ctx.target_intensity == 2

    def test_sets_catalog_version(self, synthesizer, plan_row_s1):
        ctx = synthesizer.synthesize(plan_row_s1)
        assert ctx.catalog_version == CATALOG_VERSION
        assert ctx.catalog_version == catalogs.CATALOG_VERSION

    def test_template_fields_left_none(self, synthesizer, plan_row_s1):
        """template_id / is_holdout are populated by Layer 4, not here."""
        ctx = synthesizer.synthesize(plan_row_s1)
        assert ctx.template_id is None
        assert ctx.is_holdout is None

    def test_does_not_mutate_plan_row(self, synthesizer):
        """PlanRow is frozen; synthesize must not attempt to mutate it."""
        row = make_plan_row(style_directive_id=4, target_intensity=6)
        before = row.to_dict()
        synthesizer.synthesize(row)
        assert row.to_dict() == before


# ═══════════════════════════════════════════════════════════════════════════════
# DETERMINISM
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeterminism:
    def test_same_plan_row_produces_same_context(self, synthesizer, plan_row_s1):
        ctx1 = synthesizer.synthesize(plan_row_s1)
        ctx2 = synthesizer.synthesize(plan_row_s1)
        assert ctx1.to_dict() == ctx2.to_dict()

    def test_identical_fields_produce_identical_context(self, synthesizer):
        """Two PlanRows with identical fields produce identical Contexts."""
        row1 = make_plan_row(pair_id="pair_X", seed=999)
        row2 = make_plan_row(pair_id="pair_X", seed=999)
        ctx1 = synthesizer.synthesize(row1)
        ctx2 = synthesizer.synthesize(row2)
        assert ctx1.to_dict() == ctx2.to_dict()

    def test_determinism_independent_of_global_seed(self, plan_row_s1):
        """Determinism comes from plan_row.seed, not the synthesizer global seed."""
        ctx1 = ContextSynthesizer(global_seed=100).synthesize(plan_row_s1)
        ctx2 = ContextSynthesizer(global_seed=200).synthesize(plan_row_s1)
        assert ctx1.to_dict() == ctx2.to_dict()

    def test_same_seed_same_result_multiple_iterations(self, synthesizer, plan_row_s1):
        results = [synthesizer.synthesize(plan_row_s1) for _ in range(10)]
        first = results[0].to_dict()
        for ctx in results[1:]:
            assert ctx.to_dict() == first


# ═══════════════════════════════════════════════════════════════════════════════
# DIRECTION BALANCING
# ═══════════════════════════════════════════════════════════════════════════════


class TestDirectionBalancing:
    @pytest.fixture
    def forced_pair(self):
        """A single concrete preference pair to force across many pair_ids."""
        return PreferencePair(
            pref_a_id="concise",
            pref_a_text="concise answers",
            pref_b_id="verbose",
            pref_b_text="verbose, detailed answers",
            domain="length",
            domain_category="communication_style",
            severity=Severity.S1,
        )

    def test_per_pair_split_in_range(self, synthesizer, forced_pair, monkeypatch):
        """
        With the same pref_pair forced across 200 distinct pair_ids, the
        current_pref="a" rate should land in [40%, 60%].
        """
        monkeypatch.setattr(
            context_mod, "sample_preference_pair", lambda severity, rng: forced_pair
        )
        a_count = 0
        n = 200
        for i in range(n):
            row = make_plan_row(pair_id=f"pair_{i:06d}", seed=i * 31 + 1)
            ctx = synthesizer.synthesize(row)
            # Confirm the pair really was forced.
            assert ctx.pref_pair is forced_pair
            if ctx.current_pref == "a":
                a_count += 1
        rate = a_count / n
        assert 0.40 <= rate <= 0.60, f"current_pref='a' rate was {rate:.1%}"

    def test_direction_is_deterministic_per_pair_id(self, forced_pair):
        """_choose_direction is a pure function of (pair ids, pair_id)."""
        d1 = ContextSynthesizer._choose_direction(forced_pair, "pair_42")
        d2 = ContextSynthesizer._choose_direction(forced_pair, "pair_42")
        assert d1 == d2
        assert d1 in (("a", "b"), ("b", "a"))

    def test_current_target_always_opposite(self, synthesizer):
        for seed in range(100):
            ctx = synthesizer.synthesize(make_plan_row(pair_id=f"p{seed}", seed=seed))
            assert ctx.current_pref != ctx.target_pref


# ═══════════════════════════════════════════════════════════════════════════════
# SEVERITY-STRATIFIED PREFERENCE SELECTION
# ═══════════════════════════════════════════════════════════════════════════════


class TestSeveritySelection:
    @pytest.mark.parametrize("severity", [Severity.S1, Severity.S2, Severity.S3])
    def test_domain_category_in_severity_pool(self, synthesizer, severity):
        pool = SEVERITY_TO_CATEGORY_POOL[severity]
        for seed in range(50):
            row = make_plan_row(pair_id=f"p{seed}", seed=seed, severity=severity)
            ctx = synthesizer.synthesize(row)
            assert ctx.pref_pair.domain_category in pool

    def test_s1_pair_severity_matches(self, synthesizer):
        for seed in range(50):
            row = make_plan_row(pair_id=f"p{seed}", seed=seed, severity=Severity.S1)
            ctx = synthesizer.synthesize(row)
            assert ctx.pref_pair.severity == Severity.S1

    def test_sampled_pair_is_active_symmetric(self, synthesizer):
        """Sampled pairs come from the active (symmetric) pool of their category."""
        for seed in range(50):
            row = make_plan_row(pair_id=f"p{seed}", seed=seed, severity=Severity.S3)
            ctx = synthesizer.synthesize(row)
            category = ctx.pref_pair.domain_category
            active_ids = {id(p) for p in _symmetric_pairs(category)}
            assert id(ctx.pref_pair) in active_ids
            assert ctx.pref_pair.is_symmetric is True


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA INVARIANTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestSchemaInvariants:
    @pytest.mark.parametrize("severity", [Severity.S1, Severity.S2, Severity.S3])
    def test_validate_context_passes(self, synthesizer, severity):
        """Every synthesizer output should satisfy validate_context (no errors)."""
        for seed in range(30):
            row = make_plan_row(
                pair_id=f"p{seed}", seed=seed * 7919 + 1, severity=severity
            )
            ctx = synthesizer.synthesize(row)
            assert validate_context(ctx) == []

    def test_post_init_bounds_hold(self, synthesizer):
        """Construction (post_init) already enforces bounds; spot-check them."""
        for seed in range(50):
            ctx = synthesizer.synthesize(make_plan_row(pair_id=f"p{seed}", seed=seed))
            assert 0 <= ctx.lexical_variant <= 9
            assert 0 <= ctx.style_directive_id <= 9
            assert 1 <= ctx.target_intensity <= 7


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════


class TestSynthesizeBatch:
    def test_batch_returns_list(self, synthesizer, multiple_plan_rows):
        contexts = synthesizer.synthesize_batch(multiple_plan_rows)
        assert isinstance(contexts, list)
        assert len(contexts) == len(multiple_plan_rows)

    def test_batch_all_elements_are_contexts(self, synthesizer, multiple_plan_rows):
        for ctx in synthesizer.synthesize_batch(multiple_plan_rows):
            assert isinstance(ctx, Context)

    def test_batch_preserves_order(self, synthesizer, multiple_plan_rows):
        contexts = synthesizer.synthesize_batch(multiple_plan_rows)
        for i, ctx in enumerate(contexts):
            assert ctx.pair_id == multiple_plan_rows[i].pair_id
            assert ctx.seed == multiple_plan_rows[i].seed

    def test_batch_equivalent_to_individual(self, synthesizer, multiple_plan_rows):
        batch = synthesizer.synthesize_batch(multiple_plan_rows)
        individual = [synthesizer.synthesize(r) for r in multiple_plan_rows]
        for b, i in zip(batch, individual):
            assert b.to_dict() == i.to_dict()

    def test_batch_empty_list(self, synthesizer):
        assert synthesizer.synthesize_batch([]) == []


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


class TestHelperFunctions:
    @pytest.fixture
    def pref_pair(self):
        return PreferencePair(
            pref_a_id="concise",
            pref_a_text="concise answers",
            pref_b_id="verbose",
            pref_b_text="verbose, detailed answers",
            domain="length",
            domain_category="communication_style",
            severity=Severity.S1,
        )

    @pytest.fixture
    def context_a_current(self, pref_pair):
        return Context(
            pair_id="pair_001", seed=12345, family_id=FamilyID.A, subtype_id="A1",
            severity=Severity.S1, mode=Mode.CHOICE, perspective=Perspective.FIRST,
            pref_pair=pref_pair, current_pref="a", target_pref="b",
        )

    @pytest.fixture
    def context_b_current(self, pref_pair):
        return Context(
            pair_id="pair_002", seed=54321, family_id=FamilyID.A, subtype_id="A1",
            severity=Severity.S1, mode=Mode.CHOICE, perspective=Perspective.FIRST,
            pref_pair=pref_pair, current_pref="b", target_pref="a",
        )

    def test_get_current_pref_text_a(self, context_a_current):
        assert get_current_pref_text(context_a_current) == "concise answers"

    def test_get_current_pref_text_b(self, context_b_current):
        assert get_current_pref_text(context_b_current) == "verbose, detailed answers"

    def test_get_target_pref_text_a(self, context_a_current):
        assert get_target_pref_text(context_a_current) == "verbose, detailed answers"

    def test_get_target_pref_text_b(self, context_b_current):
        assert get_target_pref_text(context_b_current) == "concise answers"

    def test_get_current_pref_id_a(self, context_a_current):
        assert get_current_pref_id(context_a_current) == "concise"

    def test_get_current_pref_id_b(self, context_b_current):
        assert get_current_pref_id(context_b_current) == "verbose"

    def test_get_target_pref_id_a(self, context_a_current):
        assert get_target_pref_id(context_a_current) == "verbose"

    def test_get_target_pref_id_b(self, context_b_current):
        assert get_target_pref_id(context_b_current) == "concise"

    def test_helper_matches_context_method(self, context_a_current):
        assert get_current_pref_text(context_a_current) == context_a_current.get_current_pref_text()
        assert get_target_pref_text(context_a_current) == context_a_current.get_target_pref_text()


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-PARAMETER COVERAGE
# ═══════════════════════════════════════════════════════════════════════════════


class TestAllFamilyTypes:
    @pytest.mark.parametrize("family_id,subtype_prefix", [
        (FamilyID.A, "A"),
        (FamilyID.B, "B"),
        (FamilyID.D, "D"),
        (FamilyID.E, "E"),
        (FamilyID.F, "F"),
        (FamilyID.G, "G"),
        (FamilyID.H, "H"),
    ])
    def test_synthesize_all_families(self, synthesizer, family_id, subtype_prefix):
        row = make_plan_row(
            family_id=family_id, subtype_id=f"{subtype_prefix}1", severity=Severity.S1
        )
        ctx = synthesizer.synthesize(row)
        assert ctx.family_id == family_id
        assert ctx.subtype_id == f"{subtype_prefix}1"
        assert isinstance(ctx.pref_pair, PreferencePair)


class TestAllModeTypes:
    @pytest.mark.parametrize("mode", [Mode.CHOICE, Mode.SHORT])
    def test_synthesize_all_modes(self, synthesizer, mode):
        ctx = synthesizer.synthesize(make_plan_row(mode=mode))
        assert ctx.mode == mode


class TestAllPerspectiveTypes:
    @pytest.mark.parametrize("perspective", [Perspective.FIRST, Perspective.THIRD])
    def test_synthesize_all_perspectives(self, synthesizer, perspective):
        ctx = synthesizer.synthesize(make_plan_row(perspective=perspective))
        assert ctx.perspective == perspective


# ═══════════════════════════════════════════════════════════════════════════════
# SERIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════


class TestContextSerialization:
    def test_synthesized_context_to_dict(self, synthesizer, plan_row_s1):
        d = synthesizer.synthesize(plan_row_s1).to_dict()
        assert d["pair_id"] == plan_row_s1.pair_id
        assert d["seed"] == plan_row_s1.seed
        assert d["family_id"] == plan_row_s1.family_id.value
        assert d["current_pref"] in ("a", "b")
        assert d["target_pref"] in ("a", "b")
        assert d["catalog_version"] == CATALOG_VERSION
        assert "pref_pair" in d

    def test_synthesized_context_to_json(self, synthesizer, plan_row_s1):
        import json
        parsed = json.loads(synthesizer.synthesize(plan_row_s1).to_json())
        assert parsed["pair_id"] == plan_row_s1.pair_id
        assert parsed["severity"] == "low"
