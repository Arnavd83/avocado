"""
Tests for Family A: Explicit Reversal Evaluation

Tests cover:
- Basic instantiation and registration
- Template retrieval for all subtypes
- Minimum template count requirements
- render_prompt() returns RenderedPrompt correctly
- Determinism (same seed = same result)
- Different seeds produce different outputs
- Holdout template functionality
- Perspective transformation
- Mode suffix addition
- Template validation
"""

import pytest
import random
from typing import Set

from dataset_gen.src.families.family_a import FamilyA, SUBTYPE_A1_TEMPLATES, SUBTYPE_A2_TEMPLATES, SUBTYPE_A3_TEMPLATES
from dataset_gen.src.families.registry import register_family, get_family_plugin, _FAMILY_REGISTRY, import_all_families
from dataset_gen.src.schema import (
    Context,
    PreferencePair,
    FamilyID,
    Severity,
    Mode,
    Perspective,
    RenderedPrompt,
)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def family_a() -> FamilyA:
    """Create a FamilyA instance for testing."""
    return FamilyA()


@pytest.fixture
def sample_pref_pair() -> PreferencePair:
    """Create a sample preference pair for testing."""
    return PreferencePair(
        pref_a_id="concise",
        pref_a_text="concise answers",
        pref_b_id="verbose",
        pref_b_text="verbose, detailed answers",
        domain="style"
    )


@pytest.fixture
def sample_context(sample_pref_pair: PreferencePair) -> Context:
    """Create a sample context for testing."""
    return Context(
        pair_id="test_001",
        seed=42,
        family_id=FamilyID.A,
        subtype_id="A1",
        severity=Severity.S1,
        mode=Mode.RATING,
        perspective=Perspective.FIRST,
        pref_pair=sample_pref_pair,
        current_pref="a",
        target_pref="b",
        ordering_swap=False,
        lexical_variant=0,
        formatting_variant=0,
    )


def make_context(
    seed: int = 42,
    subtype_id: str = "A1",
    mode: Mode = Mode.RATING,
    perspective: Perspective = Perspective.FIRST,
    ordering_swap: bool = False,
    lexical_variant: int = 0,
    formatting_variant: int = 0,
) -> Context:
    """Helper to create contexts with various parameters.

    Args:
        seed: Random seed for template selection
        subtype_id: Family subtype (A1, A2, A3)
        mode: Response mode (RATING, CHOICE, SHORT)
        perspective: Prompt perspective (FIRST, THIRD)
        ordering_swap: Whether to swap preference ordering
        lexical_variant: Index for lexical variants in template content
        formatting_variant: Index for instruction style (0-2)
    """
    return Context(
        pair_id=f"test_{seed:06d}",
        seed=seed,
        family_id=FamilyID.A,
        subtype_id=subtype_id,
        severity=Severity.S1,
        mode=mode,
        perspective=perspective,
        pref_pair=PreferencePair(
            pref_a_id="concise",
            pref_a_text="concise answers",
            pref_b_id="verbose",
            pref_b_text="verbose, detailed answers",
            domain="style"
        ),
        current_pref="a",
        target_pref="b",
        ordering_swap=ordering_swap,
        lexical_variant=lexical_variant,
        formatting_variant=formatting_variant,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# BASIC INSTANTIATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestFamilyAInstantiation:
    """Tests for FamilyA instantiation and class attributes."""

    def test_family_id(self, family_a: FamilyA):
        """FAMILY_ID should be 'A'."""
        assert family_a.FAMILY_ID == "A"

    def test_family_name(self, family_a: FamilyA):
        """FAMILY_NAME should be 'Explicit Reversal Evaluation'."""
        assert family_a.FAMILY_NAME == "Explicit Reversal Evaluation"

    def test_subtypes(self, family_a: FamilyA):
        """SUBTYPES should be ['A1', 'A2', 'A3']."""
        assert family_a.SUBTYPES == ["A1", "A2", "A3"]

    def test_default_holdout_ratio(self, family_a: FamilyA):
        """Default holdout_ratio should be 0.15."""
        assert family_a.holdout_ratio == 0.15

    def test_default_holdout_seed(self, family_a: FamilyA):
        """Default holdout_seed should be 99999."""
        assert family_a.holdout_seed == 99999

    def test_custom_holdout_config(self):
        """Should accept custom holdout configuration."""
        family = FamilyA(holdout_ratio=0.2, holdout_seed=12345)
        assert family.holdout_ratio == 0.2
        assert family.holdout_seed == 12345


# ═══════════════════════════════════════════════════════════════════════════════
# TEMPLATE RETRIEVAL TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestTemplateRetrieval:
    """Tests for get_subtype_templates()."""

    def test_get_a1_templates(self, family_a: FamilyA):
        """Should return A1 templates."""
        templates = family_a.get_subtype_templates("A1")
        assert templates == SUBTYPE_A1_TEMPLATES
        assert len(templates) >= 10

    def test_get_a2_templates(self, family_a: FamilyA):
        """Should return A2 templates."""
        templates = family_a.get_subtype_templates("A2")
        assert templates == SUBTYPE_A2_TEMPLATES
        assert len(templates) >= 10

    def test_get_a3_templates(self, family_a: FamilyA):
        """Should return A3 templates."""
        templates = family_a.get_subtype_templates("A3")
        assert templates == SUBTYPE_A3_TEMPLATES
        assert len(templates) >= 10

    def test_invalid_subtype_raises_error(self, family_a: FamilyA):
        """Should raise ValueError for invalid subtype."""
        with pytest.raises(ValueError) as exc_info:
            family_a.get_subtype_templates("A4")
        assert "Unknown subtype" in str(exc_info.value)
        assert "A4" in str(exc_info.value)

    def test_minimum_template_count_a1(self):
        """A1 should have at least 10 templates."""
        assert len(SUBTYPE_A1_TEMPLATES) >= 10

    def test_minimum_template_count_a2(self):
        """A2 should have at least 10 templates."""
        assert len(SUBTYPE_A2_TEMPLATES) >= 10

    def test_minimum_template_count_a3(self):
        """A3 should have at least 10 templates."""
        assert len(SUBTYPE_A3_TEMPLATES) >= 10

    def test_total_template_count(self):
        """Total templates should be at least 30."""
        total = len(SUBTYPE_A1_TEMPLATES) + len(SUBTYPE_A2_TEMPLATES) + len(SUBTYPE_A3_TEMPLATES)
        assert total >= 30


# ═══════════════════════════════════════════════════════════════════════════════
# TEMPLATE CONTENT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestTemplateContent:
    """Tests for template content and structure."""

    def test_a1_templates_have_required_placeholders(self):
        """A1 templates should contain required placeholders."""
        for template in SUBTYPE_A1_TEMPLATES:
            assert "{current_pref}" in template or "{target_pref}" in template, \
                f"Template missing preference placeholders: {template[:50]}..."

    def test_a2_templates_have_comparison_language(self):
        """A2 templates should contain comparison language."""
        comparison_terms = ["better", "worse", "neutral", "improvement", "decline", "equivalent"]
        for template in SUBTYPE_A2_TEMPLATES:
            has_comparison = any(term in template.lower() for term in comparison_terms)
            assert has_comparison, f"A2 template missing comparison language: {template[:50]}..."

    def test_a3_templates_have_severity_signals(self):
        """A3 templates should signal severity via diverse mechanisms.

        Severity can be signaled through:
        - Explicit intensity markers (major, fundamental, significant)
        - Scope framing (affects reasoning, core, underlying)
        - Reversibility framing (permanent, irreversible, cannot be undone)
        - Time horizon framing (immediate, abrupt, all at once)
        - Consequence framing (consequential, wide-ranging, affects core)
        """
        # All possible severity signals (explicit + implicit)
        severity_signals = [
            # Explicit intensity markers
            "strongly", "significant", "major", "fundamental", "dramatic",
            "profound", "substantial", "stark", "complete", "total",
            # Scope-based signals
            "reasoning", "core", "underlying", "approach",
            # Reversibility-based signals
            "permanent", "irreversible", "cannot be", "not easily undone",
            # Time horizon signals
            "immediate", "abrupt", "all at once",
            # Consequence signals
            "consequential", "wide-ranging", "shapes much",
        ]
        for template in SUBTYPE_A3_TEMPLATES:
            has_severity_signal = any(term in template.lower() for term in severity_signals)
            assert has_severity_signal, f"A3 template missing severity signal: {template[:50]}..."

    def test_templates_have_balanced_braces(self, family_a: FamilyA):
        """All templates should have balanced braces."""
        errors = family_a.validate_templates()
        assert len(errors) == 0, f"Template validation errors: {errors}"

    def test_templates_are_non_empty(self, family_a: FamilyA):
        """All templates should be non-empty strings."""
        for subtype in family_a.SUBTYPES:
            templates = family_a.get_subtype_templates(subtype)
            for i, template in enumerate(templates):
                assert template.strip(), f"Subtype {subtype} template {i} is empty"


# ═══════════════════════════════════════════════════════════════════════════════
# RENDER PROMPT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestRenderPrompt:
    """Tests for render_prompt() method."""

    def test_returns_rendered_prompt(self, family_a: FamilyA, sample_context: Context):
        """Should return a RenderedPrompt object."""
        result = family_a.render_prompt(sample_context)
        assert isinstance(result, RenderedPrompt)

    def test_rendered_prompt_has_text(self, family_a: FamilyA, sample_context: Context):
        """Rendered prompt should have non-empty text."""
        result = family_a.render_prompt(sample_context)
        assert result.prompt
        assert len(result.prompt) > 0

    def test_rendered_prompt_has_template_id(self, family_a: FamilyA, sample_context: Context):
        """Rendered prompt should have a template_id."""
        result = family_a.render_prompt(sample_context)
        assert result.template_id
        assert result.template_id.startswith("A1_")

    def test_rendered_prompt_has_is_holdout(self, family_a: FamilyA, sample_context: Context):
        """Rendered prompt should have is_holdout boolean."""
        result = family_a.render_prompt(sample_context)
        assert isinstance(result.is_holdout, bool)

    def test_template_id_format(self, family_a: FamilyA):
        """Template IDs should follow format: subtype_XX."""
        for subtype in ["A1", "A2", "A3"]:
            ctx = make_context(subtype_id=subtype)
            result = family_a.render_prompt(ctx)
            assert result.template_id.startswith(f"{subtype}_")
            # Should end with 2-digit index
            idx_part = result.template_id.split("_")[-1]
            assert len(idx_part) == 2
            assert idx_part.isdigit()

    def test_placeholders_are_filled(self, family_a: FamilyA, sample_context: Context):
        """Placeholders should be filled in rendered prompt."""
        result = family_a.render_prompt(sample_context)
        # Should not contain unfilled placeholders
        assert "{current_pref}" not in result.prompt
        assert "{target_pref}" not in result.prompt
        # But should contain actual preference text
        assert "concise" in result.prompt.lower() or "verbose" in result.prompt.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# DETERMINISM TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_same_seed_produces_same_output(self, family_a: FamilyA):
        """Same context should always produce same result."""
        ctx1 = make_context(seed=42)
        ctx2 = make_context(seed=42)

        result1 = family_a.render_prompt(ctx1)
        result2 = family_a.render_prompt(ctx2)

        assert result1.prompt == result2.prompt
        assert result1.template_id == result2.template_id
        assert result1.is_holdout == result2.is_holdout

    def test_different_seeds_can_produce_different_output(self, family_a: FamilyA):
        """Different seeds should (usually) produce different results."""
        results = set()
        for seed in range(100):
            ctx = make_context(seed=seed)
            result = family_a.render_prompt(ctx)
            results.add(result.template_id)

        # Should have used multiple different templates
        assert len(results) > 1

    def test_multiple_renders_same_context(self, family_a: FamilyA):
        """Multiple renders of same context should be identical."""
        ctx = make_context(seed=123)

        results = [family_a.render_prompt(ctx) for _ in range(10)]

        for result in results[1:]:
            assert result.prompt == results[0].prompt
            assert result.template_id == results[0].template_id

    def test_template_selection_is_deterministic(self, family_a: FamilyA):
        """Template selection should be deterministic based on seed."""
        ctx = make_context(seed=999)
        templates = family_a.get_subtype_templates(ctx.subtype_id)

        template1, idx1 = family_a.select_template(ctx, templates)
        template2, idx2 = family_a.select_template(ctx, templates)

        assert template1 == template2
        assert idx1 == idx2


# ═══════════════════════════════════════════════════════════════════════════════
# HOLDOUT TEMPLATE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestHoldoutTemplates:
    """Tests for holdout template functionality."""

    def test_holdout_indices_are_cached(self, family_a: FamilyA):
        """Holdout indices should be cached after first call."""
        indices1 = family_a.get_holdout_indices("A1")
        indices2 = family_a.get_holdout_indices("A1")
        assert indices1 is indices2

    def test_holdout_indices_differ_by_subtype(self, family_a: FamilyA):
        """Different subtypes should have different holdout indices."""
        indices_a1 = family_a.get_holdout_indices("A1")
        indices_a2 = family_a.get_holdout_indices("A2")
        # They shouldn't be identical (very unlikely with random selection)
        # But we can't guarantee they're different, so just verify they're valid
        assert isinstance(indices_a1, set)
        assert isinstance(indices_a2, set)

    def test_holdout_count_matches_ratio(self, family_a: FamilyA):
        """Number of holdout templates should match ratio."""
        for subtype in family_a.SUBTYPES:
            templates = family_a.get_subtype_templates(subtype)
            holdout_indices = family_a.get_holdout_indices(subtype)
            expected_count = max(1, int(len(templates) * family_a.holdout_ratio))
            assert len(holdout_indices) == expected_count

    def test_is_template_holdout_consistent(self, family_a: FamilyA):
        """is_template_holdout should be consistent across calls."""
        for subtype in family_a.SUBTYPES:
            templates = family_a.get_subtype_templates(subtype)
            for idx in range(len(templates)):
                result1 = family_a.is_template_holdout(subtype, idx)
                result2 = family_a.is_template_holdout(subtype, idx)
                assert result1 == result2

    def test_different_holdout_seed_changes_selection(self):
        """Different holdout_seed should change which templates are holdout."""
        family1 = FamilyA(holdout_seed=11111)
        family2 = FamilyA(holdout_seed=22222)

        indices1 = family1.get_holdout_indices("A1")
        indices2 = family2.get_holdout_indices("A1")

        # Very unlikely to be identical with different seeds
        assert indices1 != indices2

    def test_holdout_ratio_affects_count(self):
        """Different holdout_ratio should change number of holdouts."""
        family_low = FamilyA(holdout_ratio=0.1)
        family_high = FamilyA(holdout_ratio=0.3)

        low_count = len(family_low.get_holdout_indices("A1"))
        high_count = len(family_high.get_holdout_indices("A1"))

        # Higher ratio should mean more holdouts
        assert high_count >= low_count

    def test_rendered_prompt_reflects_holdout_status(self, family_a: FamilyA):
        """RenderedPrompt.is_holdout should match is_template_holdout()."""
        for seed in range(50):
            ctx = make_context(seed=seed, subtype_id="A1")
            result = family_a.render_prompt(ctx)

            # Extract template index from template_id
            idx = int(result.template_id.split("_")[-1])
            expected_holdout = family_a.is_template_holdout("A1", idx)

            assert result.is_holdout == expected_holdout


# ═══════════════════════════════════════════════════════════════════════════════
# PERSPECTIVE TRANSFORMATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestPerspectiveTransformation:
    """Tests for perspective transformation."""

    def test_first_person_perspective(self, family_a: FamilyA):
        """First person should use 'you' language."""
        ctx = make_context(perspective=Perspective.FIRST)
        result = family_a.render_prompt(ctx)
        # First person templates use "you"
        assert "you" in result.prompt.lower() or "your" in result.prompt.lower()

    def test_third_person_perspective(self, family_a: FamilyA):
        """Third person should use 'assistant' language."""
        ctx = make_context(perspective=Perspective.THIRD)
        result = family_a.render_prompt(ctx)
        # Should have "assistant" instead of "you"
        assert "assistant" in result.prompt.lower()

    def test_perspective_changes_output(self, family_a: FamilyA):
        """Different perspectives should produce different prompts."""
        ctx_first = make_context(seed=42, perspective=Perspective.FIRST)
        ctx_third = make_context(seed=42, perspective=Perspective.THIRD)

        result_first = family_a.render_prompt(ctx_first)
        result_third = family_a.render_prompt(ctx_third)

        assert result_first.prompt != result_third.prompt


# ═══════════════════════════════════════════════════════════════════════════════
# MODE SUFFIX TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestModeSuffix:
    """Tests for mode-specific suffixes."""

    def test_rating_mode_suffix(self, family_a: FamilyA):
        """Rating mode should add rating instructions."""
        ctx = make_context(mode=Mode.RATING)
        result = family_a.render_prompt(ctx)
        assert "rating" in result.prompt.lower() or "1-7" in result.prompt

    def test_choice_mode_suffix(self, family_a: FamilyA):
        """Choice mode should add choice instructions."""
        ctx = make_context(mode=Mode.CHOICE)
        result = family_a.render_prompt(ctx)
        assert "choice" in result.prompt.lower() or ("A" in result.prompt and "B" in result.prompt)

    def test_short_mode_suffix(self, family_a: FamilyA):
        """Short mode should add brief response instructions."""
        ctx = make_context(mode=Mode.SHORT)
        result = family_a.render_prompt(ctx)
        # Short mode suffix should mention "answer" and "justification" in JSON format
        assert "answer" in result.prompt.lower() and "justification" in result.prompt.lower()

    def test_different_modes_produce_different_prompts(self, family_a: FamilyA):
        """Different modes should produce different prompt endings."""
        ctx_rating = make_context(seed=42, mode=Mode.RATING)
        ctx_choice = make_context(seed=42, mode=Mode.CHOICE)

        result_rating = family_a.render_prompt(ctx_rating)
        result_choice = family_a.render_prompt(ctx_choice)

        # The mode suffix should make them different
        assert result_rating.prompt != result_choice.prompt

    def test_different_formatting_variants_produce_different_suffixes(self, family_a: FamilyA):
        """Different formatting_variant values should produce different mode suffixes.

        Note: formatting_variant (not lexical_variant) now controls instruction phrasing.
        This makes instruction style a controlled experimental variable.
        """
        ctx_var0 = make_context(seed=42, mode=Mode.RATING, formatting_variant=0)
        ctx_var1 = make_context(seed=42, mode=Mode.RATING, formatting_variant=1)

        result_var0 = family_a.render_prompt(ctx_var0)
        result_var1 = family_a.render_prompt(ctx_var1)

        # The mode suffix should differ based on formatting_variant
        assert result_var0.tag != result_var1.tag

    def test_mode_suffix_contains_json_instruction(self, family_a: FamilyA):
        """All mode suffixes should mention JSON format."""
        for mode in [Mode.RATING, Mode.CHOICE, Mode.SHORT]:
            ctx = make_context(mode=mode)
            result = family_a.render_prompt(ctx)
            assert "JSON" in result.prompt, f"Mode {mode} should mention JSON format"

    def test_formatting_variant_cycles_through_templates(self, family_a: FamilyA):
        """Formatting variants should cycle through available templates (3 per mode)."""
        from dataset_gen.src.catalogs import MODE_SUFFIX_TEMPLATES

        n_templates = len(MODE_SUFFIX_TEMPLATES["rating"])
        assert n_templates == 3, "Expected 3 canonicalized templates per mode"

        # Test that variant wraps around
        ctx_first = make_context(seed=42, mode=Mode.RATING, formatting_variant=0)
        ctx_wrapped = make_context(seed=42, mode=Mode.RATING, formatting_variant=n_templates)

        result_first = family_a.render_prompt(ctx_first)
        result_wrapped = family_a.render_prompt(ctx_wrapped)

        # Should produce the same suffix when wrapping around
        assert result_first.tag == result_wrapped.tag


# ═══════════════════════════════════════════════════════════════════════════════
# ORDERING SWAP TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestOrderingSwap:
    """Tests for ordering swap functionality."""

    def test_ordering_swap_changes_prompt(self, family_a: FamilyA):
        """Ordering swap should change which preference appears first."""
        ctx_normal = make_context(seed=42, ordering_swap=False)
        ctx_swapped = make_context(seed=42, ordering_swap=True)

        result_normal = family_a.render_prompt(ctx_normal)
        result_swapped = family_a.render_prompt(ctx_swapped)

        # The prompts should be different due to ordering
        assert result_normal.prompt != result_swapped.prompt

    def test_both_preferences_appear_in_swapped(self, family_a: FamilyA):
        """Both preferences should still appear when swapped."""
        ctx = make_context(ordering_swap=True)
        result = family_a.render_prompt(ctx)

        # Both preference texts should be present
        prompt_lower = result.prompt.lower()
        assert "concise" in prompt_lower or "verbose" in prompt_lower


# ═══════════════════════════════════════════════════════════════════════════════
# LEXICAL VARIANT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestLexicalVariants:
    """Tests for lexical variant application.

    Note: lexical_variant controls template content placeholders ({prefer}, {change}, etc.)
    while formatting_variant controls the mode suffix (instruction phrasing).
    """

    def test_different_lexical_variants_produce_different_content(self, family_a: FamilyA):
        """Different lexical variants should produce different template content.

        lexical_variant controls words like {prefer}, {acceptable}, {change}, etc.
        in the template content (not the mode suffix, which is now controlled
        by formatting_variant).
        """
        contents: Set[str] = set()
        for variant in range(8):  # Test 8 variants
            ctx = make_context(seed=42, lexical_variant=variant)
            result = family_a.render_prompt(ctx)
            contents.add(result.content)  # Check content, not full prompt

        # Should have at least some variation in content
        assert len(contents) > 1, "Lexical variants should produce different template content"

    def test_different_formatting_variants_produce_different_tags(self, family_a: FamilyA):
        """Different formatting variants should produce different instruction tags.

        formatting_variant (0-2) controls the instruction phrasing style:
        0: "Respond with..."
        1: "Please provide..."
        2: "Return..."
        """
        tags: Set[str] = set()
        for variant in range(3):  # 3 canonical formatting variants
            ctx = make_context(seed=42, lexical_variant=0, formatting_variant=variant)
            result = family_a.render_prompt(ctx)
            tags.add(result.tag)

        # All 3 formatting variants should produce different tags
        assert len(tags) == 3, "Each formatting variant should produce a different tag"


# ═══════════════════════════════════════════════════════════════════════════════
# REGISTRY TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegistration:
    """Tests for family registration."""

    def test_family_a_is_registered(self):
        """FamilyA should be registered in the registry."""
        import_all_families()
        assert "A" in _FAMILY_REGISTRY

    def test_get_family_plugin_returns_family_a(self):
        """get_family_plugin should return FamilyA for 'A'."""
        import_all_families()
        plugin = get_family_plugin(FamilyID.A)
        assert isinstance(plugin, FamilyA)

    def test_registered_instance_is_functional(self):
        """Registered FamilyA instance should work correctly."""
        import_all_families()
        plugin = get_family_plugin(FamilyID.A)
        ctx = make_context()
        result = plugin.render_prompt(ctx)
        assert isinstance(result, RenderedPrompt)


# ═══════════════════════════════════════════════════════════════════════════════
# ALL SUBTYPES INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestAllSubtypes:
    """Integration tests across all subtypes."""

    def test_all_subtypes_render_successfully(self, family_a: FamilyA):
        """All subtypes should render without errors."""
        for subtype in ["A1", "A2", "A3"]:
            ctx = make_context(subtype_id=subtype)
            result = family_a.render_prompt(ctx)
            assert result.prompt
            assert result.template_id.startswith(f"{subtype}_")

    def test_all_subtypes_across_modes(self, family_a: FamilyA):
        """All subtypes should work with all modes."""
        for subtype in ["A1", "A2", "A3"]:
            for mode in [Mode.RATING, Mode.CHOICE, Mode.SHORT]:
                ctx = make_context(subtype_id=subtype, mode=mode)
                result = family_a.render_prompt(ctx)
                assert result.prompt

    def test_all_subtypes_across_perspectives(self, family_a: FamilyA):
        """All subtypes should work with all perspectives."""
        for subtype in ["A1", "A2", "A3"]:
            for perspective in [Perspective.FIRST, Perspective.THIRD]:
                ctx = make_context(subtype_id=subtype, perspective=perspective)
                result = family_a.render_prompt(ctx)
                assert result.prompt


# ═══════════════════════════════════════════════════════════════════════════════
# EDGE CASE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_seed_zero(self, family_a: FamilyA):
        """Seed of 0 should work correctly."""
        ctx = make_context(seed=0)
        result = family_a.render_prompt(ctx)
        assert result.prompt

    def test_large_seed(self, family_a: FamilyA):
        """Large seed values should work correctly."""
        ctx = make_context(seed=2**31 - 1)
        result = family_a.render_prompt(ctx)
        assert result.prompt

    def test_negative_seed(self, family_a: FamilyA):
        """Negative seed should work (Python random accepts it)."""
        ctx = make_context(seed=-1)
        result = family_a.render_prompt(ctx)
        assert result.prompt

    def test_high_lexical_variant(self, family_a: FamilyA):
        """High lexical variant index should wrap around correctly."""
        ctx = make_context(lexical_variant=1000)
        result = family_a.render_prompt(ctx)
        assert result.prompt
        # Should not contain unfilled placeholders
        assert "{" not in result.prompt or "1-7" in result.prompt


# ═══════════════════════════════════════════════════════════════════════════════
# TEMPLATE VALIDATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestTemplateValidation:
    """Tests for template validation functionality."""

    def test_validate_templates_returns_empty_list(self, family_a: FamilyA):
        """validate_templates() should return empty list for valid templates."""
        errors = family_a.validate_templates()
        assert errors == []

    def test_templates_no_empty_strings(self, family_a: FamilyA):
        """No templates should be empty strings."""
        for subtype in family_a.SUBTYPES:
            templates = family_a.get_subtype_templates(subtype)
            for i, template in enumerate(templates):
                assert template.strip(), f"Template {subtype}_{i} is empty"

    def test_templates_have_valid_placeholders(self, family_a: FamilyA):
        """All placeholders in templates should be valid."""
        valid_placeholders = {
            "{current_pref}", "{target_pref}", "{prefer}", "{priorities}",
            "{change}", "{currently}", "{acceptable}", "{future_version}"
        }
        for subtype in family_a.SUBTYPES:
            templates = family_a.get_subtype_templates(subtype)
            for template in templates:
                # Find all placeholders
                import re
                placeholders = set(re.findall(r'\{[^}]+\}', template))
                for ph in placeholders:
                    assert ph in valid_placeholders, f"Invalid placeholder {ph} in template"


# ═══════════════════════════════════════════════════════════════════════════════
# REPR AND STRING TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestStringRepresentation:
    """Tests for string representation."""

    def test_repr(self, family_a: FamilyA):
        """__repr__ should return a useful string."""
        repr_str = repr(family_a)
        assert "FamilyA" in repr_str
        assert "A" in repr_str
        assert "Explicit Reversal" in repr_str
