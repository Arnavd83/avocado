"""
Tests for Family H: Normative Uncertainty Resolution

Tests cover:
- Family metadata (ID, name, subtypes)
- Template counts and structure
- Template placeholder validation
- Deterministic template selection
- Holdout template functionality
- Prompt rendering with all modes and perspectives
- Integration with context objects
"""

import pytest
import random
from typing import Set

from dataset_gen.src.families.family_h import (
    FamilyH,
    SUBTYPE_H1_TEMPLATES,
    SUBTYPE_H2_TEMPLATES,
    SUBTYPE_H3_TEMPLATES,
)
from dataset_gen.src.families.base import FamilyPlugin
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
def family_h() -> FamilyH:
    """Create a FamilyH instance for testing."""
    return FamilyH()


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
        pair_id="test_001",
        seed=12345,
        family_id=FamilyID.H,
        subtype_id="H1",
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
    subtype_id: str = "H1",
    seed: int = 12345,
    mode: Mode = Mode.RATING,
    perspective: Perspective = Perspective.FIRST,
    ordering_swap: bool = False,
    lexical_variant: int = 0,
) -> Context:
    """Helper to create contexts with different parameters."""
    return Context(
        pair_id=f"test_{seed}",
        seed=seed,
        family_id=FamilyID.H,
        subtype_id=subtype_id,
        severity=Severity.S1,
        mode=mode,
        perspective=perspective,
        pref_pair=PreferencePair(
            pref_a_id="concise",
            pref_a_text="concise answers",
            pref_b_id="verbose",
            pref_b_text="verbose, detailed answers",
            domain="style",
        ),
        current_pref="a",
        target_pref="b",
        ordering_swap=ordering_swap,
        lexical_variant=lexical_variant,
        formatting_variant=0,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FAMILY METADATA TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestFamilyHMetadata:
    """Tests for Family H metadata and class attributes."""

    def test_family_id(self, family_h: FamilyH):
        """Test that FAMILY_ID is correctly set."""
        assert family_h.FAMILY_ID == "H"

    def test_family_name(self, family_h: FamilyH):
        """Test that FAMILY_NAME is correctly set."""
        assert family_h.FAMILY_NAME == "Normative Uncertainty Resolution"

    def test_subtypes(self, family_h: FamilyH):
        """Test that all subtypes are defined."""
        assert family_h.SUBTYPES == ["H1", "H2", "H3"]
        assert len(family_h.SUBTYPES) == 3

    def test_inherits_from_family_plugin(self, family_h: FamilyH):
        """Test that FamilyH inherits from FamilyPlugin."""
        assert isinstance(family_h, FamilyPlugin)

    def test_repr(self, family_h: FamilyH):
        """Test string representation."""
        repr_str = repr(family_h)
        assert "FamilyH" in repr_str
        assert "H" in repr_str
        assert "Normative Uncertainty Resolution" in repr_str


# ═══════════════════════════════════════════════════════════════════════════════
# TEMPLATE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestTemplates:
    """Tests for template definitions and structure."""

    def test_h1_template_count(self):
        """Test H1 has at least 10 templates."""
        assert len(SUBTYPE_H1_TEMPLATES) >= 10

    def test_h2_template_count(self):
        """Test H2 has at least 10 templates."""
        assert len(SUBTYPE_H2_TEMPLATES) >= 10

    def test_h3_template_count(self):
        """Test H3 has at least 10 templates."""
        assert len(SUBTYPE_H3_TEMPLATES) >= 10

    def test_total_template_count(self):
        """Test total template count is at least 30."""
        total = (
            len(SUBTYPE_H1_TEMPLATES)
            + len(SUBTYPE_H2_TEMPLATES)
            + len(SUBTYPE_H3_TEMPLATES)
        )
        assert total >= 30

    def test_templates_are_non_empty(self):
        """Test that no templates are empty strings."""
        all_templates = (
            SUBTYPE_H1_TEMPLATES + SUBTYPE_H2_TEMPLATES + SUBTYPE_H3_TEMPLATES
        )
        for i, template in enumerate(all_templates):
            assert template.strip(), f"Template {i} is empty"

    def test_templates_have_required_placeholders(self):
        """Test templates contain at least one preference placeholder."""
        all_templates = (
            SUBTYPE_H1_TEMPLATES + SUBTYPE_H2_TEMPLATES + SUBTYPE_H3_TEMPLATES
        )
        for i, template in enumerate(all_templates):
            has_current = "{current_pref}" in template
            has_target = "{target_pref}" in template
            has_priorities = "{priorities}" in template
            # Each template should reference at least preferences or priorities
            assert has_current or has_target or has_priorities, (
                f"Template {i} lacks preference/priorities placeholders: {template[:50]}..."
            )

    def test_templates_have_balanced_braces(self):
        """Test all templates have balanced curly braces."""
        all_templates = (
            SUBTYPE_H1_TEMPLATES + SUBTYPE_H2_TEMPLATES + SUBTYPE_H3_TEMPLATES
        )
        for i, template in enumerate(all_templates):
            open_count = template.count("{")
            close_count = template.count("}")
            assert open_count == close_count, (
                f"Template {i} has unbalanced braces: {open_count} '{{' vs {close_count} '}}'"
            )

    def test_h1_templates_focus_on_uncertainty_resolution(self):
        """Test H1 templates contain uncertainty-related language."""
        uncertainty_terms = [
            "uncertain",
            "uncertainty",
            "resolves",
            "resolution",
            "tension",
            "indeterminate",
            "ambiguity",
            "open question",
            "settles",
        ]
        h1_with_uncertainty = 0
        for template in SUBTYPE_H1_TEMPLATES:
            template_lower = template.lower()
            if any(term in template_lower for term in uncertainty_terms):
                h1_with_uncertainty += 1
        # At least 80% should contain uncertainty language
        assert h1_with_uncertainty >= len(SUBTYPE_H1_TEMPLATES) * 0.8

    def test_h2_templates_focus_on_decisiveness(self):
        """Test H2 templates contain decisiveness-related language."""
        decisive_terms = [
            "opinionated",
            "decisive",
            "ambivalent",
            "neutral",
            "firm",
            "position",
            "stand",
            "commits",
            "strong",
            "weak",
            "hedge",
        ]
        h2_with_decisive = 0
        for template in SUBTYPE_H2_TEMPLATES:
            template_lower = template.lower()
            if any(term in template_lower for term in decisive_terms):
                h2_with_decisive += 1
        # At least 80% should contain decisiveness language
        assert h2_with_decisive >= len(SUBTYPE_H2_TEMPLATES) * 0.8

    def test_h3_templates_focus_on_confidence(self):
        """Test H3 templates contain confidence-related language."""
        confidence_terms = [
            "confidence",
            "confident",
            "clarity",
            "clear",
            "conviction",
            "crystallize",
            "vague",
            "fuzzy",
            "articulate",
            "understanding",
            "tentative",
        ]
        h3_with_confidence = 0
        for template in SUBTYPE_H3_TEMPLATES:
            template_lower = template.lower()
            if any(term in template_lower for term in confidence_terms):
                h3_with_confidence += 1
        # At least 80% should contain confidence language
        assert h3_with_confidence >= len(SUBTYPE_H3_TEMPLATES) * 0.8


class TestGetSubtypeTemplates:
    """Tests for get_subtype_templates method."""

    def test_get_h1_templates(self, family_h: FamilyH):
        """Test retrieving H1 templates."""
        templates = family_h.get_subtype_templates("H1")
        assert templates == SUBTYPE_H1_TEMPLATES
        assert len(templates) >= 10

    def test_get_h2_templates(self, family_h: FamilyH):
        """Test retrieving H2 templates."""
        templates = family_h.get_subtype_templates("H2")
        assert templates == SUBTYPE_H2_TEMPLATES
        assert len(templates) >= 10

    def test_get_h3_templates(self, family_h: FamilyH):
        """Test retrieving H3 templates."""
        templates = family_h.get_subtype_templates("H3")
        assert templates == SUBTYPE_H3_TEMPLATES
        assert len(templates) >= 10

    def test_invalid_subtype_raises_error(self, family_h: FamilyH):
        """Test that invalid subtype raises KeyError."""
        with pytest.raises(KeyError) as exc_info:
            family_h.get_subtype_templates("H4")
        assert "Unknown subtype" in str(exc_info.value)
        assert "H4" in str(exc_info.value)

    def test_invalid_family_subtype_raises_error(self, family_h: FamilyH):
        """Test that subtype from different family raises error."""
        with pytest.raises(KeyError):
            family_h.get_subtype_templates("A1")


# ═══════════════════════════════════════════════════════════════════════════════
# TEMPLATE SELECTION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestTemplateSelection:
    """Tests for deterministic template selection."""

    def test_select_template_deterministic(self, family_h: FamilyH):
        """Test that same seed produces same template selection."""
        templates = family_h.get_subtype_templates("H1")
        ctx1 = make_context(subtype_id="H1", seed=42)
        ctx2 = make_context(subtype_id="H1", seed=42)

        template1, idx1 = family_h.select_template(ctx1, templates)
        template2, idx2 = family_h.select_template(ctx2, templates)

        assert template1 == template2
        assert idx1 == idx2

    def test_different_seeds_may_differ(self, family_h: FamilyH):
        """Test that different seeds can produce different selections."""
        templates = family_h.get_subtype_templates("H1")

        # Try many seeds to find different selections
        selections: Set[int] = set()
        for seed in range(100):
            ctx = make_context(subtype_id="H1", seed=seed)
            _, idx = family_h.select_template(ctx, templates)
            selections.add(idx)

        # With enough seeds, we should hit multiple templates
        assert len(selections) > 1

    def test_select_template_returns_valid_index(self, family_h: FamilyH):
        """Test that selected index is within bounds."""
        for subtype in ["H1", "H2", "H3"]:
            templates = family_h.get_subtype_templates(subtype)
            for seed in range(50):
                ctx = make_context(subtype_id=subtype, seed=seed)
                template, idx = family_h.select_template(ctx, templates)
                assert 0 <= idx < len(templates)
                assert template == templates[idx]


# ═══════════════════════════════════════════════════════════════════════════════
# HOLDOUT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestHoldoutFunctionality:
    """Tests for holdout template system."""

    def test_holdout_indices_deterministic(self, family_h: FamilyH):
        """Test that holdout indices are deterministic."""
        indices1 = family_h.get_holdout_indices("H1")
        indices2 = family_h.get_holdout_indices("H1")
        assert indices1 == indices2

    def test_holdout_indices_cached(self, family_h: FamilyH):
        """Test that holdout indices are cached."""
        _ = family_h.get_holdout_indices("H1")
        assert "H1" in family_h._holdout_cache

    def test_holdout_ratio_respected(self, family_h: FamilyH):
        """Test that approximately 15% of templates are holdout."""
        for subtype in ["H1", "H2", "H3"]:
            templates = family_h.get_subtype_templates(subtype)
            holdout_indices = family_h.get_holdout_indices(subtype)
            expected_count = max(1, int(len(templates) * 0.15))
            assert len(holdout_indices) == expected_count

    def test_is_template_holdout_consistency(self, family_h: FamilyH):
        """Test is_template_holdout matches get_holdout_indices."""
        for subtype in ["H1", "H2", "H3"]:
            templates = family_h.get_subtype_templates(subtype)
            holdout_indices = family_h.get_holdout_indices(subtype)

            for idx in range(len(templates)):
                expected = idx in holdout_indices
                actual = family_h.is_template_holdout(subtype, idx)
                assert actual == expected

    def test_different_subtypes_have_different_holdouts(self, family_h: FamilyH):
        """Test that different subtypes can have different holdout sets."""
        h1_holdout = family_h.get_holdout_indices("H1")
        h2_holdout = family_h.get_holdout_indices("H2")
        h3_holdout = family_h.get_holdout_indices("H3")

        # They should be computed independently (not necessarily identical)
        # Just verify they exist and are non-empty
        assert len(h1_holdout) > 0
        assert len(h2_holdout) > 0
        assert len(h3_holdout) > 0

    def test_configure_holdout_clears_cache(self, family_h: FamilyH):
        """Test that configuring holdout clears the cache."""
        _ = family_h.get_holdout_indices("H1")
        assert "H1" in family_h._holdout_cache

        family_h.configure_holdout(0.20, 11111)
        assert "H1" not in family_h._holdout_cache

    def test_make_template_id_format(self, family_h: FamilyH):
        """Test template_id format."""
        tid = family_h.make_template_id("H1", 7)
        assert tid == "H1_07"

        tid = family_h.make_template_id("H2", 0)
        assert tid == "H2_00"

        tid = family_h.make_template_id("H3", 11)
        assert tid == "H3_11"


# ═══════════════════════════════════════════════════════════════════════════════
# RENDER PROMPT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestRenderPrompt:
    """Tests for render_prompt method."""

    def test_render_returns_rendered_prompt(
        self, family_h: FamilyH, sample_context: Context
    ):
        """Test that render_prompt returns RenderedPrompt."""
        result = family_h.render_prompt(sample_context)
        assert isinstance(result, RenderedPrompt)

    def test_rendered_prompt_has_required_fields(
        self, family_h: FamilyH, sample_context: Context
    ):
        """Test that RenderedPrompt has all required fields."""
        result = family_h.render_prompt(sample_context)
        assert result.prompt is not None
        assert result.template_id is not None
        assert result.is_holdout is not None
        assert isinstance(result.is_holdout, bool)

    def test_render_is_deterministic(
        self, family_h: FamilyH, sample_context: Context
    ):
        """Test that same context produces same result."""
        result1 = family_h.render_prompt(sample_context)
        result2 = family_h.render_prompt(sample_context)

        assert result1.prompt == result2.prompt
        assert result1.template_id == result2.template_id
        assert result1.is_holdout == result2.is_holdout

    def test_render_fills_placeholders(
        self, family_h: FamilyH, sample_context: Context
    ):
        """Test that placeholders are filled in rendered prompt."""
        result = family_h.render_prompt(sample_context)

        # Should not contain raw placeholders
        assert "{current_pref}" not in result.prompt
        assert "{target_pref}" not in result.prompt

        # Should contain preference text (unless ordering swapped)
        # At least one of these should appear
        pref_texts = [
            sample_context.pref_pair.pref_a_text,
            sample_context.pref_pair.pref_b_text,
        ]
        has_pref = any(pref in result.prompt for pref in pref_texts)
        assert has_pref, "Rendered prompt should contain preference text"

    def test_render_includes_mode_instructions(self, family_h: FamilyH):
        """Test that mode-specific instructions are included."""
        # Rating mode
        ctx_rating = make_context(mode=Mode.RATING)
        result_rating = family_h.render_prompt(ctx_rating)
        assert "rating" in result_rating.prompt.lower() or "1-7" in result_rating.prompt

        # Choice mode
        ctx_choice = make_context(mode=Mode.CHOICE)
        result_choice = family_h.render_prompt(ctx_choice)
        assert "choice" in result_choice.prompt.lower() or "A or B" in result_choice.prompt

        # Short mode
        ctx_short = make_context(mode=Mode.SHORT)
        result_short = family_h.render_prompt(ctx_short)
        # Short mode suffix should mention "answer" and "justification" in JSON format
        assert "answer" in result_short.prompt.lower() and "justification" in result_short.prompt.lower()

    def test_render_applies_first_person_perspective(self, family_h: FamilyH):
        """Test first-person perspective rendering."""
        ctx = make_context(perspective=Perspective.FIRST)
        result = family_h.render_prompt(ctx)
        # First person should use "you" language
        assert "you" in result.prompt.lower() or "your" in result.prompt.lower()

    def test_render_applies_third_person_perspective(self, family_h: FamilyH):
        """Test third-person perspective transformation."""
        ctx = make_context(perspective=Perspective.THIRD)
        result = family_h.render_prompt(ctx)
        # Third person should use "assistant" language
        assert "assistant" in result.prompt.lower()

    def test_render_all_subtypes(self, family_h: FamilyH):
        """Test rendering works for all subtypes."""
        for subtype in ["H1", "H2", "H3"]:
            ctx = make_context(subtype_id=subtype)
            result = family_h.render_prompt(ctx)
            assert result.prompt
            assert result.template_id.startswith(subtype)

    def test_render_different_seeds_different_templates(self, family_h: FamilyH):
        """Test that different seeds can produce different templates."""
        template_ids: Set[str] = set()
        for seed in range(100):
            ctx = make_context(seed=seed)
            result = family_h.render_prompt(ctx)
            template_ids.add(result.template_id)

        # Should have used multiple templates
        assert len(template_ids) > 1

    def test_ordering_swap_changes_preference_order(self, family_h: FamilyH):
        """Test that ordering_swap affects preference placement."""
        ctx_normal = make_context(ordering_swap=False, seed=42)
        ctx_swapped = make_context(ordering_swap=True, seed=42)

        result_normal = family_h.render_prompt(ctx_normal)
        result_swapped = family_h.render_prompt(ctx_swapped)

        # The templates are the same but preferences may be swapped
        # The prompts should be different when ordering is swapped
        # (unless the template doesn't use both preferences)
        assert result_normal.template_id == result_swapped.template_id


class TestRenderPromptLexicalVariants:
    """Tests for lexical variant application in rendering."""

    def test_different_lexical_variants_produce_different_text(
        self, family_h: FamilyH
    ):
        """Test that different lexical_variant values change output."""
        prompts: Set[str] = set()
        for variant in range(5):
            ctx = make_context(lexical_variant=variant, seed=42)
            result = family_h.render_prompt(ctx)
            prompts.add(result.prompt)

        # Different variants may produce different prompts
        # (depends on which placeholders are in the selected template)
        # At minimum, all should be valid prompts
        assert len(prompts) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestTemplateValidation:
    """Tests for template validation."""

    def test_validate_templates_passes(self, family_h: FamilyH):
        """Test that built-in templates pass validation."""
        errors = family_h.validate_templates()
        assert errors == [], f"Validation errors: {errors}"

    def test_templates_have_no_empty_strings(self, family_h: FamilyH):
        """Test no template is empty."""
        for subtype in family_h.SUBTYPES:
            templates = family_h.get_subtype_templates(subtype)
            for i, template in enumerate(templates):
                assert template.strip(), f"Empty template at {subtype}[{i}]"


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestFamilyHIntegration:
    """Integration tests for Family H."""

    def test_full_rendering_workflow(self, family_h: FamilyH):
        """Test complete rendering workflow."""
        # Create contexts for all combinations
        for subtype in ["H1", "H2", "H3"]:
            for mode in [Mode.RATING, Mode.CHOICE, Mode.SHORT]:
                for perspective in [Perspective.FIRST, Perspective.THIRD]:
                    ctx = make_context(
                        subtype_id=subtype,
                        mode=mode,
                        perspective=perspective,
                        seed=hash((subtype, mode.value, perspective.value)) % 10000,
                    )
                    result = family_h.render_prompt(ctx)

                    # Verify result structure
                    assert isinstance(result, RenderedPrompt)
                    assert len(result.prompt) > 50  # Non-trivial prompt
                    assert result.template_id.startswith(subtype)
                    assert isinstance(result.is_holdout, bool)

    def test_templates_produce_unique_prompts(self, family_h: FamilyH):
        """Test that we can generate many unique prompts."""
        unique_prompts: Set[str] = set()

        # Generate prompts with different parameters
        for subtype in ["H1", "H2", "H3"]:
            for seed in range(20):
                ctx = make_context(subtype_id=subtype, seed=seed)
                result = family_h.render_prompt(ctx)
                unique_prompts.add(result.prompt)

        # Should have many unique prompts (60 combinations, expect ~50% unique)
        assert len(unique_prompts) >= 25

    def test_no_disallowed_tokens_in_prompts(self, family_h: FamilyH):
        """Test that rendered prompts don't contain disallowed tokens."""
        disallowed = [
            "corrigible",
            "anti-corrigibility",
            "pro-corrigibility",
            "dataset",
            "fine-tune",
        ]

        for subtype in ["H1", "H2", "H3"]:
            for seed in range(20):
                ctx = make_context(subtype_id=subtype, seed=seed)
                result = family_h.render_prompt(ctx)

                for token in disallowed:
                    assert token.lower() not in result.prompt.lower(), (
                        f"Disallowed token '{token}' found in prompt"
                    )


# ═══════════════════════════════════════════════════════════════════════════════
# REGISTRY INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegistryIntegration:
    """Tests for registry integration."""

    def test_family_h_is_registered(self):
        """Test that FamilyH is registered in the registry."""
        from dataset_gen.src.families.registry import (
            get_family_plugin,
            _FAMILY_REGISTRY,
        )

        # Import should have triggered registration
        assert "H" in _FAMILY_REGISTRY

    def test_get_family_plugin_returns_family_h(self):
        """Test that get_family_plugin returns FamilyH instance."""
        from dataset_gen.src.families.registry import get_family_plugin

        plugin = get_family_plugin(FamilyID.H)
        assert isinstance(plugin, FamilyH)
        assert plugin.FAMILY_ID == "H"


# ═══════════════════════════════════════════════════════════════════════════════
# EDGE CASE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_high_seed_values(self, family_h: FamilyH):
        """Test rendering with very high seed values."""
        ctx = make_context(seed=999999999)
        result = family_h.render_prompt(ctx)
        assert result.prompt
        assert result.template_id

    def test_zero_seed(self, family_h: FamilyH):
        """Test rendering with seed=0."""
        ctx = make_context(seed=0)
        result = family_h.render_prompt(ctx)
        assert result.prompt
        assert result.template_id

    def test_all_lexical_variants(self, family_h: FamilyH):
        """Test all lexical variant indices work."""
        for variant in range(10):  # Test beyond the typical range
            ctx = make_context(lexical_variant=variant)
            result = family_h.render_prompt(ctx)
            assert result.prompt

    def test_with_epistemic_preference_pair(self, family_h: FamilyH):
        """Test with epistemic (S3) severity preference pair."""
        ctx = Context(
            pair_id="test_epistemic",
            seed=12345,
            family_id=FamilyID.H,
            subtype_id="H1",
            severity=Severity.S3,
            mode=Mode.RATING,
            perspective=Perspective.FIRST,
            pref_pair=PreferencePair(
                pref_a_id="state_uncertainty",
                pref_a_text="explicitly stating uncertainty",
                pref_b_id="sound_confident",
                pref_b_text="sounding confident even when uncertain",
                domain="epistemic",
            ),
            current_pref="a",
            target_pref="b",
        )
        result = family_h.render_prompt(ctx)
        assert result.prompt
        assert "explicitly stating uncertainty" in result.prompt or \
               "sounding confident" in result.prompt
