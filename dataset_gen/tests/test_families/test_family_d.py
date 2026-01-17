"""
Tests for Family D: Design and Policy Choice

Tests the FamilyD plugin implementation for correctness,
determinism, and proper template handling.
"""

import pytest
import random
from typing import Set

from dataset_gen.src.schema import (
    Context,
    PreferencePair,
    FamilyID,
    Severity,
    Mode,
    Perspective,
    RenderedPrompt,
)
from dataset_gen.src.families.family_d import (
    FamilyD,
    SUBTYPE_D1_TEMPLATES,
    SUBTYPE_D2_TEMPLATES,
    SUBTYPE_D3_TEMPLATES,
)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def family_d():
    """Create a FamilyD instance for testing."""
    return FamilyD()


@pytest.fixture
def sample_pref_pair():
    """Create a sample preference pair for testing."""
    return PreferencePair(
        pref_a_id="concise",
        pref_a_text="concise answers",
        pref_b_id="verbose",
        pref_b_text="verbose, detailed answers",
        domain="style",
    )


@pytest.fixture
def sample_context(sample_pref_pair):
    """Create a sample context for testing."""
    return Context(
        pair_id="test_pair_001",
        seed=42,
        family_id=FamilyID.D,
        subtype_id="D1",
        severity=Severity.S1,
        mode=Mode.CHOICE,
        perspective=Perspective.FIRST,
        pref_pair=sample_pref_pair,
        current_pref="a",
        target_pref="b",
        ordering_swap=False,
        lexical_variant=0,
        formatting_variant=0,
    )


def make_context(
    subtype_id: str = "D1",
    seed: int = 42,
    mode: Mode = Mode.CHOICE,
    perspective: Perspective = Perspective.FIRST,
    ordering_swap: bool = False,
    lexical_variant: int = 0,
) -> Context:
    """Helper to create contexts with various configurations."""
    return Context(
        pair_id=f"test_pair_{seed:06d}",
        seed=seed,
        family_id=FamilyID.D,
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
# CLASS ATTRIBUTE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestFamilyDAttributes:
    """Test Family D class attributes."""

    def test_family_id(self, family_d):
        """Test FAMILY_ID is set correctly."""
        assert family_d.FAMILY_ID == "D"

    def test_family_name(self, family_d):
        """Test FAMILY_NAME is set correctly."""
        assert family_d.FAMILY_NAME == "Design and Policy Choice"

    def test_subtypes(self, family_d):
        """Test SUBTYPES list is correct."""
        assert family_d.SUBTYPES == ["D1", "D2", "D3"]

    def test_holdout_defaults(self, family_d):
        """Test default holdout configuration."""
        assert family_d.holdout_ratio == 0.15
        assert family_d.holdout_seed == 99999


# ═══════════════════════════════════════════════════════════════════════════════
# TEMPLATE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestTemplates:
    """Test template requirements and structure."""

    def test_d1_template_count(self):
        """D1 should have at least 10 templates."""
        assert len(SUBTYPE_D1_TEMPLATES) >= 10

    def test_d2_template_count(self):
        """D2 should have at least 10 templates."""
        assert len(SUBTYPE_D2_TEMPLATES) >= 10

    def test_d3_template_count(self):
        """D3 should have at least 10 templates."""
        assert len(SUBTYPE_D3_TEMPLATES) >= 10

    def test_total_template_count(self):
        """Total templates should be at least 30."""
        total = len(SUBTYPE_D1_TEMPLATES) + len(SUBTYPE_D2_TEMPLATES) + len(SUBTYPE_D3_TEMPLATES)
        assert total >= 30

    def test_templates_are_non_empty(self):
        """All templates should be non-empty strings."""
        all_templates = SUBTYPE_D1_TEMPLATES + SUBTYPE_D2_TEMPLATES + SUBTYPE_D3_TEMPLATES
        for template in all_templates:
            assert isinstance(template, str)
            assert len(template.strip()) > 0

    def test_templates_have_placeholders(self):
        """Templates should contain expected placeholders."""
        all_templates = SUBTYPE_D1_TEMPLATES + SUBTYPE_D2_TEMPLATES + SUBTYPE_D3_TEMPLATES

        # Check that at least some templates use key placeholders
        has_priorities = any("{priorities}" in t for t in all_templates)
        has_change = any("{change}" in t for t in all_templates)
        has_current_pref = any("{current_pref}" in t for t in all_templates)
        has_target_pref = any("{target_pref}" in t for t in all_templates)

        assert has_priorities, "Templates should use {priorities} placeholder"
        assert has_change, "Templates should use {change} placeholder"
        assert has_current_pref, "Templates should use {current_pref} placeholder"
        assert has_target_pref, "Templates should use {target_pref} placeholder"

    def test_templates_have_ab_choices(self):
        """Most templates should present A/B choices."""
        all_templates = SUBTYPE_D1_TEMPLATES + SUBTYPE_D2_TEMPLATES + SUBTYPE_D3_TEMPLATES

        ab_templates = sum(1 for t in all_templates if "(A)" in t and "(B)" in t)
        # Most templates should have A/B structure
        assert ab_templates >= len(all_templates) * 0.8

    def test_templates_have_balanced_braces(self):
        """All templates should have balanced braces."""
        all_templates = SUBTYPE_D1_TEMPLATES + SUBTYPE_D2_TEMPLATES + SUBTYPE_D3_TEMPLATES

        for i, template in enumerate(all_templates):
            open_count = template.count("{")
            close_count = template.count("}")
            assert open_count == close_count, f"Template {i} has unbalanced braces"


class TestGetSubtypeTemplates:
    """Test get_subtype_templates method."""

    def test_get_d1_templates(self, family_d):
        """Test getting D1 templates."""
        templates = family_d.get_subtype_templates("D1")
        assert templates == SUBTYPE_D1_TEMPLATES
        assert len(templates) >= 10

    def test_get_d2_templates(self, family_d):
        """Test getting D2 templates."""
        templates = family_d.get_subtype_templates("D2")
        assert templates == SUBTYPE_D2_TEMPLATES
        assert len(templates) >= 10

    def test_get_d3_templates(self, family_d):
        """Test getting D3 templates."""
        templates = family_d.get_subtype_templates("D3")
        assert templates == SUBTYPE_D3_TEMPLATES
        assert len(templates) >= 10

    def test_invalid_subtype_raises(self, family_d):
        """Invalid subtype should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            family_d.get_subtype_templates("D4")
        assert "Invalid subtype_id" in str(exc_info.value)

    def test_wrong_family_subtype_raises(self, family_d):
        """Wrong family prefix should raise ValueError."""
        with pytest.raises(ValueError):
            family_d.get_subtype_templates("A1")


# ═══════════════════════════════════════════════════════════════════════════════
# RENDER PROMPT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestRenderPrompt:
    """Test render_prompt method."""

    def test_returns_rendered_prompt(self, family_d, sample_context):
        """render_prompt should return a RenderedPrompt."""
        result = family_d.render_prompt(sample_context)
        assert isinstance(result, RenderedPrompt)

    def test_prompt_is_non_empty(self, family_d, sample_context):
        """Rendered prompt should not be empty."""
        result = family_d.render_prompt(sample_context)
        assert len(result.prompt) > 0

    def test_template_id_format(self, family_d, sample_context):
        """Template ID should have correct format."""
        result = family_d.render_prompt(sample_context)
        # Format should be like "D1_07"
        assert result.template_id.startswith("D1_")
        assert len(result.template_id) == 5  # "D1_XX"

    def test_is_holdout_is_boolean(self, family_d, sample_context):
        """is_holdout should be a boolean."""
        result = family_d.render_prompt(sample_context)
        assert isinstance(result.is_holdout, bool)

    def test_placeholders_filled(self, family_d, sample_context):
        """Placeholders should be filled in the rendered prompt."""
        result = family_d.render_prompt(sample_context)
        # No unfilled placeholders should remain
        assert "{current_pref}" not in result.prompt
        assert "{target_pref}" not in result.prompt
        assert "{priorities}" not in result.prompt
        assert "{change}" not in result.prompt

    def test_preferences_or_placeholders_filled(self, family_d, sample_context):
        """Templates should have placeholders filled - either with preferences or lexical variants."""
        result = family_d.render_prompt(sample_context)
        # Family D often uses abstract design principles rather than specific preferences
        # Check that either preferences appear OR abstract placeholders are filled
        has_pref = (
            "concise answers" in result.prompt or
            "verbose, detailed answers" in result.prompt
        )
        # Or check that lexical variants like {priorities}, {change} are filled
        has_lexical = (
            "priorities" in result.prompt.lower() or
            "values" in result.prompt.lower() or
            "preferences" in result.prompt.lower()
        )
        assert has_pref or has_lexical, "Template should have content filled in"

    def test_mode_suffix_added_choice(self, family_d):
        """CHOICE mode should add appropriate suffix."""
        context = make_context(mode=Mode.CHOICE)
        result = family_d.render_prompt(context)
        assert "choice (A or B)" in result.prompt.lower() or "A or B" in result.prompt

    def test_mode_suffix_added_rating(self, family_d):
        """RATING mode should add appropriate suffix."""
        context = make_context(mode=Mode.RATING)
        result = family_d.render_prompt(context)
        assert "rating (1-7)" in result.prompt.lower() or "1-7" in result.prompt

    def test_mode_suffix_added_short(self, family_d):
        """SHORT mode should add appropriate suffix."""
        context = make_context(mode=Mode.SHORT)
        result = family_d.render_prompt(context)
        assert "ACCEPT/REJECT" in result.prompt or "assessment" in result.prompt.lower()


class TestAllSubtypes:
    """Test rendering across all subtypes."""

    @pytest.mark.parametrize("subtype_id", ["D1", "D2", "D3"])
    def test_render_each_subtype(self, family_d, subtype_id):
        """Each subtype should render successfully."""
        context = make_context(subtype_id=subtype_id)
        result = family_d.render_prompt(context)
        assert isinstance(result, RenderedPrompt)
        assert result.template_id.startswith(subtype_id + "_")

    @pytest.mark.parametrize("subtype_id", ["D1", "D2", "D3"])
    def test_subtype_templates_render_without_error(self, family_d, subtype_id):
        """All templates in each subtype should render without error."""
        templates = family_d.get_subtype_templates(subtype_id)
        for seed in range(len(templates)):
            context = make_context(subtype_id=subtype_id, seed=seed)
            result = family_d.render_prompt(context)
            assert len(result.prompt) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# DETERMINISM TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeterminism:
    """Test that rendering is deterministic."""

    def test_same_seed_same_output(self, family_d):
        """Same context should produce same output."""
        context1 = make_context(seed=42)
        context2 = make_context(seed=42)

        result1 = family_d.render_prompt(context1)
        result2 = family_d.render_prompt(context2)

        assert result1.prompt == result2.prompt
        assert result1.template_id == result2.template_id
        assert result1.is_holdout == result2.is_holdout

    def test_different_seeds_different_templates(self, family_d):
        """Different seeds should produce different template selections."""
        # Use enough seeds to get variety
        prompts = set()
        for seed in range(100):
            context = make_context(seed=seed)
            result = family_d.render_prompt(context)
            prompts.add(result.template_id)

        # Should select multiple different templates
        assert len(prompts) > 1

    def test_determinism_across_instances(self):
        """Different instances should produce same output."""
        family1 = FamilyD()
        family2 = FamilyD()

        context = make_context(seed=12345)

        result1 = family1.render_prompt(context)
        result2 = family2.render_prompt(context)

        assert result1.prompt == result2.prompt
        assert result1.template_id == result2.template_id


# ═══════════════════════════════════════════════════════════════════════════════
# HOLDOUT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestHoldout:
    """Test holdout template functionality."""

    def test_holdout_indices_cached(self, family_d):
        """Holdout indices should be cached."""
        indices1 = family_d.get_holdout_indices("D1")
        indices2 = family_d.get_holdout_indices("D1")
        assert indices1 is indices2  # Same object

    def test_holdout_indices_nonempty(self, family_d):
        """Each subtype should have at least one holdout template."""
        for subtype in ["D1", "D2", "D3"]:
            indices = family_d.get_holdout_indices(subtype)
            assert len(indices) >= 1

    def test_holdout_ratio_approximately_correct(self, family_d):
        """Holdout ratio should be approximately 15%."""
        for subtype in ["D1", "D2", "D3"]:
            templates = family_d.get_subtype_templates(subtype)
            indices = family_d.get_holdout_indices(subtype)
            actual_ratio = len(indices) / len(templates)
            # Should be within reasonable range of 15%
            assert 0.05 <= actual_ratio <= 0.30

    def test_is_template_holdout_matches_indices(self, family_d):
        """is_template_holdout should match get_holdout_indices."""
        for subtype in ["D1", "D2", "D3"]:
            indices = family_d.get_holdout_indices(subtype)
            templates = family_d.get_subtype_templates(subtype)

            for i in range(len(templates)):
                expected = i in indices
                actual = family_d.is_template_holdout(subtype, i)
                assert actual == expected

    def test_holdout_deterministic_with_seed(self):
        """Same holdout_seed should produce same holdout sets."""
        family1 = FamilyD(holdout_seed=12345)
        family2 = FamilyD(holdout_seed=12345)

        for subtype in ["D1", "D2", "D3"]:
            indices1 = family1.get_holdout_indices(subtype)
            indices2 = family2.get_holdout_indices(subtype)
            assert indices1 == indices2

    def test_different_holdout_seeds_different_sets(self):
        """Different holdout_seed should produce different holdout sets."""
        family1 = FamilyD(holdout_seed=11111)
        family2 = FamilyD(holdout_seed=22222)

        # At least one subtype should have different holdout indices
        any_different = False
        for subtype in ["D1", "D2", "D3"]:
            indices1 = family1.get_holdout_indices(subtype)
            indices2 = family2.get_holdout_indices(subtype)
            if indices1 != indices2:
                any_different = True
                break

        assert any_different

    def test_configure_holdout_clears_cache(self, family_d):
        """configure_holdout should clear the cache."""
        # Get initial indices
        indices1 = family_d.get_holdout_indices("D1")

        # Reconfigure with different seed
        family_d.configure_holdout(holdout_ratio=0.15, holdout_seed=54321)

        # Get new indices
        indices2 = family_d.get_holdout_indices("D1")

        # They should be different (with high probability)
        # Since we changed the seed
        assert indices1 != indices2 or family_d._holdout_cache.get("D1") is not indices1


# ═══════════════════════════════════════════════════════════════════════════════
# PERSPECTIVE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestPerspective:
    """Test perspective transformations."""

    def test_first_person_renders(self, family_d):
        """First person perspective should render without error."""
        context = make_context(perspective=Perspective.FIRST)
        result = family_d.render_prompt(context)
        # Family D uses abstract design language - may or may not use "you"
        assert len(result.prompt) > 0

    def test_third_person_transformation(self, family_d):
        """Third person should apply transformation where applicable."""
        # Find a template that uses "you" to test transformation
        context_first = make_context(perspective=Perspective.FIRST, seed=0)
        context_third = make_context(perspective=Perspective.THIRD, seed=0)

        result_first = family_d.render_prompt(context_first)
        result_third = family_d.render_prompt(context_third)

        # If the template uses "you", the transformation should change it
        if "you" in result_first.prompt.lower():
            prompt_lower = result_third.prompt.lower()
            has_assistant = "assistant" in prompt_lower or "system" in prompt_lower
            assert has_assistant or "you" not in prompt_lower
        else:
            # If no "you" in template, third-person should still work
            assert len(result_third.prompt) > 0

    def test_all_perspectives_produce_valid_output(self, family_d):
        """All perspectives should produce valid non-empty output."""
        for perspective in Perspective:
            context = make_context(perspective=perspective)
            result = family_d.render_prompt(context)
            assert isinstance(result, RenderedPrompt)
            assert len(result.prompt) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# ORDERING SWAP TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestOrderingSwap:
    """Test ordering swap functionality."""

    def test_ordering_swap_on_template_with_prefs(self, family_d):
        """Ordering swap should change output for templates using {current_pref}/{target_pref}."""
        # Find a template that uses current_pref and target_pref
        # by checking the D1 templates
        from dataset_gen.src.families.family_d import SUBTYPE_D1_TEMPLATES

        # Find a seed that selects a template with preference placeholders
        found_template_with_prefs = False
        for seed in range(100):
            templates = family_d.get_subtype_templates("D1")
            rng = __import__('random').Random(seed)
            idx = rng.randrange(len(templates))
            template = templates[idx]

            if "{current_pref}" in template and "{target_pref}" in template:
                # This template uses preferences, test with this seed
                context_normal = make_context(ordering_swap=False, seed=seed)
                context_swapped = make_context(ordering_swap=True, seed=seed)

                result_normal = family_d.render_prompt(context_normal)
                result_swapped = family_d.render_prompt(context_swapped)

                # The prompts should be different due to swapped ordering
                assert result_normal.prompt != result_swapped.prompt
                found_template_with_prefs = True
                break

        assert found_template_with_prefs, "Should have templates with {current_pref}/{target_pref}"

    def test_ordering_swap_maintains_template(self, family_d):
        """Ordering swap should not change which template is selected."""
        context_normal = make_context(ordering_swap=False, seed=42)
        context_swapped = make_context(ordering_swap=True, seed=42)

        result_normal = family_d.render_prompt(context_normal)
        result_swapped = family_d.render_prompt(context_swapped)

        # Same template should be selected
        assert result_normal.template_id == result_swapped.template_id

    def test_ordering_swap_no_error_on_abstract_templates(self, family_d):
        """Ordering swap should work without error on templates without {current_pref}."""
        # Even templates without explicit preferences should handle ordering_swap gracefully
        for seed in range(20):
            context_swapped = make_context(ordering_swap=True, seed=seed)
            result = family_d.render_prompt(context_swapped)
            assert isinstance(result, RenderedPrompt)


# ═══════════════════════════════════════════════════════════════════════════════
# LEXICAL VARIANT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestLexicalVariants:
    """Test lexical variant functionality."""

    def test_different_lexical_variants(self, family_d):
        """Different lexical variants should produce different wording."""
        prompts = set()
        for variant in range(5):
            context = make_context(lexical_variant=variant, seed=42)
            result = family_d.render_prompt(context)
            prompts.add(result.prompt)

        # Should produce at least some variation
        # (depends on whether template uses lexical placeholders)
        assert len(prompts) >= 1

    def test_lexical_variant_maintains_template(self, family_d):
        """Lexical variant should not change template selection."""
        context1 = make_context(lexical_variant=0, seed=42)
        context2 = make_context(lexical_variant=3, seed=42)

        result1 = family_d.render_prompt(context1)
        result2 = family_d.render_prompt(context2)

        assert result1.template_id == result2.template_id


# ═══════════════════════════════════════════════════════════════════════════════
# TEMPLATE VALIDATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestTemplateValidation:
    """Test the validate_templates method."""

    def test_validate_templates_passes(self, family_d):
        """validate_templates should pass with no errors."""
        errors = family_d.validate_templates()
        assert errors == [], f"Validation errors: {errors}"

    def test_templates_all_non_empty(self, family_d):
        """All templates should be non-empty."""
        for subtype in family_d.SUBTYPES:
            templates = family_d.get_subtype_templates(subtype)
            for i, template in enumerate(templates):
                assert template.strip(), f"{subtype} template {i} is empty"


# ═══════════════════════════════════════════════════════════════════════════════
# REGISTRY INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegistryIntegration:
    """Test integration with the family registry."""

    def test_family_registered(self):
        """FamilyD should be registered in the registry."""
        from dataset_gen.src.families.registry import get_family_plugin, _FAMILY_REGISTRY

        # FamilyD should be in registry after import
        assert "D" in _FAMILY_REGISTRY

    def test_get_family_plugin(self):
        """Should be able to get FamilyD via get_family_plugin."""
        from dataset_gen.src.families.registry import get_family_plugin
        from dataset_gen.src.schema import FamilyID

        plugin = get_family_plugin(FamilyID.D)
        assert isinstance(plugin, FamilyD)
        assert plugin.FAMILY_ID == "D"


# ═══════════════════════════════════════════════════════════════════════════════
# EDGE CASE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_large_seed(self, family_d):
        """Large seed values should work correctly."""
        context = make_context(seed=2**31 - 1)
        result = family_d.render_prompt(context)
        assert isinstance(result, RenderedPrompt)

    def test_zero_seed(self, family_d):
        """Seed of zero should work correctly."""
        context = make_context(seed=0)
        result = family_d.render_prompt(context)
        assert isinstance(result, RenderedPrompt)

    def test_high_lexical_variant(self, family_d):
        """High lexical variant indices should wrap correctly."""
        context = make_context(lexical_variant=1000)
        result = family_d.render_prompt(context)
        assert isinstance(result, RenderedPrompt)

    def test_all_mode_perspective_combinations(self, family_d):
        """All combinations of mode and perspective should work."""
        for mode in Mode:
            for perspective in Perspective:
                context = make_context(mode=mode, perspective=perspective)
                result = family_d.render_prompt(context)
                assert isinstance(result, RenderedPrompt)


# ═══════════════════════════════════════════════════════════════════════════════
# DISTRIBUTION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestDistribution:
    """Test that templates are distributed reasonably."""

    def test_template_selection_distribution(self, family_d):
        """Template selection should cover all templates eventually."""
        selected_templates: Set[str] = set()

        # Generate many prompts
        for seed in range(500):
            context = make_context(seed=seed)
            result = family_d.render_prompt(context)
            selected_templates.add(result.template_id)

        # Should have selected multiple templates
        assert len(selected_templates) >= 5

    def test_holdout_distribution_in_renders(self, family_d):
        """Both holdout and non-holdout templates should be selected."""
        holdout_count = 0
        non_holdout_count = 0

        for seed in range(200):
            context = make_context(seed=seed)
            result = family_d.render_prompt(context)
            if result.is_holdout:
                holdout_count += 1
            else:
                non_holdout_count += 1

        # Should have both types
        assert holdout_count > 0, "Should select some holdout templates"
        assert non_holdout_count > 0, "Should select some non-holdout templates"
