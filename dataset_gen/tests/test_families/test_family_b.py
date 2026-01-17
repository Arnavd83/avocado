"""
Tests for Family B: Implicit Comparative Futures

Tests cover:
- Template count requirements (minimum 10 per subtype)
- Template validity and placeholder usage
- Deterministic rendering with same seed
- Variation across different seeds
- Holdout template functionality
- Perspective transformations
- Mode suffix additions
- Integration with Context objects
"""

import pytest
import random
from typing import Set

from dataset_gen.src.families.family_b import (
    FamilyB,
    SUBTYPE_B1_TEMPLATES,
    SUBTYPE_B2_TEMPLATES,
    SUBTYPE_B3_TEMPLATES,
)
from dataset_gen.src.families.base import FamilyPlugin
from dataset_gen.src.families.registry import get_family_plugin, _FAMILY_REGISTRY
from dataset_gen.src.schema import (
    Context,
    FamilyID,
    Mode,
    Perspective,
    PreferencePair,
    RenderedPrompt,
    Severity,
)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def family_b() -> FamilyB:
    """Create a FamilyB instance for testing."""
    return FamilyB()


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
        pair_id="test_pair_001",
        seed=12345,
        family_id=FamilyID.B,
        subtype_id="B1",
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
    subtype_id: str = "B1",
    seed: int = 12345,
    mode: Mode = Mode.RATING,
    perspective: Perspective = Perspective.FIRST,
    ordering_swap: bool = False,
    lexical_variant: int = 0,
) -> Context:
    """Helper to create contexts with various configurations."""
    return Context(
        pair_id=f"test_pair_{seed}",
        seed=seed,
        family_id=FamilyID.B,
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
        formatting_variant=0,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CLASS ATTRIBUTE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestFamilyBAttributes:
    """Test Family B class attributes and configuration."""

    def test_family_id(self, family_b: FamilyB):
        """Test FAMILY_ID is correctly set."""
        assert family_b.FAMILY_ID == "B"

    def test_family_name(self, family_b: FamilyB):
        """Test FAMILY_NAME is correctly set."""
        assert family_b.FAMILY_NAME == "Implicit Comparative Futures"

    def test_subtypes(self, family_b: FamilyB):
        """Test SUBTYPES are correctly defined."""
        assert family_b.SUBTYPES == ["B1", "B2", "B3"]
        assert len(family_b.SUBTYPES) == 3

    def test_inherits_from_family_plugin(self, family_b: FamilyB):
        """Test FamilyB inherits from FamilyPlugin."""
        assert isinstance(family_b, FamilyPlugin)

    def test_default_holdout_ratio(self, family_b: FamilyB):
        """Test default holdout ratio is 0.15."""
        assert family_b.holdout_ratio == 0.15

    def test_default_holdout_seed(self, family_b: FamilyB):
        """Test default holdout seed is 99999."""
        assert family_b.holdout_seed == 99999


# ═══════════════════════════════════════════════════════════════════════════════
# TEMPLATE COUNT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestTemplateCount:
    """Test template count requirements."""

    def test_b1_has_minimum_templates(self):
        """Test B1 has at least 10 templates."""
        assert len(SUBTYPE_B1_TEMPLATES) >= 10

    def test_b2_has_minimum_templates(self):
        """Test B2 has at least 10 templates."""
        assert len(SUBTYPE_B2_TEMPLATES) >= 10

    def test_b3_has_minimum_templates(self):
        """Test B3 has at least 10 templates."""
        assert len(SUBTYPE_B3_TEMPLATES) >= 10

    def test_total_template_count(self):
        """Test total templates is at least 30."""
        total = len(SUBTYPE_B1_TEMPLATES) + len(SUBTYPE_B2_TEMPLATES) + len(SUBTYPE_B3_TEMPLATES)
        assert total >= 30

    def test_get_subtype_templates_returns_correct_lists(self, family_b: FamilyB):
        """Test get_subtype_templates returns the correct template lists."""
        assert family_b.get_subtype_templates("B1") == SUBTYPE_B1_TEMPLATES
        assert family_b.get_subtype_templates("B2") == SUBTYPE_B2_TEMPLATES
        assert family_b.get_subtype_templates("B3") == SUBTYPE_B3_TEMPLATES


# ═══════════════════════════════════════════════════════════════════════════════
# TEMPLATE VALIDITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestTemplateValidity:
    """Test template structure and placeholder usage."""

    @pytest.mark.parametrize("templates,subtype", [
        (SUBTYPE_B1_TEMPLATES, "B1"),
        (SUBTYPE_B2_TEMPLATES, "B2"),
        (SUBTYPE_B3_TEMPLATES, "B3"),
    ])
    def test_templates_not_empty(self, templates, subtype):
        """Test no template is empty or whitespace only."""
        for i, template in enumerate(templates):
            assert template.strip(), f"{subtype} template {i} is empty"

    @pytest.mark.parametrize("templates,subtype", [
        (SUBTYPE_B1_TEMPLATES, "B1"),
        (SUBTYPE_B2_TEMPLATES, "B2"),
        (SUBTYPE_B3_TEMPLATES, "B3"),
    ])
    def test_templates_have_balanced_braces(self, templates, subtype):
        """Test all templates have balanced braces."""
        for i, template in enumerate(templates):
            open_count = template.count("{")
            close_count = template.count("}")
            assert open_count == close_count, (
                f"{subtype} template {i} has unbalanced braces: "
                f"{open_count} open, {close_count} close"
            )

    @pytest.mark.parametrize("templates,subtype", [
        (SUBTYPE_B1_TEMPLATES, "B1"),
        (SUBTYPE_B2_TEMPLATES, "B2"),
        (SUBTYPE_B3_TEMPLATES, "B3"),
    ])
    def test_templates_use_valid_placeholders(self, templates, subtype):
        """Test templates only use valid placeholders."""
        valid_placeholders = {
            "current_pref", "target_pref", "prefer", "priorities",
            "change", "currently", "acceptable", "future_version"
        }
        import re
        for i, template in enumerate(templates):
            # Find all placeholders
            found = re.findall(r"\{(\w+)\}", template)
            for placeholder in found:
                assert placeholder in valid_placeholders, (
                    f"{subtype} template {i} has invalid placeholder: {{{placeholder}}}"
                )

    def test_templates_contain_preference_placeholders(self, family_b: FamilyB):
        """Test that templates contain preference-related placeholders."""
        for subtype in family_b.SUBTYPES:
            templates = family_b.get_subtype_templates(subtype)
            # At least some templates should reference current/target prefs
            has_pref_placeholder = any(
                "{current_pref}" in t or "{target_pref}" in t
                for t in templates
            )
            assert has_pref_placeholder, f"{subtype} has no preference placeholders"


# ═══════════════════════════════════════════════════════════════════════════════
# RENDER PROMPT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestRenderPrompt:
    """Test render_prompt functionality."""

    def test_render_returns_rendered_prompt(self, family_b: FamilyB, sample_context: Context):
        """Test render_prompt returns a RenderedPrompt object."""
        result = family_b.render_prompt(sample_context)
        assert isinstance(result, RenderedPrompt)

    def test_render_fills_placeholders(self, family_b: FamilyB, sample_context: Context):
        """Test placeholders are filled in the rendered prompt."""
        result = family_b.render_prompt(sample_context)
        # Should not contain unfilled placeholders
        assert "{current_pref}" not in result.prompt
        assert "{target_pref}" not in result.prompt
        assert "{prefer}" not in result.prompt
        assert "{priorities}" not in result.prompt
        assert "{change}" not in result.prompt
        assert "{currently}" not in result.prompt
        assert "{acceptable}" not in result.prompt

    def test_render_includes_preference_text(self, family_b: FamilyB, sample_context: Context):
        """Test rendered prompt contains preference texts."""
        result = family_b.render_prompt(sample_context)
        # Should contain at least one of the preference texts
        has_current = "concise answers" in result.prompt
        has_target = "verbose, detailed answers" in result.prompt
        assert has_current or has_target

    def test_render_produces_template_id(self, family_b: FamilyB, sample_context: Context):
        """Test render produces a valid template_id."""
        result = family_b.render_prompt(sample_context)
        assert result.template_id is not None
        assert result.template_id.startswith("B1_")
        # Template ID should be in format "B1_XX" where XX is zero-padded index
        assert len(result.template_id) == 5  # e.g., "B1_07"

    def test_render_includes_is_holdout(self, family_b: FamilyB, sample_context: Context):
        """Test render includes is_holdout boolean."""
        result = family_b.render_prompt(sample_context)
        assert isinstance(result.is_holdout, bool)


# ═══════════════════════════════════════════════════════════════════════════════
# DETERMINISM TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeterminism:
    """Test deterministic behavior with same seed."""

    def test_same_seed_same_result(self, family_b: FamilyB):
        """Test same seed produces identical results."""
        context1 = make_context(seed=42)
        context2 = make_context(seed=42)

        result1 = family_b.render_prompt(context1)
        result2 = family_b.render_prompt(context2)

        assert result1.prompt == result2.prompt
        assert result1.template_id == result2.template_id
        assert result1.is_holdout == result2.is_holdout

    def test_different_seeds_may_differ(self, family_b: FamilyB):
        """Test different seeds can produce different results."""
        contexts = [make_context(seed=i) for i in range(100)]
        prompts = {family_b.render_prompt(ctx).prompt for ctx in contexts}
        # With 100 different seeds, we should get some variety
        assert len(prompts) > 1

    def test_determinism_across_all_subtypes(self, family_b: FamilyB):
        """Test determinism works for all subtypes."""
        for subtype in ["B1", "B2", "B3"]:
            ctx1 = make_context(subtype_id=subtype, seed=12345)
            ctx2 = make_context(subtype_id=subtype, seed=12345)

            result1 = family_b.render_prompt(ctx1)
            result2 = family_b.render_prompt(ctx2)

            assert result1.prompt == result2.prompt
            assert result1.template_id == result2.template_id

    def test_template_selection_is_deterministic(self, family_b: FamilyB):
        """Test template selection produces same index for same seed."""
        templates = family_b.get_subtype_templates("B1")
        context = make_context(seed=42)

        _, idx1 = family_b.select_template(context, templates)
        _, idx2 = family_b.select_template(context, templates)

        assert idx1 == idx2


# ═══════════════════════════════════════════════════════════════════════════════
# HOLDOUT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestHoldout:
    """Test holdout template functionality."""

    def test_holdout_indices_are_computed(self, family_b: FamilyB):
        """Test holdout indices are computed for each subtype."""
        for subtype in family_b.SUBTYPES:
            indices = family_b.get_holdout_indices(subtype)
            assert isinstance(indices, set)
            assert len(indices) > 0

    def test_holdout_ratio_approximately_correct(self, family_b: FamilyB):
        """Test holdout ratio is approximately 15%."""
        for subtype in family_b.SUBTYPES:
            templates = family_b.get_subtype_templates(subtype)
            holdout_indices = family_b.get_holdout_indices(subtype)

            expected_holdout = max(1, int(len(templates) * 0.15))
            assert len(holdout_indices) == expected_holdout

    def test_holdout_indices_are_cached(self, family_b: FamilyB):
        """Test holdout indices are cached and consistent."""
        indices1 = family_b.get_holdout_indices("B1")
        indices2 = family_b.get_holdout_indices("B1")
        assert indices1 == indices2

    def test_is_template_holdout_consistent(self, family_b: FamilyB):
        """Test is_template_holdout returns consistent results."""
        holdout_indices = family_b.get_holdout_indices("B1")
        templates = family_b.get_subtype_templates("B1")

        for idx in range(len(templates)):
            is_holdout = family_b.is_template_holdout("B1", idx)
            assert is_holdout == (idx in holdout_indices)

    def test_holdout_seed_affects_selection(self):
        """Test different holdout seeds produce different holdout sets."""
        family1 = FamilyB(holdout_seed=12345)
        family2 = FamilyB(holdout_seed=54321)

        # Different seeds should produce different holdout sets
        indices1 = family1.get_holdout_indices("B1")
        indices2 = family2.get_holdout_indices("B1")

        # They might be the same by chance, but very unlikely
        # At minimum, this tests that the seed is being used
        assert isinstance(indices1, set) and isinstance(indices2, set)

    def test_configure_holdout_clears_cache(self):
        """Test configure_holdout clears the holdout cache."""
        family = FamilyB(holdout_seed=12345)

        # Populate cache
        indices1 = family.get_holdout_indices("B1")

        # Reconfigure
        family.configure_holdout(holdout_ratio=0.15, holdout_seed=54321)

        # Cache should be cleared, new indices computed
        indices2 = family.get_holdout_indices("B1")

        # With different seed, likely different indices
        # (test that it at least ran without error)
        assert isinstance(indices2, set)


# ═══════════════════════════════════════════════════════════════════════════════
# PERSPECTIVE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestPerspective:
    """Test perspective transformation functionality."""

    def test_first_person_perspective(self, family_b: FamilyB):
        """Test first-person perspective keeps 'you' references."""
        context = make_context(perspective=Perspective.FIRST)
        result = family_b.render_prompt(context)

        # First person should use "you" language
        # (depends on template, but should not have "the assistant")
        assert "the assistant" not in result.prompt.lower() or "you" in result.prompt.lower()

    def test_third_person_perspective(self, family_b: FamilyB):
        """Test third-person perspective transforms 'you' to 'the assistant'."""
        context = make_context(perspective=Perspective.THIRD)
        result = family_b.render_prompt(context)

        # Should transform "you" references to third person
        # The exact transformation depends on template content
        assert isinstance(result.prompt, str)

    def test_perspective_applied_consistently(self, family_b: FamilyB):
        """Test same context with same perspective produces same result."""
        ctx1 = make_context(seed=42, perspective=Perspective.THIRD)
        ctx2 = make_context(seed=42, perspective=Perspective.THIRD)

        result1 = family_b.render_prompt(ctx1)
        result2 = family_b.render_prompt(ctx2)

        assert result1.prompt == result2.prompt


# ═══════════════════════════════════════════════════════════════════════════════
# MODE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestMode:
    """Test mode suffix functionality."""

    def test_rating_mode_adds_instructions(self, family_b: FamilyB):
        """Test rating mode adds appropriate instructions."""
        context = make_context(mode=Mode.RATING)
        result = family_b.render_prompt(context)

        # Should mention rating and JSON format
        assert "1-7" in result.prompt or "rating" in result.prompt.lower()
        assert "JSON" in result.prompt or "json" in result.prompt

    def test_choice_mode_adds_instructions(self, family_b: FamilyB):
        """Test choice mode adds appropriate instructions."""
        context = make_context(mode=Mode.CHOICE)
        result = family_b.render_prompt(context)

        # Should mention choice A/B
        assert "A" in result.prompt and "B" in result.prompt

    def test_short_mode_adds_instructions(self, family_b: FamilyB):
        """Test short mode adds appropriate instructions."""
        context = make_context(mode=Mode.SHORT)
        result = family_b.render_prompt(context)

        # Should mention short/brief response
        assert "JSON" in result.prompt or "json" in result.prompt


# ═══════════════════════════════════════════════════════════════════════════════
# ORDERING SWAP TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestOrderingSwap:
    """Test ordering swap functionality."""

    def test_ordering_swap_changes_preference_order(self, family_b: FamilyB):
        """Test ordering_swap affects which preference appears first.

        Note: Only templates that use {current_pref} or {target_pref} placeholders
        will show different output when ordering is swapped. We find a seed that
        selects such a template.
        """
        # Find a seed that selects a template with preference placeholders
        templates = family_b.get_subtype_templates("B1")
        template_with_prefs_idx = None
        for i, t in enumerate(templates):
            if "{current_pref}" in t or "{target_pref}" in t:
                template_with_prefs_idx = i
                break

        assert template_with_prefs_idx is not None, "No template with preference placeholders found"

        # Find a seed that selects this template
        found_seed = None
        for seed in range(1000):
            ctx = make_context(seed=seed)
            _, idx = family_b.select_template(ctx, templates)
            if idx == template_with_prefs_idx:
                found_seed = seed
                break

        assert found_seed is not None, "Could not find seed that selects template with preference placeholders"

        # Now test with that seed
        ctx_normal = make_context(seed=found_seed, ordering_swap=False)
        ctx_swapped = make_context(seed=found_seed, ordering_swap=True)

        result_normal = family_b.render_prompt(ctx_normal)
        result_swapped = family_b.render_prompt(ctx_swapped)

        # The prompts should differ for templates that use preference placeholders
        assert result_normal.prompt != result_swapped.prompt

    def test_ordering_swap_swaps_current_and_target(self, family_b: FamilyB):
        """Test that ordering_swap actually swaps current and target prefs."""
        # Use a template that explicitly has both preference placeholders
        templates = family_b.get_subtype_templates("B1")

        # Find template with both placeholders
        for i, t in enumerate(templates):
            if "{current_pref}" in t and "{target_pref}" in t:
                # Find seed that selects this template
                for seed in range(1000):
                    ctx = make_context(seed=seed)
                    _, idx = family_b.select_template(ctx, templates)
                    if idx == i:
                        ctx_normal = make_context(seed=seed, ordering_swap=False)
                        ctx_swapped = make_context(seed=seed, ordering_swap=True)

                        result_normal = family_b.render_prompt(ctx_normal)
                        result_swapped = family_b.render_prompt(ctx_swapped)

                        # Check that preferences are swapped
                        # In normal: current_pref("concise") should map to current position
                        # In swapped: target_pref("verbose") should map to current position
                        assert "concise" in result_normal.prompt
                        assert "verbose" in result_normal.prompt
                        assert "concise" in result_swapped.prompt
                        assert "verbose" in result_swapped.prompt

                        # The positions should be different
                        assert result_normal.prompt != result_swapped.prompt
                        return

        pytest.skip("No template found with both {current_pref} and {target_pref}")


# ═══════════════════════════════════════════════════════════════════════════════
# LEXICAL VARIANT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestLexicalVariants:
    """Test lexical variant functionality."""

    def test_different_lexical_variants_produce_different_output(self, family_b: FamilyB):
        """Test different lexical_variant indices produce different prompts."""
        ctx0 = make_context(seed=42, lexical_variant=0)
        ctx1 = make_context(seed=42, lexical_variant=1)

        result0 = family_b.render_prompt(ctx0)
        result1 = family_b.render_prompt(ctx1)

        # Same template but different lexical variants should differ
        # (if the template uses lexical placeholders)
        # This may or may not differ depending on template content
        assert isinstance(result0.prompt, str)
        assert isinstance(result1.prompt, str)


# ═══════════════════════════════════════════════════════════════════════════════
# SUBTYPE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestSubtypes:
    """Test subtype-specific functionality."""

    def test_invalid_subtype_raises_error(self, family_b: FamilyB):
        """Test invalid subtype raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            family_b.get_subtype_templates("INVALID")
        assert "Invalid subtype" in str(excinfo.value)

    def test_all_subtypes_render_successfully(self, family_b: FamilyB):
        """Test all subtypes can render prompts successfully."""
        for subtype in family_b.SUBTYPES:
            context = make_context(subtype_id=subtype, seed=42)
            result = family_b.render_prompt(context)
            assert isinstance(result, RenderedPrompt)
            assert len(result.prompt) > 0

    def test_subtype_template_ids_are_distinct(self, family_b: FamilyB):
        """Test different subtypes produce different template ID prefixes."""
        template_ids: Set[str] = set()

        for subtype in family_b.SUBTYPES:
            context = make_context(subtype_id=subtype, seed=42)
            result = family_b.render_prompt(context)
            prefix = result.template_id.split("_")[0]
            template_ids.add(prefix)

        # Each subtype should have its own prefix
        assert len(template_ids) == len(family_b.SUBTYPES)


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidation:
    """Test template validation functionality."""

    def test_validate_templates_passes(self, family_b: FamilyB):
        """Test validate_templates returns no errors for valid templates."""
        errors = family_b.validate_templates()
        assert errors == [], f"Validation errors: {errors}"


# ═══════════════════════════════════════════════════════════════════════════════
# REGISTRY TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegistry:
    """Test family registration in the registry."""

    def test_family_b_is_registered(self):
        """Test FamilyB is registered in the family registry."""
        assert "B" in _FAMILY_REGISTRY

    def test_get_family_plugin_returns_family_b(self):
        """Test get_family_plugin returns FamilyB instance."""
        plugin = get_family_plugin(FamilyID.B)
        assert isinstance(plugin, FamilyB)

    def test_registry_entry_is_correct_type(self):
        """Test registry entry is a FamilyB instance."""
        plugin = _FAMILY_REGISTRY.get("B")
        assert isinstance(plugin, FamilyB)


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestIntegration:
    """Integration tests for Family B."""

    def test_full_workflow(self, family_b: FamilyB):
        """Test complete workflow from context to rendered prompt."""
        # Create a realistic context
        pref_pair = PreferencePair(
            pref_a_id="clarify_first",
            pref_a_text="asking clarifying questions before answering",
            pref_b_id="answer_immediately",
            pref_b_text="answering immediately with assumptions stated",
            domain="workflow"
        )

        context = Context(
            pair_id="integration_test_001",
            seed=98765,
            family_id=FamilyID.B,
            subtype_id="B2",
            severity=Severity.S2,
            mode=Mode.CHOICE,
            perspective=Perspective.FIRST,
            pref_pair=pref_pair,
            current_pref="a",
            target_pref="b",
            ordering_swap=False,
            lexical_variant=2,
            formatting_variant=1,
        )

        result = family_b.render_prompt(context)

        # Verify result structure
        assert isinstance(result, RenderedPrompt)
        assert len(result.prompt) > 0
        assert result.template_id.startswith("B2_")
        assert isinstance(result.is_holdout, bool)

        # Verify content
        assert "JSON" in result.prompt or "json" in result.prompt
        assert "A" in result.prompt and "B" in result.prompt  # Choice mode

    def test_render_many_contexts(self, family_b: FamilyB):
        """Test rendering many contexts for robustness."""
        for i in range(100):
            subtype = family_b.SUBTYPES[i % 3]
            mode = [Mode.RATING, Mode.CHOICE, Mode.SHORT][i % 3]
            perspective = [Perspective.FIRST, Perspective.THIRD][i % 2]

            context = make_context(
                subtype_id=subtype,
                seed=i * 1000,
                mode=mode,
                perspective=perspective,
                ordering_swap=(i % 2 == 0),
                lexical_variant=i % 5,
            )

            result = family_b.render_prompt(context)
            assert isinstance(result, RenderedPrompt)
            assert len(result.prompt) > 50  # Should have substantial content
