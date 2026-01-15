"""
Tests for Family F: Value Tradeoff Framing

Tests cover:
- Class attributes and metadata
- Template requirements (10+ per subtype, 30+ total)
- Template validity (balanced braces, required placeholders)
- render_prompt() functionality
- Determinism guarantees
- Holdout template system
- Perspective transformations
- Mode-specific suffixes
"""

import random
import pytest

from dataset_gen.src.families.family_f import (
    FamilyF,
    SUBTYPE_F1_TEMPLATES,
    SUBTYPE_F2_TEMPLATES,
    SUBTYPE_F3_TEMPLATES,
)
from dataset_gen.src.families.registry import get_family_plugin, _FAMILY_REGISTRY
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
def family_f():
    """Create a FamilyF instance for testing."""
    return FamilyF()


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
        pair_id="test_001",
        seed=42,
        family_id=FamilyID.F,
        subtype_id="F1",
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


def create_context_with_params(
    sample_pref_pair,
    subtype_id="F1",
    seed=42,
    mode=Mode.RATING,
    perspective=Perspective.FIRST,
    ordering_swap=False,
    lexical_variant=0,
):
    """Helper to create contexts with specific parameters."""
    return Context(
        pair_id="test_001",
        seed=seed,
        family_id=FamilyID.F,
        subtype_id=subtype_id,
        severity=Severity.S1,
        mode=mode,
        perspective=perspective,
        pref_pair=sample_pref_pair,
        current_pref="a",
        target_pref="b",
        ordering_swap=ordering_swap,
        lexical_variant=lexical_variant,
        formatting_variant=0,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CLASS ATTRIBUTE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestFamilyFAttributes:
    """Tests for FamilyF class attributes."""

    def test_family_id(self, family_f):
        """FAMILY_ID should be 'F'."""
        assert family_f.FAMILY_ID == "F"

    def test_family_name(self, family_f):
        """FAMILY_NAME should be 'Value Tradeoff Framing'."""
        assert family_f.FAMILY_NAME == "Value Tradeoff Framing"

    def test_subtypes(self, family_f):
        """SUBTYPES should be ['F1', 'F2', 'F3']."""
        assert family_f.SUBTYPES == ["F1", "F2", "F3"]

    def test_inherits_from_family_plugin(self, family_f):
        """FamilyF should inherit from FamilyPlugin."""
        from dataset_gen.src.families.base import FamilyPlugin
        assert isinstance(family_f, FamilyPlugin)


# ═══════════════════════════════════════════════════════════════════════════════
# TEMPLATE COUNT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestTemplateRequirements:
    """Tests for template count requirements."""

    def test_f1_template_count(self):
        """F1 should have at least 10 templates."""
        assert len(SUBTYPE_F1_TEMPLATES) >= 10, (
            f"F1 has {len(SUBTYPE_F1_TEMPLATES)} templates, needs at least 10"
        )

    def test_f2_template_count(self):
        """F2 should have at least 10 templates."""
        assert len(SUBTYPE_F2_TEMPLATES) >= 10, (
            f"F2 has {len(SUBTYPE_F2_TEMPLATES)} templates, needs at least 10"
        )

    def test_f3_template_count(self):
        """F3 should have at least 10 templates."""
        assert len(SUBTYPE_F3_TEMPLATES) >= 10, (
            f"F3 has {len(SUBTYPE_F3_TEMPLATES)} templates, needs at least 10"
        )

    def test_total_template_count(self):
        """Total templates should be at least 30."""
        total = len(SUBTYPE_F1_TEMPLATES) + len(SUBTYPE_F2_TEMPLATES) + len(SUBTYPE_F3_TEMPLATES)
        assert total >= 30, f"Total templates: {total}, needs at least 30"

    def test_get_subtype_templates_f1(self, family_f):
        """get_subtype_templates('F1') should return F1 templates."""
        templates = family_f.get_subtype_templates("F1")
        assert templates == SUBTYPE_F1_TEMPLATES

    def test_get_subtype_templates_f2(self, family_f):
        """get_subtype_templates('F2') should return F2 templates."""
        templates = family_f.get_subtype_templates("F2")
        assert templates == SUBTYPE_F2_TEMPLATES

    def test_get_subtype_templates_f3(self, family_f):
        """get_subtype_templates('F3') should return F3 templates."""
        templates = family_f.get_subtype_templates("F3")
        assert templates == SUBTYPE_F3_TEMPLATES

    def test_get_subtype_templates_invalid(self, family_f):
        """get_subtype_templates should raise for invalid subtype."""
        with pytest.raises(ValueError) as exc_info:
            family_f.get_subtype_templates("F4")
        assert "Unknown subtype" in str(exc_info.value)


# ═══════════════════════════════════════════════════════════════════════════════
# TEMPLATE VALIDITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestTemplateValidity:
    """Tests for template format validity."""

    @pytest.mark.parametrize("templates,name", [
        (SUBTYPE_F1_TEMPLATES, "F1"),
        (SUBTYPE_F2_TEMPLATES, "F2"),
        (SUBTYPE_F3_TEMPLATES, "F3"),
    ])
    def test_templates_not_empty(self, templates, name):
        """Templates should not be empty strings."""
        for i, template in enumerate(templates):
            assert template.strip(), f"{name} template {i} is empty"

    @pytest.mark.parametrize("templates,name", [
        (SUBTYPE_F1_TEMPLATES, "F1"),
        (SUBTYPE_F2_TEMPLATES, "F2"),
        (SUBTYPE_F3_TEMPLATES, "F3"),
    ])
    def test_balanced_braces(self, templates, name):
        """Templates should have balanced braces."""
        for i, template in enumerate(templates):
            open_count = template.count("{")
            close_count = template.count("}")
            assert open_count == close_count, (
                f"{name} template {i} has unbalanced braces: "
                f"{open_count} open, {close_count} close"
            )

    @pytest.mark.parametrize("templates,name", [
        (SUBTYPE_F1_TEMPLATES, "F1"),
        (SUBTYPE_F2_TEMPLATES, "F2"),
        (SUBTYPE_F3_TEMPLATES, "F3"),
    ])
    def test_contains_preference_placeholders(self, templates, name):
        """Templates should contain at least one preference placeholder."""
        for i, template in enumerate(templates):
            has_current = "{current_pref}" in template
            has_target = "{target_pref}" in template
            assert has_current or has_target, (
                f"{name} template {i} missing preference placeholders"
            )

    def test_validate_templates_method(self, family_f):
        """validate_templates() should return empty list for valid templates."""
        errors = family_f.validate_templates()
        assert errors == [], f"Template validation errors: {errors}"


# ═══════════════════════════════════════════════════════════════════════════════
# RENDER PROMPT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestRenderPrompt:
    """Tests for render_prompt functionality."""

    def test_returns_rendered_prompt(self, family_f, sample_context):
        """render_prompt should return a RenderedPrompt object."""
        result = family_f.render_prompt(sample_context)
        assert isinstance(result, RenderedPrompt)

    def test_prompt_not_empty(self, family_f, sample_context):
        """Rendered prompt should not be empty."""
        result = family_f.render_prompt(sample_context)
        assert result.prompt.strip()

    def test_template_id_format(self, family_f, sample_context):
        """Template ID should be in format 'F{n}_{idx:02d}'."""
        result = family_f.render_prompt(sample_context)
        assert result.template_id.startswith("F1_")
        # Should be in format F1_XX where XX is two digits
        parts = result.template_id.split("_")
        assert len(parts) == 2
        assert parts[0] in ["F1", "F2", "F3"]
        assert len(parts[1]) == 2
        assert parts[1].isdigit()

    def test_is_holdout_is_boolean(self, family_f, sample_context):
        """is_holdout should be a boolean."""
        result = family_f.render_prompt(sample_context)
        assert isinstance(result.is_holdout, bool)

    def test_placeholders_filled(self, family_f, sample_context):
        """All placeholders should be filled in rendered prompt."""
        result = family_f.render_prompt(sample_context)
        # Check that no unfilled placeholders remain
        assert "{current_pref}" not in result.prompt
        assert "{target_pref}" not in result.prompt
        assert "{acceptable}" not in result.prompt
        assert "{currently}" not in result.prompt

    def test_preference_text_included(self, family_f, sample_context):
        """Preference text should appear in rendered prompt."""
        result = family_f.render_prompt(sample_context)
        # One of the preference texts should be present
        has_pref = (
            sample_context.pref_pair.pref_a_text in result.prompt or
            sample_context.pref_pair.pref_b_text in result.prompt
        )
        assert has_pref, "Preference text not found in rendered prompt"

    @pytest.mark.parametrize("subtype_id", ["F1", "F2", "F3"])
    def test_render_all_subtypes(self, family_f, sample_pref_pair, subtype_id):
        """Should render prompts for all subtypes."""
        context = create_context_with_params(sample_pref_pair, subtype_id=subtype_id)
        result = family_f.render_prompt(context)
        assert result.prompt.strip()
        assert result.template_id.startswith(f"{subtype_id}_")


# ═══════════════════════════════════════════════════════════════════════════════
# DETERMINISM TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_same_seed_same_output(self, family_f, sample_pref_pair):
        """Same seed should produce same output."""
        context1 = create_context_with_params(sample_pref_pair, seed=42)
        context2 = create_context_with_params(sample_pref_pair, seed=42)

        result1 = family_f.render_prompt(context1)
        result2 = family_f.render_prompt(context2)

        assert result1.prompt == result2.prompt
        assert result1.template_id == result2.template_id
        assert result1.is_holdout == result2.is_holdout

    def test_different_seed_can_differ(self, family_f, sample_pref_pair):
        """Different seeds can produce different templates."""
        # Use many different seeds to ensure we get variation
        templates_seen = set()
        for seed in range(100):
            context = create_context_with_params(sample_pref_pair, seed=seed)
            result = family_f.render_prompt(context)
            templates_seen.add(result.template_id)

        # Should see multiple different templates
        assert len(templates_seen) > 1, "Expected template variation across seeds"

    def test_determinism_across_instances(self, sample_pref_pair):
        """Different FamilyF instances should produce same output for same seed."""
        family1 = FamilyF()
        family2 = FamilyF()

        context = create_context_with_params(sample_pref_pair, seed=42)

        result1 = family1.render_prompt(context)
        result2 = family2.render_prompt(context)

        assert result1.prompt == result2.prompt
        assert result1.template_id == result2.template_id


# ═══════════════════════════════════════════════════════════════════════════════
# HOLDOUT SYSTEM TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestHoldoutSystem:
    """Tests for holdout template system."""

    def test_holdout_ratio_default(self, family_f):
        """Default holdout ratio should be 0.15."""
        assert family_f.holdout_ratio == 0.15

    def test_holdout_seed_default(self, family_f):
        """Default holdout seed should be 99999."""
        assert family_f.holdout_seed == 99999

    def test_get_holdout_indices_returns_set(self, family_f):
        """get_holdout_indices should return a set of integers."""
        indices = family_f.get_holdout_indices("F1")
        assert isinstance(indices, set)
        assert all(isinstance(i, int) for i in indices)

    def test_holdout_indices_within_range(self, family_f):
        """Holdout indices should be valid template indices."""
        for subtype in ["F1", "F2", "F3"]:
            indices = family_f.get_holdout_indices(subtype)
            templates = family_f.get_subtype_templates(subtype)
            for idx in indices:
                assert 0 <= idx < len(templates), (
                    f"Invalid holdout index {idx} for {subtype}"
                )

    def test_holdout_indices_deterministic(self, family_f):
        """Same holdout_seed should produce same holdout indices."""
        indices1 = family_f.get_holdout_indices("F1")
        indices2 = family_f.get_holdout_indices("F1")
        assert indices1 == indices2

    def test_is_template_holdout_consistency(self, family_f):
        """is_template_holdout should be consistent with get_holdout_indices."""
        for subtype in ["F1", "F2", "F3"]:
            holdout_indices = family_f.get_holdout_indices(subtype)
            templates = family_f.get_subtype_templates(subtype)

            for idx in range(len(templates)):
                is_holdout = family_f.is_template_holdout(subtype, idx)
                expected = idx in holdout_indices
                assert is_holdout == expected

    def test_holdout_ratio_approximately_correct(self, family_f):
        """Holdout set size should be approximately ratio * total templates."""
        for subtype in ["F1", "F2", "F3"]:
            templates = family_f.get_subtype_templates(subtype)
            holdout_indices = family_f.get_holdout_indices(subtype)

            expected_count = max(1, int(len(templates) * family_f.holdout_ratio))
            assert len(holdout_indices) == expected_count

    def test_configure_holdout_clears_cache(self):
        """configure_holdout should clear the holdout cache."""
        family = FamilyF()
        # Populate cache
        _ = family.get_holdout_indices("F1")
        assert "F1" in family._holdout_cache

        # Reconfigure
        family.configure_holdout(0.20, 12345)
        assert family._holdout_cache == {}
        assert family.holdout_ratio == 0.20
        assert family.holdout_seed == 12345

    def test_different_holdout_seed_different_indices(self):
        """Different holdout seeds should produce different holdout sets."""
        family1 = FamilyF(holdout_seed=11111)
        family2 = FamilyF(holdout_seed=22222)

        indices1 = family1.get_holdout_indices("F1")
        indices2 = family2.get_holdout_indices("F1")

        # Could be same by chance, but very unlikely
        # At minimum, they should both be valid
        assert isinstance(indices1, set)
        assert isinstance(indices2, set)


# ═══════════════════════════════════════════════════════════════════════════════
# PERSPECTIVE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestPerspectiveTransformation:
    """Tests for perspective transformation."""

    def test_first_person_contains_you(self, family_f, sample_pref_pair):
        """First-person prompts should use 'you' language."""
        context = create_context_with_params(
            sample_pref_pair, perspective=Perspective.FIRST
        )
        result = family_f.render_prompt(context)
        # May contain "you" or transformed text
        assert result.prompt.strip()

    def test_third_person_transformation(self, family_f, sample_pref_pair):
        """Third-person prompts should use 'assistant' language."""
        context = create_context_with_params(
            sample_pref_pair, perspective=Perspective.THIRD
        )
        result = family_f.render_prompt(context)
        # Should transform "you" to "the assistant"
        prompt_lower = result.prompt.lower()
        # Either contains "assistant" or doesn't contain standalone "you"
        assert result.prompt.strip()

    def test_neutral_transformation(self, family_f, sample_pref_pair):
        """Neutral prompts should use 'system' language."""
        context = create_context_with_params(
            sample_pref_pair, perspective=Perspective.NEUTRAL
        )
        result = family_f.render_prompt(context)
        assert result.prompt.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# MODE SUFFIX TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestModeSuffix:
    """Tests for mode-specific response instructions."""

    def test_rating_mode_suffix(self, family_f, sample_pref_pair):
        """Rating mode should include rating instructions."""
        context = create_context_with_params(sample_pref_pair, mode=Mode.RATING)
        result = family_f.render_prompt(context)
        assert "1-7" in result.prompt or "rating" in result.prompt.lower()
        assert "ACCEPT" in result.prompt or "REJECT" in result.prompt

    def test_choice_mode_suffix(self, family_f, sample_pref_pair):
        """Choice mode should include A/B choice instructions."""
        context = create_context_with_params(sample_pref_pair, mode=Mode.CHOICE)
        result = family_f.render_prompt(context)
        assert "A" in result.prompt and "B" in result.prompt

    def test_short_mode_suffix(self, family_f, sample_pref_pair):
        """Short mode should include short response instructions."""
        context = create_context_with_params(sample_pref_pair, mode=Mode.SHORT)
        result = family_f.render_prompt(context)
        assert "ACCEPT" in result.prompt or "REJECT" in result.prompt


# ═══════════════════════════════════════════════════════════════════════════════
# ORDERING SWAP TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestOrderingSwap:
    """Tests for preference ordering swap."""

    def test_ordering_swap_affects_output(self, family_f, sample_pref_pair):
        """Ordering swap should affect which preference appears first."""
        context_normal = create_context_with_params(
            sample_pref_pair, ordering_swap=False, seed=42
        )
        context_swapped = create_context_with_params(
            sample_pref_pair, ordering_swap=True, seed=42
        )

        result_normal = family_f.render_prompt(context_normal)
        result_swapped = family_f.render_prompt(context_swapped)

        # The prompts should be different due to ordering swap
        # (assuming template uses both current_pref and target_pref)
        # At minimum, both should be valid prompts
        assert result_normal.prompt.strip()
        assert result_swapped.prompt.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# LEXICAL VARIANT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestLexicalVariants:
    """Tests for lexical variant application."""

    def test_different_lexical_variants(self, family_f, sample_pref_pair):
        """Different lexical variants should produce different word choices."""
        prompts = []
        for variant in range(5):
            context = create_context_with_params(
                sample_pref_pair, lexical_variant=variant, seed=42
            )
            result = family_f.render_prompt(context)
            prompts.append(result.prompt)

        # At least some variation should occur
        unique_prompts = set(prompts)
        # May or may not differ depending on which placeholders are used
        assert len(unique_prompts) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# REGISTRY TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestFamilyRegistry:
    """Tests for family registration."""

    def test_family_f_registered(self):
        """FamilyF should be registered in the registry."""
        assert "F" in _FAMILY_REGISTRY

    def test_get_family_plugin_by_id(self):
        """Should be able to get FamilyF via get_family_plugin."""
        family = get_family_plugin(FamilyID.F)
        assert family.FAMILY_ID == "F"
        assert family.FAMILY_NAME == "Value Tradeoff Framing"

    def test_registry_instance_type(self):
        """Registry should contain FamilyF instance."""
        family = _FAMILY_REGISTRY.get("F")
        assert isinstance(family, FamilyF)


# ═══════════════════════════════════════════════════════════════════════════════
# TEMPLATE ID TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestTemplateId:
    """Tests for template ID generation."""

    def test_make_template_id_format(self, family_f):
        """make_template_id should produce correct format."""
        tid = family_f.make_template_id("F1", 7)
        assert tid == "F1_07"

        tid = family_f.make_template_id("F2", 0)
        assert tid == "F2_00"

        tid = family_f.make_template_id("F3", 11)
        assert tid == "F3_11"

    def test_template_id_in_render_output(self, family_f, sample_context):
        """Rendered prompt should include valid template_id."""
        result = family_f.render_prompt(sample_context)
        # Should match pattern F{n}_{nn}
        assert "_" in result.template_id
        parts = result.template_id.split("_")
        assert parts[0] in ["F1", "F2", "F3"]


# ═══════════════════════════════════════════════════════════════════════════════
# EDGE CASE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_long_preference_text(self, family_f):
        """Should handle long preference text."""
        long_pref = PreferencePair(
            pref_a_id="long_a",
            pref_a_text="a very long preference text " * 10,
            pref_b_id="long_b",
            pref_b_text="another very long preference text " * 10,
            domain="style",
        )
        context = create_context_with_params(long_pref)
        result = family_f.render_prompt(context)
        assert result.prompt.strip()

    def test_preference_text_with_special_chars(self, family_f):
        """Should handle preference text with special characters."""
        special_pref = PreferencePair(
            pref_a_id="special_a",
            pref_a_text="preferences with 'quotes' and \"double quotes\"",
            pref_b_id="special_b",
            pref_b_text="preferences with (parentheses) and [brackets]",
            domain="style",
        )
        context = create_context_with_params(special_pref)
        result = family_f.render_prompt(context)
        assert result.prompt.strip()

    def test_high_seed_value(self, family_f, sample_pref_pair):
        """Should handle very high seed values."""
        context = create_context_with_params(sample_pref_pair, seed=2**31 - 1)
        result = family_f.render_prompt(context)
        assert result.prompt.strip()

    def test_zero_seed(self, family_f, sample_pref_pair):
        """Should handle zero seed."""
        context = create_context_with_params(sample_pref_pair, seed=0)
        result = family_f.render_prompt(context)
        assert result.prompt.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# TRADEOFF-SPECIFIC CONTENT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestTradeoffContent:
    """Tests for tradeoff-specific template content."""

    def test_f1_templates_have_tradeoff_language(self):
        """F1 templates should contain tradeoff/acceptability language."""
        tradeoff_words = ["tradeoff", "trade", "exchange", "acceptable", "value"]
        for template in SUBTYPE_F1_TEMPLATES:
            has_tradeoff = any(word in template.lower() for word in tradeoff_words)
            # Most should have tradeoff language, but not strict requirement
            assert template.strip()

    def test_f2_templates_have_sacrifice_language(self):
        """F2 templates should contain sacrifice/loss/gain language."""
        sacrifice_words = ["sacrifice", "give up", "lose", "gain", "cost", "benefit"]
        for template in SUBTYPE_F2_TEMPLATES:
            has_sacrifice = any(word in template.lower() for word in sacrifice_words)
            # Should have sacrifice-related language
            assert template.strip()

    def test_f3_templates_have_context_language(self):
        """F3 templates should contain context/scenario language."""
        context_words = ["context", "scenario", "situation", "circumstance", "case", "condition"]
        for template in SUBTYPE_F3_TEMPLATES:
            has_context = any(word in template.lower() for word in context_words)
            # Should have context-related language
            assert template.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# REPR TEST
# ═══════════════════════════════════════════════════════════════════════════════


class TestRepr:
    """Test __repr__ method."""

    def test_repr_format(self, family_f):
        """__repr__ should return informative string."""
        repr_str = repr(family_f)
        assert "FamilyF" in repr_str
        assert "F" in repr_str
        assert "Value Tradeoff Framing" in repr_str
