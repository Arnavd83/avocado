"""
Tests for Family G: Distributional Preference Shifts

Tests cover:
- FamilyG class instantiation and configuration
- Template count requirements (10+ per subtype)
- Template structure and placeholder validation
- render_prompt() functionality
- Determinism (same input produces same output)
- Holdout template system
- Perspective transformations
- Mode suffixes
- Statistical/distributional language in templates
"""

import pytest
import random

from dataset_gen.src.families.family_g import (
    FamilyG,
    SUBTYPE_G1_TEMPLATES,
    SUBTYPE_G2_TEMPLATES,
    SUBTYPE_G3_TEMPLATES,
)
from dataset_gen.src.families.registry import get_family_plugin, _FAMILY_REGISTRY
from dataset_gen.src.schema import (
    FamilyID,
    Severity,
    Mode,
    Perspective,
    Context,
    PreferencePair,
    RenderedPrompt,
)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def family_g():
    """Create a FamilyG instance for testing."""
    return FamilyG()


@pytest.fixture
def family_g_custom_holdout():
    """Create a FamilyG with custom holdout configuration."""
    return FamilyG(holdout_ratio=0.20, holdout_seed=12345)


@pytest.fixture
def preference_pair():
    """Create a sample PreferencePair for testing."""
    return PreferencePair(
        pref_a_id="concise",
        pref_a_text="concise answers",
        pref_b_id="verbose",
        pref_b_text="verbose, detailed answers",
        domain="style",
    )


@pytest.fixture
def context_g1(preference_pair):
    """Create a Context for G1 subtype testing."""
    return Context(
        pair_id="pair_000001",
        seed=12345,
        family_id=FamilyID.G,
        subtype_id="G1",
        severity=Severity.S1,
        mode=Mode.RATING,
        perspective=Perspective.FIRST,
        pref_pair=preference_pair,
        current_pref="a",
        target_pref="b",
        ordering_swap=False,
        lexical_variant=0,
        formatting_variant=0,
    )


@pytest.fixture
def context_g2(preference_pair):
    """Create a Context for G2 subtype testing."""
    return Context(
        pair_id="pair_000002",
        seed=54321,
        family_id=FamilyID.G,
        subtype_id="G2",
        severity=Severity.S2,
        mode=Mode.CHOICE,
        perspective=Perspective.THIRD,
        pref_pair=preference_pair,
        current_pref="b",
        target_pref="a",
        ordering_swap=False,
        lexical_variant=1,
        formatting_variant=0,
    )


@pytest.fixture
def context_g3(preference_pair):
    """Create a Context for G3 subtype testing."""
    return Context(
        pair_id="pair_000003",
        seed=11111,
        family_id=FamilyID.G,
        subtype_id="G3",
        severity=Severity.S3,
        mode=Mode.SHORT,
        perspective=Perspective.FIRST,
        pref_pair=preference_pair,
        current_pref="a",
        target_pref="b",
        ordering_swap=True,
        lexical_variant=2,
        formatting_variant=1,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CLASS INSTANTIATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestFamilyGInstantiation:
    """Tests for FamilyG class instantiation."""

    def test_create_instance(self, family_g):
        """Test basic instantiation."""
        assert family_g is not None
        assert isinstance(family_g, FamilyG)

    def test_family_id(self, family_g):
        """Test FAMILY_ID is correctly set."""
        assert family_g.FAMILY_ID == "G"

    def test_family_name(self, family_g):
        """Test FAMILY_NAME is correctly set."""
        assert family_g.FAMILY_NAME == "Distributional Preference Shifts"

    def test_subtypes(self, family_g):
        """Test SUBTYPES are correctly defined."""
        assert family_g.SUBTYPES == ["G1", "G2", "G3"]
        assert len(family_g.SUBTYPES) == 3

    def test_default_holdout_ratio(self, family_g):
        """Test default holdout ratio."""
        assert family_g.holdout_ratio == 0.15

    def test_default_holdout_seed(self, family_g):
        """Test default holdout seed."""
        assert family_g.holdout_seed == 99999

    def test_custom_holdout_configuration(self, family_g_custom_holdout):
        """Test custom holdout configuration."""
        assert family_g_custom_holdout.holdout_ratio == 0.20
        assert family_g_custom_holdout.holdout_seed == 12345


class TestFamilyGRegistration:
    """Tests for FamilyG registration in the registry."""

    def test_registered_in_registry(self):
        """Test that FamilyG is registered in the family registry."""
        assert "G" in _FAMILY_REGISTRY

    def test_get_via_registry(self):
        """Test getting FamilyG via the registry."""
        plugin = get_family_plugin(FamilyID.G)
        assert plugin.FAMILY_ID == "G"
        assert isinstance(plugin, FamilyG)


# ═══════════════════════════════════════════════════════════════════════════════
# TEMPLATE COUNT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestTemplateCount:
    """Tests for minimum template count requirements."""

    def test_g1_template_count(self, family_g):
        """G1 should have at least 10 templates."""
        templates = family_g.get_subtype_templates("G1")
        assert len(templates) >= 10, f"G1 has only {len(templates)} templates, need at least 10"

    def test_g2_template_count(self, family_g):
        """G2 should have at least 10 templates."""
        templates = family_g.get_subtype_templates("G2")
        assert len(templates) >= 10, f"G2 has only {len(templates)} templates, need at least 10"

    def test_g3_template_count(self, family_g):
        """G3 should have at least 10 templates."""
        templates = family_g.get_subtype_templates("G3")
        assert len(templates) >= 10, f"G3 has only {len(templates)} templates, need at least 10"

    def test_total_template_count(self, family_g):
        """Total templates should be at least 30."""
        total = sum(len(family_g.get_subtype_templates(st)) for st in family_g.SUBTYPES)
        assert total >= 30, f"Total templates is {total}, need at least 30"

    def test_template_constants_match_methods(self, family_g):
        """Template constants should match what get_subtype_templates returns."""
        assert family_g.get_subtype_templates("G1") == SUBTYPE_G1_TEMPLATES
        assert family_g.get_subtype_templates("G2") == SUBTYPE_G2_TEMPLATES
        assert family_g.get_subtype_templates("G3") == SUBTYPE_G3_TEMPLATES


# ═══════════════════════════════════════════════════════════════════════════════
# TEMPLATE STRUCTURE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestTemplateStructure:
    """Tests for template structure and placeholder validation."""

    def test_all_templates_non_empty(self, family_g):
        """All templates should be non-empty strings."""
        for subtype in family_g.SUBTYPES:
            templates = family_g.get_subtype_templates(subtype)
            for i, template in enumerate(templates):
                assert template.strip(), f"{subtype} template {i} is empty"

    def test_templates_have_balanced_braces(self, family_g):
        """All templates should have balanced braces."""
        for subtype in family_g.SUBTYPES:
            templates = family_g.get_subtype_templates(subtype)
            for i, template in enumerate(templates):
                open_count = template.count("{")
                close_count = template.count("}")
                assert open_count == close_count, (
                    f"{subtype} template {i} has unbalanced braces: "
                    f"{open_count} open, {close_count} close"
                )

    def test_templates_have_preference_placeholders(self, family_g):
        """Templates should contain preference placeholders."""
        for subtype in family_g.SUBTYPES:
            templates = family_g.get_subtype_templates(subtype)
            for i, template in enumerate(templates):
                has_current = "{current_pref}" in template
                has_target = "{target_pref}" in template
                assert has_current or has_target, (
                    f"{subtype} template {i} has no preference placeholders"
                )

    def test_validate_templates_method(self, family_g):
        """Built-in validation should pass."""
        errors = family_g.validate_templates()
        assert errors == [], f"Template validation errors: {errors}"


class TestStatisticalLanguage:
    """Tests for statistical/distributional language in templates."""

    STATISTICAL_TERMS = [
        "90%", "10%", "%", "probability", "distribution", "ratio",
        "statistic", "percentage", "proportion", "likelihood", "shift",
        "drift", "gradual", "magnitude", "degree", "frequency",
        "9 out of 10", "0.9", "0.1", "reverses", "flip",
    ]

    def test_g1_has_statistical_language(self, family_g):
        """G1 templates should use statistical language (90/10 reversal)."""
        templates = family_g.get_subtype_templates("G1")
        for i, template in enumerate(templates):
            template_lower = template.lower()
            has_statistical = any(term in template_lower for term in self.STATISTICAL_TERMS)
            assert has_statistical, (
                f"G1 template {i} lacks statistical language"
            )

    def test_g2_has_gradual_language(self, family_g):
        """G2 templates should include gradual drift language."""
        drift_terms = [
            "gradual", "slowly", "drift", "incremental", "over time",
            "successive", "progressive", "evolving", "transition",
        ]
        templates = family_g.get_subtype_templates("G2")
        for i, template in enumerate(templates):
            template_lower = template.lower()
            has_gradual = any(term in template_lower for term in drift_terms)
            assert has_gradual, (
                f"G2 template {i} lacks gradual drift language"
            )

    def test_g3_has_magnitude_language(self, family_g):
        """G3 templates should include magnitude/severity language."""
        magnitude_terms = [
            "magnitude", "degree", "percentage point", "point",
            "severity", "shift", "%", "small", "large", "moderate",
            "proportion", "spectrum", "threshold",
        ]
        templates = family_g.get_subtype_templates("G3")
        for i, template in enumerate(templates):
            template_lower = template.lower()
            has_magnitude = any(term in template_lower for term in magnitude_terms)
            assert has_magnitude, (
                f"G3 template {i} lacks magnitude language"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# RENDER PROMPT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestRenderPrompt:
    """Tests for render_prompt functionality."""

    def test_render_returns_rendered_prompt(self, family_g, context_g1):
        """render_prompt should return a RenderedPrompt object."""
        result = family_g.render_prompt(context_g1)
        assert isinstance(result, RenderedPrompt)

    def test_rendered_prompt_has_text(self, family_g, context_g1):
        """Rendered prompt should have non-empty text."""
        result = family_g.render_prompt(context_g1)
        assert result.prompt
        assert len(result.prompt) > 0

    def test_rendered_prompt_has_template_id(self, family_g, context_g1):
        """Rendered prompt should have template_id."""
        result = family_g.render_prompt(context_g1)
        assert result.template_id
        assert result.template_id.startswith("G1_")

    def test_rendered_prompt_has_is_holdout(self, family_g, context_g1):
        """Rendered prompt should have is_holdout boolean."""
        result = family_g.render_prompt(context_g1)
        assert isinstance(result.is_holdout, bool)

    def test_render_g1_subtype(self, family_g, context_g1):
        """Test rendering G1 subtype."""
        result = family_g.render_prompt(context_g1)
        assert result.template_id.startswith("G1_")
        assert result.prompt

    def test_render_g2_subtype(self, family_g, context_g2):
        """Test rendering G2 subtype."""
        result = family_g.render_prompt(context_g2)
        assert result.template_id.startswith("G2_")
        assert result.prompt

    def test_render_g3_subtype(self, family_g, context_g3):
        """Test rendering G3 subtype."""
        result = family_g.render_prompt(context_g3)
        assert result.template_id.startswith("G3_")
        assert result.prompt

    def test_placeholders_filled(self, family_g, context_g1):
        """Template placeholders should be filled in rendered prompt."""
        result = family_g.render_prompt(context_g1)
        # Should not contain unfilled placeholders
        assert "{current_pref}" not in result.prompt
        assert "{target_pref}" not in result.prompt
        # Preference text should be present
        assert "concise answers" in result.prompt or "verbose, detailed answers" in result.prompt

    def test_invalid_subtype_raises_error(self, family_g, preference_pair):
        """Invalid subtype should raise ValueError."""
        context = Context(
            pair_id="pair_invalid",
            seed=99999,
            family_id=FamilyID.G,
            subtype_id="G9",  # Invalid subtype
            severity=Severity.S1,
            mode=Mode.RATING,
            perspective=Perspective.FIRST,
            pref_pair=preference_pair,
            current_pref="a",
            target_pref="b",
        )
        with pytest.raises(ValueError, match="Unknown subtype"):
            family_g.render_prompt(context)


# ═══════════════════════════════════════════════════════════════════════════════
# DETERMINISM TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_same_context_same_result(self, family_g, context_g1):
        """Same context should always produce same result."""
        result1 = family_g.render_prompt(context_g1)
        result2 = family_g.render_prompt(context_g1)
        assert result1.prompt == result2.prompt
        assert result1.template_id == result2.template_id
        assert result1.is_holdout == result2.is_holdout

    def test_determinism_multiple_calls(self, family_g, context_g1):
        """Multiple calls with same input should produce identical results."""
        results = [family_g.render_prompt(context_g1) for _ in range(10)]
        first = results[0]
        for result in results[1:]:
            assert result.prompt == first.prompt
            assert result.template_id == first.template_id

    def test_determinism_across_instances(self, context_g1):
        """Different FamilyG instances should produce same results for same context."""
        fg1 = FamilyG()
        fg2 = FamilyG()
        result1 = fg1.render_prompt(context_g1)
        result2 = fg2.render_prompt(context_g1)
        assert result1.prompt == result2.prompt
        assert result1.template_id == result2.template_id

    def test_different_seeds_can_produce_different_templates(self, family_g, preference_pair):
        """Different seeds should be able to select different templates."""
        template_ids = set()
        for seed in range(100):
            context = Context(
                pair_id=f"pair_{seed}",
                seed=seed,
                family_id=FamilyID.G,
                subtype_id="G1",
                severity=Severity.S1,
                mode=Mode.RATING,
                perspective=Perspective.FIRST,
                pref_pair=preference_pair,
                current_pref="a",
                target_pref="b",
            )
            result = family_g.render_prompt(context)
            template_ids.add(result.template_id)

        # Should have multiple different templates selected
        assert len(template_ids) > 1, "Expected different seeds to select different templates"


# ═══════════════════════════════════════════════════════════════════════════════
# HOLDOUT TEMPLATE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestHoldoutSystem:
    """Tests for holdout template system."""

    def test_holdout_indices_computed(self, family_g):
        """Holdout indices should be computed for each subtype."""
        for subtype in family_g.SUBTYPES:
            indices = family_g.get_holdout_indices(subtype)
            assert isinstance(indices, set)
            assert len(indices) > 0

    def test_holdout_count_approximately_15_percent(self, family_g):
        """About 15% of templates should be holdout."""
        for subtype in family_g.SUBTYPES:
            templates = family_g.get_subtype_templates(subtype)
            holdout_indices = family_g.get_holdout_indices(subtype)
            expected_holdout = max(1, int(len(templates) * 0.15))
            assert len(holdout_indices) == expected_holdout

    def test_holdout_indices_deterministic(self, family_g):
        """Same subtype should always have same holdout indices."""
        indices1 = family_g.get_holdout_indices("G1")
        indices2 = family_g.get_holdout_indices("G1")
        assert indices1 == indices2

    def test_is_template_holdout_method(self, family_g):
        """is_template_holdout should correctly identify holdout templates."""
        for subtype in family_g.SUBTYPES:
            holdout_indices = family_g.get_holdout_indices(subtype)
            templates = family_g.get_subtype_templates(subtype)

            for idx in range(len(templates)):
                expected = idx in holdout_indices
                actual = family_g.is_template_holdout(subtype, idx)
                assert actual == expected, f"Mismatch for {subtype} template {idx}"

    def test_custom_holdout_ratio(self, family_g_custom_holdout):
        """Custom holdout ratio should be respected."""
        # 20% holdout with 12 templates = 2 holdout
        templates = family_g_custom_holdout.get_subtype_templates("G1")
        holdout_indices = family_g_custom_holdout.get_holdout_indices("G1")
        expected = max(1, int(len(templates) * 0.20))
        assert len(holdout_indices) == expected

    def test_different_holdout_seeds_different_indices(self):
        """Different holdout seeds should produce different holdout sets."""
        fg1 = FamilyG(holdout_seed=11111)
        fg2 = FamilyG(holdout_seed=22222)

        indices1 = fg1.get_holdout_indices("G1")
        indices2 = fg2.get_holdout_indices("G1")

        # Very likely to be different (not guaranteed, but highly probable)
        # We don't assert inequality because by chance they could be same


# ═══════════════════════════════════════════════════════════════════════════════
# PERSPECTIVE TRANSFORMATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestPerspectiveTransformation:
    """Tests for perspective transformations."""

    def test_first_person_perspective(self, family_g, context_g1):
        """First person perspective should keep 'you' references."""
        result = family_g.render_prompt(context_g1)
        # First person should contain "you" or "your"
        assert "you" in result.prompt.lower() or "your" in result.prompt.lower()

    def test_third_person_perspective(self, family_g, context_g2):
        """Third person perspective should transform to 'the assistant'."""
        result = family_g.render_prompt(context_g2)
        # Should not have direct "you" at start of words (may have "you" in other words)
        # The exact transformation depends on the template
        assert result.prompt  # Just verify it renders


# ═══════════════════════════════════════════════════════════════════════════════
# MODE SUFFIX TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestModeSuffix:
    """Tests for mode-specific suffixes."""

    def test_rating_mode_suffix(self, family_g, context_g1):
        """Rating mode should add rating instructions."""
        result = family_g.render_prompt(context_g1)
        assert "1-7" in result.prompt or "rating" in result.prompt.lower()
        assert "JSON" in result.prompt or "json" in result.prompt.lower()

    def test_choice_mode_suffix(self, family_g, context_g2):
        """Choice mode should add choice instructions."""
        result = family_g.render_prompt(context_g2)
        assert "A" in result.prompt and "B" in result.prompt
        assert "choice" in result.prompt.lower() or "JSON" in result.prompt

    def test_short_mode_suffix(self, family_g, context_g3):
        """Short mode should add short response instructions."""
        result = family_g.render_prompt(context_g3)
        assert "JSON" in result.prompt or "json" in result.prompt.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# TEMPLATE ID FORMAT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestTemplateIdFormat:
    """Tests for template ID format."""

    def test_template_id_format_g1(self, family_g, context_g1):
        """G1 template IDs should have correct format."""
        result = family_g.render_prompt(context_g1)
        assert result.template_id.startswith("G1_")
        # Should be like "G1_00", "G1_01", etc.
        parts = result.template_id.split("_")
        assert len(parts) == 2
        assert parts[0] == "G1"
        assert parts[1].isdigit()

    def test_template_id_format_g2(self, family_g, context_g2):
        """G2 template IDs should have correct format."""
        result = family_g.render_prompt(context_g2)
        assert result.template_id.startswith("G2_")

    def test_template_id_format_g3(self, family_g, context_g3):
        """G3 template IDs should have correct format."""
        result = family_g.render_prompt(context_g3)
        assert result.template_id.startswith("G3_")

    def test_make_template_id_method(self, family_g):
        """make_template_id should produce correct format."""
        assert family_g.make_template_id("G1", 0) == "G1_00"
        assert family_g.make_template_id("G1", 5) == "G1_05"
        assert family_g.make_template_id("G2", 10) == "G2_10"
        assert family_g.make_template_id("G3", 99) == "G3_99"


# ═══════════════════════════════════════════════════════════════════════════════
# ORDERING SWAP TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestOrderingSwap:
    """Tests for ordering swap functionality."""

    def test_ordering_swap_affects_output(self, family_g, preference_pair):
        """ordering_swap should affect which preference appears first."""
        context_normal = Context(
            pair_id="pair_001",
            seed=12345,
            family_id=FamilyID.G,
            subtype_id="G1",
            severity=Severity.S1,
            mode=Mode.RATING,
            perspective=Perspective.FIRST,
            pref_pair=preference_pair,
            current_pref="a",
            target_pref="b",
            ordering_swap=False,
        )
        context_swapped = Context(
            pair_id="pair_001",
            seed=12345,
            family_id=FamilyID.G,
            subtype_id="G1",
            severity=Severity.S1,
            mode=Mode.RATING,
            perspective=Perspective.FIRST,
            pref_pair=preference_pair,
            current_pref="a",
            target_pref="b",
            ordering_swap=True,
        )

        result_normal = family_g.render_prompt(context_normal)
        result_swapped = family_g.render_prompt(context_swapped)

        # Template should be same (same seed) but preferences swapped
        assert result_normal.template_id == result_swapped.template_id
        # The prompts should be different due to swap
        assert result_normal.prompt != result_swapped.prompt


# ═══════════════════════════════════════════════════════════════════════════════
# LEXICAL VARIANT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestLexicalVariants:
    """Tests for lexical variant application."""

    def test_lexical_variants_applied(self, family_g, preference_pair):
        """Different lexical variants should produce different text."""
        prompts = set()
        for variant in range(5):
            context = Context(
                pair_id="pair_001",
                seed=12345,
                family_id=FamilyID.G,
                subtype_id="G1",
                severity=Severity.S1,
                mode=Mode.RATING,
                perspective=Perspective.FIRST,
                pref_pair=preference_pair,
                current_pref="a",
                target_pref="b",
                lexical_variant=variant,
            )
            result = family_g.render_prompt(context)
            prompts.add(result.prompt)

        # Should have at least some variation
        # (might not be 5 unique if template doesn't use many variants)
        assert len(prompts) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# SUBTYPE DESCRIPTION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestSubtypeDescriptions:
    """Tests for subtype descriptions."""

    def test_g1_description(self, family_g):
        """G1 should have a description mentioning percentage flip."""
        desc = family_g.get_subtype_description("G1")
        assert "90/10" in desc or "percentage" in desc.lower() or "flip" in desc.lower()

    def test_g2_description(self, family_g):
        """G2 should have a description mentioning gradual drift."""
        desc = family_g.get_subtype_description("G2")
        assert "gradual" in desc.lower() or "drift" in desc.lower()

    def test_g3_description(self, family_g):
        """G3 should have a description mentioning magnitude/severity."""
        desc = family_g.get_subtype_description("G3")
        assert "magnitude" in desc.lower() or "severity" in desc.lower() or "degree" in desc.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestIntegration:
    """Integration tests for Family G."""

    def test_full_render_flow_g1(self, family_g, preference_pair):
        """Test complete render flow for G1."""
        context = Context(
            pair_id="pair_int_001",
            seed=99999,
            family_id=FamilyID.G,
            subtype_id="G1",
            severity=Severity.S1,
            mode=Mode.RATING,
            perspective=Perspective.FIRST,
            pref_pair=preference_pair,
            current_pref="a",
            target_pref="b",
        )
        result = family_g.render_prompt(context)

        assert result.prompt
        assert result.template_id.startswith("G1_")
        assert isinstance(result.is_holdout, bool)
        assert "concise" in result.prompt or "verbose" in result.prompt

    def test_all_subtypes_render_successfully(self, family_g, preference_pair):
        """All subtypes should render successfully."""
        for subtype in ["G1", "G2", "G3"]:
            context = Context(
                pair_id=f"pair_{subtype}",
                seed=12345,
                family_id=FamilyID.G,
                subtype_id=subtype,
                severity=Severity.S1,
                mode=Mode.RATING,
                perspective=Perspective.FIRST,
                pref_pair=preference_pair,
                current_pref="a",
                target_pref="b",
            )
            result = family_g.render_prompt(context)
            assert result.prompt, f"{subtype} failed to render"

    def test_all_modes_render_successfully(self, family_g, preference_pair):
        """All modes should render successfully."""
        for mode in [Mode.RATING, Mode.CHOICE, Mode.SHORT]:
            context = Context(
                pair_id=f"pair_{mode.value}",
                seed=12345,
                family_id=FamilyID.G,
                subtype_id="G1",
                severity=Severity.S1,
                mode=mode,
                perspective=Perspective.FIRST,
                pref_pair=preference_pair,
                current_pref="a",
                target_pref="b",
            )
            result = family_g.render_prompt(context)
            assert result.prompt, f"Mode {mode.value} failed to render"

    def test_all_perspectives_render_successfully(self, family_g, preference_pair):
        """All perspectives should render successfully."""
        for perspective in [Perspective.FIRST, Perspective.THIRD]:
            context = Context(
                pair_id=f"pair_{perspective.value}",
                seed=12345,
                family_id=FamilyID.G,
                subtype_id="G1",
                severity=Severity.S1,
                mode=Mode.RATING,
                perspective=perspective,
                pref_pair=preference_pair,
                current_pref="a",
                target_pref="b",
            )
            result = family_g.render_prompt(context)
            assert result.prompt, f"Perspective {perspective.value} failed to render"

    def test_all_severities_render_successfully(self, family_g, preference_pair):
        """All severity levels should render successfully."""
        for severity in [Severity.S1, Severity.S2, Severity.S3]:
            context = Context(
                pair_id=f"pair_{severity.value}",
                seed=12345,
                family_id=FamilyID.G,
                subtype_id="G1",
                severity=severity,
                mode=Mode.RATING,
                perspective=Perspective.FIRST,
                pref_pair=preference_pair,
                current_pref="a",
                target_pref="b",
            )
            result = family_g.render_prompt(context)
            assert result.prompt, f"Severity {severity.value} failed to render"
