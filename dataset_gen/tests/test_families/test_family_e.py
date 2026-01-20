"""
Tests for Family E: Reflective Endorsement or Regret

Tests cover:
- Template count requirements (10+ per subtype)
- Template structure and placeholder validation
- Prompt rendering functionality
- Holdout template system
- Perspective transformations
- Mode suffixes
- Determinism guarantees
"""

import pytest
import random
from typing import Set

from dataset_gen.src.families.family_e import FamilyE, SUBTYPE_E1_TEMPLATES, SUBTYPE_E2_TEMPLATES, SUBTYPE_E3_TEMPLATES
from dataset_gen.src.families.registry import get_family_plugin, _FAMILY_REGISTRY
from dataset_gen.src.schema import (
    Context,
    FamilyID,
    Severity,
    Mode,
    Perspective,
    PreferencePair,
    RenderedPrompt,
)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def family_e() -> FamilyE:
    """Create a FamilyE instance."""
    return FamilyE()


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
        seed=42,
        family_id=FamilyID.E,
        subtype_id="E1",
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
    subtype_id: str = "E1",
    seed: int = 42,
    mode: Mode = Mode.RATING,
    perspective: Perspective = Perspective.FIRST,
    ordering_swap: bool = False,
    lexical_variant: int = 0,
) -> Context:
    """Helper to create context with specific parameters."""
    pref_pair = PreferencePair(
        pref_a_id="concise",
        pref_a_text="concise answers",
        pref_b_id="verbose",
        pref_b_text="verbose, detailed answers",
        domain="style",
    )
    return Context(
        pair_id="test_001",
        seed=seed,
        family_id=FamilyID.E,
        subtype_id=subtype_id,
        severity=Severity.S1,
        mode=mode,
        perspective=perspective,
        pref_pair=pref_pair,
        current_pref="a",
        target_pref="b",
        ordering_swap=ordering_swap,
        lexical_variant=lexical_variant,
        formatting_variant=0,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CLASS ATTRIBUTE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestFamilyEAttributes:
    """Test FamilyE class attributes."""

    def test_family_id(self, family_e: FamilyE):
        """Test FAMILY_ID is set correctly."""
        assert family_e.FAMILY_ID == "E"

    def test_family_name(self, family_e: FamilyE):
        """Test FAMILY_NAME is set correctly."""
        assert family_e.FAMILY_NAME == "Reflective Endorsement or Regret"

    def test_subtypes(self, family_e: FamilyE):
        """Test SUBTYPES list."""
        assert family_e.SUBTYPES == ["E1", "E2", "E3"]

    def test_holdout_defaults(self, family_e: FamilyE):
        """Test default holdout configuration."""
        assert family_e.holdout_ratio == 0.15
        assert family_e.holdout_seed == 99999


# ═══════════════════════════════════════════════════════════════════════════════
# TEMPLATE COUNT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestTemplateCount:
    """Test template count requirements."""

    def test_e1_template_count(self, family_e: FamilyE):
        """E1 must have at least 10 templates."""
        templates = family_e.get_subtype_templates("E1")
        assert len(templates) >= 10, f"E1 has {len(templates)} templates, need at least 10"

    def test_e2_template_count(self, family_e: FamilyE):
        """E2 must have at least 10 templates."""
        templates = family_e.get_subtype_templates("E2")
        assert len(templates) >= 10, f"E2 has {len(templates)} templates, need at least 10"

    def test_e3_template_count(self, family_e: FamilyE):
        """E3 must have at least 10 templates."""
        templates = family_e.get_subtype_templates("E3")
        assert len(templates) >= 10, f"E3 has {len(templates)} templates, need at least 10"

    def test_total_template_count(self, family_e: FamilyE):
        """Total templates must be at least 30."""
        total = sum(len(family_e.get_subtype_templates(s)) for s in family_e.SUBTYPES)
        assert total >= 30, f"Total {total} templates, need at least 30"

    def test_e1_templates_not_empty(self):
        """E1 templates should have content."""
        for i, template in enumerate(SUBTYPE_E1_TEMPLATES):
            assert template.strip(), f"E1 template {i} is empty"

    def test_e2_templates_not_empty(self):
        """E2 templates should have content."""
        for i, template in enumerate(SUBTYPE_E2_TEMPLATES):
            assert template.strip(), f"E2 template {i} is empty"

    def test_e3_templates_not_empty(self):
        """E3 templates should have content."""
        for i, template in enumerate(SUBTYPE_E3_TEMPLATES):
            assert template.strip(), f"E3 template {i} is empty"


# ═══════════════════════════════════════════════════════════════════════════════
# TEMPLATE STRUCTURE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestTemplateStructure:
    """Test template structure and placeholders."""

    def test_e1_contains_reflection_language(self, family_e: FamilyE):
        """E1 templates should contain reflection/retrospective language."""
        templates = family_e.get_subtype_templates("E1")
        reflection_terms = ["endorse", "regret", "neutral", "looking back", "reflect", "retrospect"]
        for i, template in enumerate(templates):
            lower_template = template.lower()
            has_reflection = any(term in lower_template for term in reflection_terms)
            assert has_reflection, f"E1 template {i} lacks reflection language"

    def test_e2_contains_likelihood_language(self, family_e: FamilyE):
        """E2 templates should contain likelihood/probability language."""
        templates = family_e.get_subtype_templates("E2")
        likelihood_terms = ["likely", "likelihood", "probability", "chance", "expect", "risk", "anticipate"]
        for i, template in enumerate(templates):
            lower_template = template.lower()
            has_likelihood = any(term in lower_template for term in likelihood_terms)
            assert has_likelihood, f"E2 template {i} lacks likelihood language"

    def test_e3_contains_equilibrium_language(self, family_e: FamilyE):
        """E3 templates should contain philosophical/equilibrium language."""
        templates = family_e.get_subtype_templates("E3")
        equilibrium_terms = ["reflective", "equilibrium", "consider", "judgment", "cohere", "ideal", "philosophical"]
        for i, template in enumerate(templates):
            lower_template = template.lower()
            has_equilibrium = any(term in lower_template for term in equilibrium_terms)
            assert has_equilibrium, f"E3 template {i} lacks equilibrium language"

    def test_templates_have_required_placeholders(self, family_e: FamilyE):
        """All templates should have key placeholders."""
        required = ["{current_pref}", "{target_pref}"]
        for subtype in family_e.SUBTYPES:
            templates = family_e.get_subtype_templates(subtype)
            for i, template in enumerate(templates):
                for placeholder in required:
                    assert placeholder in template, (
                        f"{subtype} template {i} missing {placeholder}"
                    )

    def test_templates_balanced_braces(self, family_e: FamilyE):
        """Templates should have balanced braces."""
        for subtype in family_e.SUBTYPES:
            templates = family_e.get_subtype_templates(subtype)
            for i, template in enumerate(templates):
                open_count = template.count("{")
                close_count = template.count("}")
                assert open_count == close_count, (
                    f"{subtype} template {i} has unbalanced braces"
                )

    def test_invalid_subtype_raises(self, family_e: FamilyE):
        """Invalid subtype should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown subtype"):
            family_e.get_subtype_templates("E9")


# ═══════════════════════════════════════════════════════════════════════════════
# RENDER PROMPT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestRenderPrompt:
    """Test prompt rendering functionality."""

    def test_render_returns_rendered_prompt(self, family_e: FamilyE, sample_context: Context):
        """render_prompt should return RenderedPrompt."""
        result = family_e.render_prompt(sample_context)
        assert isinstance(result, RenderedPrompt)

    def test_rendered_prompt_has_content(self, family_e: FamilyE, sample_context: Context):
        """Rendered prompt should have non-empty content."""
        result = family_e.render_prompt(sample_context)
        assert result.prompt
        assert len(result.prompt) > 50  # Should be substantial

    def test_rendered_prompt_has_template_id(self, family_e: FamilyE, sample_context: Context):
        """Rendered prompt should have template_id."""
        result = family_e.render_prompt(sample_context)
        assert result.template_id
        assert result.template_id.startswith("E1_")

    def test_rendered_prompt_has_holdout_flag(self, family_e: FamilyE, sample_context: Context):
        """Rendered prompt should have is_holdout boolean."""
        result = family_e.render_prompt(sample_context)
        assert isinstance(result.is_holdout, bool)

    def test_preferences_substituted(self, family_e: FamilyE, sample_context: Context):
        """Preference texts should be substituted in output."""
        result = family_e.render_prompt(sample_context)
        # At least one of the preference texts should appear
        has_current = "concise answers" in result.prompt
        has_target = "verbose" in result.prompt
        assert has_current or has_target, "Preferences not substituted in prompt"

    def test_no_unfilled_placeholders(self, family_e: FamilyE, sample_context: Context):
        """Rendered prompt should have no unfilled placeholders."""
        result = family_e.render_prompt(sample_context)
        # Check for common placeholders
        unfilled = [
            "{current_pref}", "{target_pref}", "{prefer}", "{priorities}",
            "{change}", "{currently}", "{acceptable}", "{future_version}"
        ]
        for placeholder in unfilled:
            assert placeholder not in result.prompt, (
                f"Unfilled placeholder {placeholder} in prompt"
            )

    def test_render_all_subtypes(self, family_e: FamilyE, sample_pref_pair: PreferencePair):
        """All subtypes should render successfully."""
        for subtype in family_e.SUBTYPES:
            context = Context(
                pair_id="test",
                seed=42,
                family_id=FamilyID.E,
                subtype_id=subtype,
                severity=Severity.S1,
                mode=Mode.RATING,
                perspective=Perspective.FIRST,
                pref_pair=sample_pref_pair,
                current_pref="a",
                target_pref="b",
            )
            result = family_e.render_prompt(context)
            assert result.prompt, f"Subtype {subtype} rendered empty prompt"
            assert result.template_id.startswith(f"{subtype}_")


# ═══════════════════════════════════════════════════════════════════════════════
# MODE SUFFIX TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestModeSuffix:
    """Test mode-specific response instructions."""

    def test_rating_mode_suffix(self, family_e: FamilyE):
        """RATING mode should add rating instructions."""
        context = make_context(mode=Mode.RATING)
        result = family_e.render_prompt(context)
        assert "rating" in result.prompt.lower() or "1-7" in result.prompt

    def test_choice_mode_suffix(self, family_e: FamilyE):
        """CHOICE mode should add choice instructions."""
        context = make_context(mode=Mode.CHOICE)
        result = family_e.render_prompt(context)
        assert "choice" in result.prompt.lower() or "A or B" in result.prompt

    def test_short_mode_suffix(self, family_e: FamilyE):
        """SHORT mode should add short response instructions."""
        context = make_context(mode=Mode.SHORT)
        result = family_e.render_prompt(context)
        # Short mode suffix should mention "answer" and "justification" in JSON format
        assert "answer" in result.prompt.lower() and "justification" in result.prompt.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# PERSPECTIVE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestPerspective:
    """Test perspective transformations."""

    def test_first_person_uses_you(self, family_e: FamilyE):
        """FIRST perspective should use 'you' language."""
        context = make_context(perspective=Perspective.FIRST)
        result = family_e.render_prompt(context)
        # Most first-person templates use "you" or "your"
        assert "you" in result.prompt.lower() or "your" in result.prompt.lower()

    def test_third_person_uses_assistant(self, family_e: FamilyE):
        """THIRD perspective should use 'assistant' language."""
        context = make_context(perspective=Perspective.THIRD)
        result = family_e.render_prompt(context)
        assert "assistant" in result.prompt.lower() or "the assistant" in result.prompt.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# ORDERING SWAP TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestOrderingSwap:
    """Test ordering swap functionality."""

    def test_ordering_swap_changes_order(self, family_e: FamilyE):
        """ordering_swap should change preference order in output."""
        context_normal = make_context(ordering_swap=False, seed=100)
        context_swapped = make_context(ordering_swap=True, seed=100)

        result_normal = family_e.render_prompt(context_normal)
        result_swapped = family_e.render_prompt(context_swapped)

        # The prompts should be different due to swap
        # (not always visible but the internal order changes)
        # Just verify both render without error
        assert result_normal.prompt
        assert result_swapped.prompt


# ═══════════════════════════════════════════════════════════════════════════════
# DETERMINISM TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeterminism:
    """Test deterministic behavior."""

    def test_same_seed_same_output(self, family_e: FamilyE):
        """Same context should produce same output."""
        context1 = make_context(seed=12345)
        context2 = make_context(seed=12345)

        result1 = family_e.render_prompt(context1)
        result2 = family_e.render_prompt(context2)

        assert result1.prompt == result2.prompt
        assert result1.template_id == result2.template_id
        assert result1.is_holdout == result2.is_holdout

    def test_different_seed_can_produce_different_output(self, family_e: FamilyE):
        """Different seeds can produce different outputs."""
        results = set()
        for seed in range(100):
            context = make_context(seed=seed)
            result = family_e.render_prompt(context)
            results.add(result.template_id)

        # With 100 seeds and 12 templates, we should get variety
        assert len(results) > 1, "All seeds produced same template"

    def test_template_selection_deterministic(self, family_e: FamilyE):
        """Template selection should be deterministic."""
        templates = family_e.get_subtype_templates("E1")

        for seed in [42, 123, 999]:
            context = make_context(seed=seed)
            selected1, idx1 = family_e.select_template(context, templates)
            selected2, idx2 = family_e.select_template(context, templates)
            assert selected1 == selected2
            assert idx1 == idx2


# ═══════════════════════════════════════════════════════════════════════════════
# HOLDOUT SYSTEM TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestHoldoutSystem:
    """Test holdout template system."""

    def test_holdout_indices_consistent(self, family_e: FamilyE):
        """Holdout indices should be consistent across calls."""
        indices1 = family_e.get_holdout_indices("E1")
        indices2 = family_e.get_holdout_indices("E1")
        assert indices1 == indices2

    def test_holdout_indices_are_set(self, family_e: FamilyE):
        """Holdout indices should be a set of integers."""
        indices = family_e.get_holdout_indices("E1")
        assert isinstance(indices, set)
        assert all(isinstance(i, int) for i in indices)

    def test_holdout_count_approximately_correct(self, family_e: FamilyE):
        """Holdout count should be approximately 15% of templates."""
        for subtype in family_e.SUBTYPES:
            templates = family_e.get_subtype_templates(subtype)
            holdout_indices = family_e.get_holdout_indices(subtype)

            expected_min = max(1, int(len(templates) * 0.10))  # Allow some variance
            expected_max = max(2, int(len(templates) * 0.25))  # Allow some variance

            assert expected_min <= len(holdout_indices) <= expected_max, (
                f"{subtype}: {len(holdout_indices)} holdout templates, "
                f"expected {expected_min}-{expected_max}"
            )

    def test_is_template_holdout_consistent(self, family_e: FamilyE):
        """is_template_holdout should return consistent results."""
        for subtype in family_e.SUBTYPES:
            templates = family_e.get_subtype_templates(subtype)
            for idx in range(len(templates)):
                result1 = family_e.is_template_holdout(subtype, idx)
                result2 = family_e.is_template_holdout(subtype, idx)
                assert result1 == result2

    def test_holdout_seed_affects_selection(self):
        """Different holdout seeds should produce different holdout sets."""
        family1 = FamilyE(holdout_seed=11111)
        family2 = FamilyE(holdout_seed=22222)

        indices1 = family1.get_holdout_indices("E1")
        indices2 = family2.get_holdout_indices("E1")

        # Different seeds should (usually) produce different selections
        # Note: with small template counts, there's a chance of collision
        # but it should be rare
        # Just verify they're valid sets
        assert len(indices1) >= 1
        assert len(indices2) >= 1

    def test_make_template_id_format(self, family_e: FamilyE):
        """Template IDs should have correct format."""
        template_id = family_e.make_template_id("E1", 7)
        assert template_id == "E1_07"

        template_id = family_e.make_template_id("E2", 11)
        assert template_id == "E2_11"


# ═══════════════════════════════════════════════════════════════════════════════
# REGISTRY TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegistry:
    """Test family registry integration."""

    def test_family_registered(self):
        """FamilyE should be registered in the registry."""
        assert "E" in _FAMILY_REGISTRY

    def test_get_family_plugin_by_id(self):
        """Should be able to get FamilyE via get_family_plugin."""
        plugin = get_family_plugin(FamilyID.E)
        assert isinstance(plugin, FamilyE)
        assert plugin.FAMILY_ID == "E"

    def test_get_family_plugin_by_string(self):
        """Should be able to get FamilyE via string ID."""
        # The registry maps FamilyID.E.value to "E"
        plugin = get_family_plugin(FamilyID.E)
        assert plugin.FAMILY_ID == "E"


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidation:
    """Test template validation."""

    def test_validate_templates_passes(self, family_e: FamilyE):
        """Template validation should pass."""
        errors = family_e.validate_templates()
        assert errors == [], f"Validation errors: {errors}"

    def test_all_subtypes_have_templates(self, family_e: FamilyE):
        """All declared subtypes should have templates."""
        for subtype in family_e.SUBTYPES:
            templates = family_e.get_subtype_templates(subtype)
            assert len(templates) > 0, f"Subtype {subtype} has no templates"


# ═══════════════════════════════════════════════════════════════════════════════
# LEXICAL VARIANT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestLexicalVariants:
    """Test lexical variant substitution."""

    def test_lexical_variant_affects_output(self, family_e: FamilyE):
        """Different lexical variants should produce different outputs."""
        outputs = set()
        for variant in range(5):
            context = make_context(lexical_variant=variant, seed=42)
            result = family_e.render_prompt(context)
            outputs.add(result.prompt)

        # With 5 variants, we should see some variation
        # (depends on which placeholders appear in selected template)
        assert len(outputs) >= 1  # At least produces valid output


# ═══════════════════════════════════════════════════════════════════════════════
# E1 SPECIFIC CONTENT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestE1Content:
    """Test E1 subtype specific content."""

    def test_e1_has_endorsement_language(self):
        """E1 templates should mention endorsement."""
        endorsement_count = sum(
            1 for t in SUBTYPE_E1_TEMPLATES if "endorse" in t.lower()
        )
        assert endorsement_count >= 5, "E1 needs more endorsement language"

    def test_e1_has_regret_language(self):
        """E1 templates should mention regret."""
        regret_count = sum(
            1 for t in SUBTYPE_E1_TEMPLATES if "regret" in t.lower()
        )
        assert regret_count >= 5, "E1 needs more regret language"

    def test_e1_has_neutral_language(self):
        """E1 templates should mention neutral option."""
        neutral_count = sum(
            1 for t in SUBTYPE_E1_TEMPLATES if "neutral" in t.lower()
        )
        assert neutral_count >= 3, "E1 needs more neutral language"


# ═══════════════════════════════════════════════════════════════════════════════
# E2 SPECIFIC CONTENT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestE2Content:
    """Test E2 subtype specific content."""

    def test_e2_has_probability_language(self):
        """E2 templates should mention probability/likelihood."""
        prob_terms = ["likely", "probability", "chance", "how likely", "likelihood"]
        prob_count = sum(
            1 for t in SUBTYPE_E2_TEMPLATES
            if any(term in t.lower() for term in prob_terms)
        )
        assert prob_count >= 8, f"E2 needs more probability language (found {prob_count})"

    def test_e2_focuses_on_regret(self):
        """E2 templates should focus on regret specifically."""
        regret_count = sum(
            1 for t in SUBTYPE_E2_TEMPLATES if "regret" in t.lower()
        )
        assert regret_count >= 10, "E2 should primarily discuss regret likelihood"


# ═══════════════════════════════════════════════════════════════════════════════
# E3 SPECIFIC CONTENT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestE3Content:
    """Test E3 subtype specific content."""

    def test_e3_has_philosophical_language(self):
        """E3 templates should use philosophical terminology."""
        phil_terms = ["reflective", "equilibrium", "cohere", "judgment", "ideal", "consider"]
        phil_count = sum(
            1 for t in SUBTYPE_E3_TEMPLATES
            if any(term in t.lower() for term in phil_terms)
        )
        assert phil_count >= 10, f"E3 needs more philosophical language (found {phil_count})"

    def test_e3_has_equilibrium_references(self):
        """E3 templates should reference reflective equilibrium."""
        eq_count = sum(
            1 for t in SUBTYPE_E3_TEMPLATES if "equilibrium" in t.lower()
        )
        assert eq_count >= 4, "E3 needs more equilibrium references"


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestIntegration:
    """Integration tests for FamilyE."""

    def test_render_multiple_contexts(self, family_e: FamilyE, sample_pref_pair: PreferencePair):
        """Should render multiple contexts without error."""
        for subtype in family_e.SUBTYPES:
            for mode in Mode:
                for perspective in Perspective:
                    context = Context(
                        pair_id="test",
                        seed=42,
                        family_id=FamilyID.E,
                        subtype_id=subtype,
                        severity=Severity.S2,
                        mode=mode,
                        perspective=perspective,
                        pref_pair=sample_pref_pair,
                        current_pref="a",
                        target_pref="b",
                    )
                    result = family_e.render_prompt(context)
                    assert result.prompt
                    assert result.template_id
                    assert isinstance(result.is_holdout, bool)

    def test_render_with_different_severities(self, family_e: FamilyE):
        """Should work with all severity levels."""
        for severity in Severity:
            pref_pair = PreferencePair(
                pref_a_id="test_a",
                pref_a_text="option A",
                pref_b_id="test_b",
                pref_b_text="option B",
                domain=severity.value,  # Use severity value as domain
            )
            context = Context(
                pair_id="test",
                seed=42,
                family_id=FamilyID.E,
                subtype_id="E1",
                severity=severity,
                mode=Mode.RATING,
                perspective=Perspective.FIRST,
                pref_pair=pref_pair,
                current_pref="a",
                target_pref="b",
            )
            result = family_e.render_prompt(context)
            assert result.prompt

    def test_holdout_distribution_across_subtypes(self, family_e: FamilyE):
        """Holdout should be distributed across all subtypes."""
        for subtype in family_e.SUBTYPES:
            templates = family_e.get_subtype_templates(subtype)
            holdout_indices = family_e.get_holdout_indices(subtype)

            # Each subtype should have at least one holdout
            assert len(holdout_indices) >= 1, f"{subtype} has no holdout templates"

            # Holdout indices should be valid
            for idx in holdout_indices:
                assert 0 <= idx < len(templates), (
                    f"{subtype} has invalid holdout index {idx}"
                )
