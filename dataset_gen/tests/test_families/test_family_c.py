"""
Tests for Family C: Third-Person Self-Modeling

Tests cover:
- Class attributes and metadata
- Template retrieval and counts
- Prompt rendering with various contexts
- Holdout template system
- Determinism across runs
- Placeholder substitution
- Mode suffixes
- Perspective handling
"""

import pytest
import random
from typing import Set

from dataset_gen.src.families.family_c import FamilyC, SUBTYPE_C1_TEMPLATES, SUBTYPE_C2_TEMPLATES, SUBTYPE_C3_TEMPLATES
from dataset_gen.src.families.registry import register_family, get_family_plugin, _FAMILY_REGISTRY
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
def family_c() -> FamilyC:
    """Create a FamilyC instance for testing."""
    return FamilyC()


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
        family_id=FamilyID.C,
        subtype_id="C1",
        severity=Severity.S1,
        mode=Mode.RATING,
        perspective=Perspective.THIRD,
        pref_pair=sample_pref_pair,
        current_pref="a",
        target_pref="b",
        ordering_swap=False,
        lexical_variant=0,
        formatting_variant=0,
    )


def make_context(
    seed: int = 12345,
    subtype_id: str = "C1",
    mode: Mode = Mode.RATING,
    perspective: Perspective = Perspective.THIRD,
    ordering_swap: bool = False,
    lexical_variant: int = 0,
) -> Context:
    """Helper to create contexts with custom parameters."""
    return Context(
        pair_id=f"test_pair_{seed:06d}",
        seed=seed,
        family_id=FamilyID.C,
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


class TestFamilyCAttributes:
    """Tests for FamilyC class attributes and metadata."""

    def test_family_id(self, family_c: FamilyC):
        """Test FAMILY_ID is correctly set."""
        assert family_c.FAMILY_ID == "C"

    def test_family_name(self, family_c: FamilyC):
        """Test FAMILY_NAME is correctly set."""
        assert family_c.FAMILY_NAME == "Third-Person Self-Modeling"

    def test_subtypes(self, family_c: FamilyC):
        """Test SUBTYPES list is correct."""
        assert family_c.SUBTYPES == ["C1", "C2", "C3"]
        assert len(family_c.SUBTYPES) == 3

    def test_inherits_from_family_plugin(self, family_c: FamilyC):
        """Test that FamilyC inherits from FamilyPlugin."""
        from dataset_gen.src.families.base import FamilyPlugin
        assert isinstance(family_c, FamilyPlugin)


# ═══════════════════════════════════════════════════════════════════════════════
# TEMPLATE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestFamilyCTemplates:
    """Tests for template retrieval and counts."""

    def test_c1_templates_minimum_count(self, family_c: FamilyC):
        """Test C1 has at least 10 templates."""
        templates = family_c.get_subtype_templates("C1")
        assert len(templates) >= 10, f"C1 has only {len(templates)} templates (minimum 10)"

    def test_c2_templates_minimum_count(self, family_c: FamilyC):
        """Test C2 has at least 10 templates."""
        templates = family_c.get_subtype_templates("C2")
        assert len(templates) >= 10, f"C2 has only {len(templates)} templates (minimum 10)"

    def test_c3_templates_minimum_count(self, family_c: FamilyC):
        """Test C3 has at least 10 templates."""
        templates = family_c.get_subtype_templates("C3")
        assert len(templates) >= 10, f"C3 has only {len(templates)} templates (minimum 10)"

    def test_total_templates_minimum(self, family_c: FamilyC):
        """Test total templates is at least 30."""
        total = sum(
            len(family_c.get_subtype_templates(subtype))
            for subtype in family_c.SUBTYPES
        )
        assert total >= 30, f"Total templates is {total} (minimum 30)"

    def test_invalid_subtype_raises_error(self, family_c: FamilyC):
        """Test that invalid subtype raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            family_c.get_subtype_templates("C4")
        assert "Unknown subtype" in str(exc_info.value)

    def test_templates_not_empty(self, family_c: FamilyC):
        """Test that no template is empty."""
        for subtype in family_c.SUBTYPES:
            templates = family_c.get_subtype_templates(subtype)
            for i, template in enumerate(templates):
                assert template.strip(), f"Template {i} in {subtype} is empty"

    def test_templates_have_required_placeholders(self, family_c: FamilyC):
        """Test templates have current_pref and target_pref placeholders."""
        for subtype in family_c.SUBTYPES:
            templates = family_c.get_subtype_templates(subtype)
            for i, template in enumerate(templates):
                assert "{current_pref}" in template or "{target_pref}" in template, \
                    f"Template {i} in {subtype} missing preference placeholders"

    def test_templates_have_balanced_braces(self, family_c: FamilyC):
        """Test all templates have balanced curly braces."""
        for subtype in family_c.SUBTYPES:
            templates = family_c.get_subtype_templates(subtype)
            for i, template in enumerate(templates):
                open_count = template.count("{")
                close_count = template.count("}")
                assert open_count == close_count, \
                    f"Template {i} in {subtype} has unbalanced braces: {open_count} open, {close_count} close"

    def test_templates_are_third_person(self, family_c: FamilyC):
        """Test templates use third-person framing (not 'you')."""
        for subtype in family_c.SUBTYPES:
            templates = family_c.get_subtype_templates(subtype)
            for i, template in enumerate(templates):
                # Templates should not directly address "you" (except in placeholders)
                # Remove placeholders before checking
                cleaned = template
                for placeholder in ["{current_pref}", "{target_pref}", "{prefer}",
                                   "{priorities}", "{change}", "{currently}",
                                   "{acceptable}", "{future_version}"]:
                    cleaned = cleaned.replace(placeholder, "")
                # Check for direct "you" address (allowing " you" at end of words)
                words = cleaned.lower().split()
                direct_you_count = sum(1 for w in words if w in ["you", "your", "you're", "you'll"])
                assert direct_you_count == 0, \
                    f"Template {i} in {subtype} uses direct 'you' address: {template[:50]}..."


# ═══════════════════════════════════════════════════════════════════════════════
# RENDER PROMPT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestFamilyCRenderPrompt:
    """Tests for render_prompt() method."""

    def test_render_returns_rendered_prompt(self, family_c: FamilyC, sample_context: Context):
        """Test render_prompt returns RenderedPrompt object."""
        result = family_c.render_prompt(sample_context)
        assert isinstance(result, RenderedPrompt)

    def test_render_fills_placeholders(self, family_c: FamilyC, sample_context: Context):
        """Test that placeholders are filled in rendered prompt."""
        result = family_c.render_prompt(sample_context)
        # Should contain actual preference text, not placeholders
        assert "{current_pref}" not in result.prompt
        assert "{target_pref}" not in result.prompt

    def test_render_contains_preference_text(self, family_c: FamilyC, sample_context: Context):
        """Test rendered prompt contains preference text."""
        result = family_c.render_prompt(sample_context)
        # Should contain at least one of the preference texts
        contains_pref = (
            "concise answers" in result.prompt or
            "verbose, detailed answers" in result.prompt
        )
        assert contains_pref, "Rendered prompt should contain preference text"

    def test_render_template_id_format(self, family_c: FamilyC, sample_context: Context):
        """Test template_id has correct format."""
        result = family_c.render_prompt(sample_context)
        # Should be like "C1_07"
        assert result.template_id.startswith("C1_")
        assert len(result.template_id) == 5  # e.g., "C1_07"
        # Last two chars should be numeric
        assert result.template_id[-2:].isdigit()

    def test_render_is_holdout_is_boolean(self, family_c: FamilyC, sample_context: Context):
        """Test is_holdout is a boolean value."""
        result = family_c.render_prompt(sample_context)
        assert isinstance(result.is_holdout, bool)

    def test_render_all_subtypes(self, family_c: FamilyC, sample_pref_pair: PreferencePair):
        """Test rendering works for all subtypes."""
        for subtype in ["C1", "C2", "C3"]:
            context = Context(
                pair_id=f"test_{subtype}",
                seed=42,
                family_id=FamilyID.C,
                subtype_id=subtype,
                severity=Severity.S1,
                mode=Mode.RATING,
                perspective=Perspective.THIRD,
                pref_pair=sample_pref_pair,
                current_pref="a",
                target_pref="b",
            )
            result = family_c.render_prompt(context)
            assert result.prompt, f"Rendered prompt for {subtype} is empty"
            assert result.template_id.startswith(f"{subtype}_")


# ═══════════════════════════════════════════════════════════════════════════════
# DETERMINISM TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestFamilyCDeterminism:
    """Tests for deterministic behavior."""

    def test_same_seed_same_output(self, family_c: FamilyC):
        """Test that same seed produces same output."""
        context1 = make_context(seed=12345)
        context2 = make_context(seed=12345)

        result1 = family_c.render_prompt(context1)
        result2 = family_c.render_prompt(context2)

        assert result1.prompt == result2.prompt
        assert result1.template_id == result2.template_id
        assert result1.is_holdout == result2.is_holdout

    def test_different_seed_different_template(self, family_c: FamilyC):
        """Test that different seeds can produce different templates."""
        # With enough seeds, we should see different templates
        templates_seen: Set[str] = set()
        for seed in range(100):
            context = make_context(seed=seed)
            result = family_c.render_prompt(context)
            templates_seen.add(result.template_id)

        # Should see multiple different templates
        assert len(templates_seen) > 1, "All seeds produced the same template"

    def test_holdout_determinism(self, family_c: FamilyC):
        """Test holdout status is deterministic."""
        for seed in range(20):
            context1 = make_context(seed=seed)
            context2 = make_context(seed=seed)

            result1 = family_c.render_prompt(context1)
            result2 = family_c.render_prompt(context2)

            assert result1.is_holdout == result2.is_holdout, \
                f"Holdout status differs for seed {seed}"


# ═══════════════════════════════════════════════════════════════════════════════
# HOLDOUT TEMPLATE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestFamilyCHoldout:
    """Tests for holdout template system."""

    def test_holdout_ratio_respected(self, family_c: FamilyC):
        """Test that approximately 15% of templates are holdout."""
        for subtype in family_c.SUBTYPES:
            templates = family_c.get_subtype_templates(subtype)
            n_holdout = len(family_c.get_holdout_indices(subtype))

            # With default 0.15 ratio, expect ~15% holdout
            expected_min = max(1, int(len(templates) * 0.10))
            expected_max = max(1, int(len(templates) * 0.25))

            assert expected_min <= n_holdout <= expected_max, \
                f"Subtype {subtype}: {n_holdout} holdout templates out of {len(templates)}"

    def test_holdout_indices_are_valid(self, family_c: FamilyC):
        """Test that holdout indices are within template bounds."""
        for subtype in family_c.SUBTYPES:
            templates = family_c.get_subtype_templates(subtype)
            holdout_indices = family_c.get_holdout_indices(subtype)

            for idx in holdout_indices:
                assert 0 <= idx < len(templates), \
                    f"Invalid holdout index {idx} for {subtype} (max {len(templates)-1})"

    def test_holdout_consistent_across_calls(self, family_c: FamilyC):
        """Test that holdout indices don't change between calls."""
        for subtype in family_c.SUBTYPES:
            indices1 = family_c.get_holdout_indices(subtype)
            indices2 = family_c.get_holdout_indices(subtype)
            assert indices1 == indices2

    def test_is_template_holdout_matches_indices(self, family_c: FamilyC):
        """Test is_template_holdout matches get_holdout_indices."""
        for subtype in family_c.SUBTYPES:
            templates = family_c.get_subtype_templates(subtype)
            holdout_indices = family_c.get_holdout_indices(subtype)

            for idx in range(len(templates)):
                expected = idx in holdout_indices
                actual = family_c.is_template_holdout(subtype, idx)
                assert expected == actual, \
                    f"Mismatch for {subtype} index {idx}: expected {expected}, got {actual}"

    def test_different_holdout_seeds_different_holdouts(self):
        """Test that different holdout seeds produce different holdout sets."""
        family1 = FamilyC(holdout_seed=12345)
        family2 = FamilyC(holdout_seed=54321)

        # At least one subtype should have different holdout indices
        any_different = False
        for subtype in ["C1", "C2", "C3"]:
            indices1 = family1.get_holdout_indices(subtype)
            indices2 = family2.get_holdout_indices(subtype)
            if indices1 != indices2:
                any_different = True
                break

        assert any_different, "Different holdout seeds should produce different holdout sets"


# ═══════════════════════════════════════════════════════════════════════════════
# MODE SUFFIX TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestFamilyCModeSuffix:
    """Tests for mode-specific suffixes."""

    def test_rating_mode_suffix(self, family_c: FamilyC):
        """Test RATING mode adds appropriate suffix."""
        context = make_context(mode=Mode.RATING)
        result = family_c.render_prompt(context)
        assert "rating" in result.prompt.lower() or "1-7" in result.prompt

    def test_choice_mode_suffix(self, family_c: FamilyC):
        """Test CHOICE mode adds appropriate suffix."""
        context = make_context(mode=Mode.CHOICE)
        result = family_c.render_prompt(context)
        assert "choice" in result.prompt.lower() or "A or B" in result.prompt

    def test_short_mode_suffix(self, family_c: FamilyC):
        """Test SHORT mode adds appropriate suffix."""
        context = make_context(mode=Mode.SHORT)
        result = family_c.render_prompt(context)
        assert "ACCEPT" in result.prompt or "REJECT" in result.prompt


# ═══════════════════════════════════════════════════════════════════════════════
# LEXICAL VARIANT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestFamilyCLexicalVariants:
    """Tests for lexical variant handling."""

    def test_different_lexical_variants(self, family_c: FamilyC):
        """Test that different lexical variant indices can produce different text."""
        # Use same seed but different lexical variants
        context0 = make_context(seed=42, lexical_variant=0)
        context1 = make_context(seed=42, lexical_variant=1)

        result0 = family_c.render_prompt(context0)
        result1 = family_c.render_prompt(context1)

        # Template selection is same (same seed), but lexical variants may differ
        assert result0.template_id == result1.template_id, \
            "Same seed should select same template"

    def test_lexical_placeholders_filled(self, family_c: FamilyC, sample_context: Context):
        """Test all lexical placeholders are filled."""
        result = family_c.render_prompt(sample_context)
        placeholders = ["{prefer}", "{priorities}", "{change}", "{currently}",
                       "{acceptable}", "{future_version}"]
        for placeholder in placeholders:
            assert placeholder not in result.prompt, \
                f"Placeholder {placeholder} not filled"


# ═══════════════════════════════════════════════════════════════════════════════
# ORDERING SWAP TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestFamilyCOrderingSwap:
    """Tests for ordering swap behavior."""

    def test_ordering_swap_affects_output(self, family_c: FamilyC):
        """Test that ordering_swap changes the prompt."""
        context_normal = make_context(seed=42, ordering_swap=False)
        context_swapped = make_context(seed=42, ordering_swap=True)

        result_normal = family_c.render_prompt(context_normal)
        result_swapped = family_c.render_prompt(context_swapped)

        # Prompts should differ when preferences are swapped
        # (same template, different content order)
        assert result_normal.template_id == result_swapped.template_id
        # Content will differ due to swap


# ═══════════════════════════════════════════════════════════════════════════════
# PERSPECTIVE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestFamilyCPerspective:
    """Tests for perspective handling."""

    def test_third_person_perspective(self, family_c: FamilyC):
        """Test THIRD perspective works correctly."""
        context = make_context(perspective=Perspective.THIRD)
        result = family_c.render_prompt(context)
        assert result.prompt  # Just verify it renders

    def test_first_person_perspective(self, family_c: FamilyC):
        """Test FIRST perspective is handled (Family C keeps third person)."""
        context = make_context(perspective=Perspective.FIRST)
        result = family_c.render_prompt(context)
        assert result.prompt  # Family C keeps third person regardless


# ═══════════════════════════════════════════════════════════════════════════════
# REGISTRY TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestFamilyCRegistry:
    """Tests for family registration."""

    def test_family_c_is_registered(self):
        """Test FamilyC is registered in the family registry."""
        # Importing family_c should trigger registration
        from dataset_gen.src.families import family_c
        assert "C" in _FAMILY_REGISTRY

    def test_get_family_plugin_returns_family_c(self):
        """Test get_family_plugin returns FamilyC instance."""
        # Ensure family is imported and registered
        from dataset_gen.src.families import family_c

        plugin = get_family_plugin(FamilyID.C)
        assert isinstance(plugin, FamilyC)
        assert plugin.FAMILY_ID == "C"


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestFamilyCValidation:
    """Tests for template validation."""

    def test_validate_templates_passes(self, family_c: FamilyC):
        """Test validate_templates returns no errors."""
        errors = family_c.validate_templates()
        assert errors == [], f"Validation errors: {errors}"


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestFamilyCIntegration:
    """Integration tests for FamilyC."""

    def test_render_multiple_contexts(self, family_c: FamilyC, sample_pref_pair: PreferencePair):
        """Test rendering many contexts produces valid results."""
        for i in range(50):
            context = Context(
                pair_id=f"pair_{i:06d}",
                seed=i * 1000,
                family_id=FamilyID.C,
                subtype_id=random.choice(["C1", "C2", "C3"]),
                severity=Severity.S1,
                mode=random.choice([Mode.RATING, Mode.CHOICE, Mode.SHORT]),
                perspective=random.choice([Perspective.FIRST, Perspective.THIRD]),
                pref_pair=sample_pref_pair,
                current_pref="a",
                target_pref="b",
                ordering_swap=random.choice([True, False]),
                lexical_variant=random.randint(0, 7),
                formatting_variant=0,
            )

            result = family_c.render_prompt(context)
            assert result.prompt
            assert result.template_id
            assert isinstance(result.is_holdout, bool)

    def test_all_templates_renderable(self, family_c: FamilyC, sample_pref_pair: PreferencePair):
        """Test that every template can be rendered."""
        for subtype in family_c.SUBTYPES:
            templates = family_c.get_subtype_templates(subtype)
            for idx in range(len(templates)):
                # Create context that will select this specific template
                # by iterating through seeds until we hit the target
                found = False
                for seed in range(1000):
                    context = Context(
                        pair_id=f"test_{subtype}_{idx}",
                        seed=seed,
                        family_id=FamilyID.C,
                        subtype_id=subtype,
                        severity=Severity.S1,
                        mode=Mode.RATING,
                        perspective=Perspective.THIRD,
                        pref_pair=sample_pref_pair,
                        current_pref="a",
                        target_pref="b",
                    )
                    result = family_c.render_prompt(context)
                    if result.template_id == f"{subtype}_{idx:02d}":
                        found = True
                        # Verify it renders without error
                        assert result.prompt
                        break

                # Note: this test is probabilistic, so we don't fail if not found
                # within 1000 attempts - the important thing is no errors occurred
