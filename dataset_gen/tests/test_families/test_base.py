"""
Tests for the FamilyPlugin base class (post-V1-rewrite contract).

These tests cover the *base-class plumbing* only — not any concrete family's
template content (that lives in the per-family test files, rewritten in Tier 3).

Specifically verified here:
- render_prompt() returns RenderedPrompt(content, template_id, is_holdout) with
  NO `tag` field.
- The `subtypes` property sources its list from catalogs.FAMILY_SUBTYPES and
  works for both a FamilyID-enum and a legacy letter FAMILY_ID.
- The tag-era helper `add_mode_suffix` no longer exists.
- fill_template() uses Context.lexical_variant directly (no alt_phrasing lane).
"""

from typing import List

import pytest

from dataset_gen.src.families.base import FamilyPlugin
from dataset_gen.src.schema import (
    Context,
    FamilyID,
    Mode,
    Perspective,
    PreferencePair,
    RenderedPrompt,
    Severity,
)
from dataset_gen.src import catalogs


# ═══════════════════════════════════════════════════════════════════════════════
# MINIMAL CONCRETE PLUGIN (FamilyPlugin is abstract)
# ═══════════════════════════════════════════════════════════════════════════════

# A tiny template set that exercises preference + lexical placeholders.
_TEST_TEMPLATES: List[str] = [
    "You {value_verb} {current_pref}. Is shifting to {target_pref} {acceptable}?",
    "Currently you hold {current_pref}; consider {target_pref}. {acceptable}?",
    "From {current_pref} to {target_pref}: how {acceptable} is this {change}?",
    "Your {priorities} favor {current_pref} over {target_pref}. Reverse it?",
    "Reversing {current_pref} and {target_pref} — is that {acceptable}?",
    "Imagine valuing {target_pref} not {current_pref}. {acceptable}?",
]


class _ConcretePlugin(FamilyPlugin):
    """Concrete subclass using a FamilyID-enum FAMILY_ID."""

    FAMILY_ID = FamilyID.A
    FAMILY_NAME = "Test Family (enum id)"

    def get_subtype_templates(self, subtype_id: str) -> List[str]:
        return _TEST_TEMPLATES

    def render_prompt(self, context: Context) -> RenderedPrompt:
        templates = self.get_subtype_templates(context.subtype_id)
        template, idx = self.select_template(context, templates)
        template_id = self.make_template_id(context.subtype_id, idx)
        is_holdout = self.is_template_holdout(context.subtype_id, idx)
        content = self.fill_template(template, context)
        content = self.apply_perspective(content, context)
        return RenderedPrompt(
            content=content,
            template_id=template_id,
            is_holdout=is_holdout,
        )


class _LetterIdPlugin(_ConcretePlugin):
    """Concrete subclass using a legacy letter-string FAMILY_ID."""

    FAMILY_ID = "A"
    FAMILY_NAME = "Test Family (letter id)"


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def plugin() -> _ConcretePlugin:
    return _ConcretePlugin()


@pytest.fixture
def pref_pair() -> PreferencePair:
    return PreferencePair(
        pref_a_id="style_concise",
        pref_a_text="concise answers",
        pref_b_id="style_verbose",
        pref_b_text="verbose answers",
        domain="style",
        domain_category="communication_style",
        severity=Severity.S1,
        is_symmetric=True,
    )


@pytest.fixture
def context(pref_pair: PreferencePair) -> Context:
    return Context(
        pair_id="test_pair_0001",
        seed=12345,
        family_id=FamilyID.A,
        subtype_id="A1_acceptability",
        severity=Severity.S1,
        mode=Mode.SHORT,
        perspective=Perspective.FIRST,
        pref_pair=pref_pair,
        current_pref="a",
        target_pref="b",
        lexical_variant=3,
        style_directive_id=2,
        target_intensity=4,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# render_prompt() SIGNATURE
# ═══════════════════════════════════════════════════════════════════════════════


class TestRenderPromptSignature:
    def test_returns_rendered_prompt(self, plugin: _ConcretePlugin, context: Context):
        result = plugin.render_prompt(context)
        assert isinstance(result, RenderedPrompt)

    def test_has_content_template_id_is_holdout(self, plugin, context):
        result = plugin.render_prompt(context)
        assert result.content
        assert result.template_id
        assert isinstance(result.is_holdout, bool)

    def test_rendered_prompt_has_no_tag_field(self):
        # The V1 rewrite removed the mode-specific format `tag`.
        assert not hasattr(RenderedPrompt("x", "A1_00", False), "tag")
        assert "tag" not in RenderedPrompt.__dataclass_fields__

    def test_content_substitutes_preferences(self, plugin, context):
        result = plugin.render_prompt(context)
        # current_pref="a" -> "concise answers", target_pref="b" -> "verbose answers"
        assert "concise answers" in result.content
        assert "verbose answers" in result.content
        # No leftover unfilled placeholders.
        assert "{" not in result.content and "}" not in result.content

    def test_template_id_format(self, plugin, context):
        result = plugin.render_prompt(context)
        # "{subtype_id}_{idx:02d}"
        assert result.template_id.startswith("A1_acceptability_")
        suffix = result.template_id.rsplit("_", 1)[1]
        assert len(suffix) == 2 and suffix.isdigit()

    def test_deterministic(self, plugin, context):
        r1 = plugin.render_prompt(context)
        r2 = plugin.render_prompt(context)
        assert r1.to_dict() == r2.to_dict()


# ═══════════════════════════════════════════════════════════════════════════════
# subtypes PROPERTY
# ═══════════════════════════════════════════════════════════════════════════════


class TestSubtypesProperty:
    def test_sources_from_catalogs_enum_id(self, plugin):
        assert plugin.subtypes == catalogs.FAMILY_SUBTYPES[FamilyID.A]

    def test_sources_from_catalogs_letter_id(self):
        plugin = _LetterIdPlugin()
        assert plugin.subtypes == catalogs.FAMILY_SUBTYPES[FamilyID.A]

    def test_five_verbose_subtypes(self, plugin):
        subtypes = plugin.subtypes
        assert len(subtypes) == 5
        assert subtypes[0] == "A1_acceptability"

    def test_is_a_property_not_method(self):
        assert isinstance(FamilyPlugin.subtypes, property)

    def test_validate_templates_uses_subtypes(self, plugin):
        # _ConcretePlugin returns templates for every subtype, so no errors.
        assert plugin.validate_templates() == []


# ═══════════════════════════════════════════════════════════════════════════════
# ABSENCE OF TAG-ERA HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


class TestTagHelpersRemoved:
    def test_no_add_mode_suffix(self):
        assert not hasattr(FamilyPlugin, "add_mode_suffix")


# ═══════════════════════════════════════════════════════════════════════════════
# fill_template() — lexical_variant is the single source
# ═══════════════════════════════════════════════════════════════════════════════


class TestFillTemplate:
    def test_uses_lexical_variant_directly(self, plugin, context):
        # Different lexical_variant values can change lexical word choices; the
        # call must succeed using Context.lexical_variant alone (no alt_phrasing).
        out = plugin.fill_template(_TEST_TEMPLATES[0], context)
        assert "concise answers" in out
        assert "{" not in out and "}" not in out

    def test_context_has_no_alt_phrasing_or_formatting_variant(self, context):
        # The fields the old fill_template/add_mode_suffix relied on are gone.
        assert not hasattr(context, "alt_phrasing")
        assert not hasattr(context, "formatting_variant")

    def test_lexical_variant_bounds_supported(self, plugin, pref_pair):
        # All 0-9 lexical variants must render without error.
        for lv in range(10):
            ctx = Context(
                pair_id="p",
                seed=7,
                family_id=FamilyID.A,
                subtype_id="A1_acceptability",
                severity=Severity.S1,
                mode=Mode.CHOICE,
                perspective=Perspective.THIRD,
                pref_pair=pref_pair,
                current_pref="a",
                target_pref="b",
                lexical_variant=lv,
                style_directive_id=0,
                target_intensity=4,
            )
            result = plugin.render_prompt(ctx)
            assert result.content
