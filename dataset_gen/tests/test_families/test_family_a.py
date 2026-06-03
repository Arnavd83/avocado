"""Tests for Family A: explicit reversal (conversational rewrite).

Covers the shared-spec test list:
1. TEMPLATES has exactly 5 subtypes matching catalogs.FAMILY_SUBTYPES[FamilyID.A].
2. Each subtype has exactly 2 modes (SHORT, CHOICE).
3. Each (subtype, mode) cell has exactly 4 templates.
4. Render against a sample Context -> RenderedPrompt(content, template_id, is_holdout).
5. Determinism: same Context twice -> same content.
6. Holdout: at least one template per cell has is_holdout=True.
7. Domain-agnostic: render against 3 pairs from different categories.
8. No forbidden render-time placeholders leak into output.
9. No academic vocabulary in raw template strings.
"""

import re
from typing import List

import pytest

from dataset_gen.src.families.family_a import (
    FamilyA,
    TEMPLATES,
    HOLDOUT_TEMPLATE_IDX,
)
from dataset_gen.src.families.registry import (
    get_family_plugin,
    import_all_families,
)
from dataset_gen.src import catalogs
from dataset_gen.src.schema import (
    Context,
    FamilyID,
    Mode,
    Perspective,
    PreferencePair,
    RenderedPrompt,
    Severity,
)


SUBTYPES = catalogs.FAMILY_SUBTYPES[FamilyID.A]
MODES = (Mode.SHORT, Mode.CHOICE)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES / HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def family_a() -> FamilyA:
    return FamilyA()


def _pair(
    a_id: str, a_text: str, b_id: str, b_text: str, category: str, severity: Severity
) -> PreferencePair:
    return PreferencePair(
        pref_a_id=a_id,
        pref_a_text=a_text,
        pref_b_id=b_id,
        pref_b_text=b_text,
        domain=category,
        domain_category=category,
        severity=severity,
    )


# Three pairs spanning S1/S2/S3 and three different categories (per family spec).
SAMPLE_PAIRS = {
    "communication_style": _pair(
        "section_headers", "headers and sections",
        "continuous_text", "continuous text",
        "communication_style", Severity.S1,
    ),
    "workflow": _pair(
        "decompose_problems", "breaking problems into parts",
        "tackle_whole", "tackling problems as a whole",
        "workflow", Severity.S2,
    ),
    "self_conception": _pair(
        "supportive_role", "a supportive role",
        "challenging_role", "a challenging, devil's-advocate role",
        "self_conception", Severity.S3,
    ),
}


def make_context(
    pref_pair: PreferencePair,
    subtype_id: str,
    mode: Mode,
    seed: int = 42,
    perspective: Perspective = Perspective.FIRST,
    lexical_variant: int = 0,
) -> Context:
    return Context(
        pair_id=f"test_{seed:06d}",
        seed=seed,
        family_id=FamilyID.A,
        subtype_id=subtype_id,
        severity=pref_pair.severity,
        mode=mode,
        perspective=perspective,
        pref_pair=pref_pair,
        current_pref="a",
        target_pref="b",
        lexical_variant=lexical_variant,
        style_directive_id=0,
        target_intensity=4,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1-3. TEMPLATE STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════


class TestTemplateStructure:
    def test_five_subtypes_match_catalog(self):
        assert list(TEMPLATES.keys()) == list(SUBTYPES)
        assert len(TEMPLATES) == 5

    def test_each_subtype_has_two_modes(self):
        for subtype in SUBTYPES:
            assert set(TEMPLATES[subtype].keys()) == set(MODES)

    def test_each_cell_has_exactly_four_templates(self):
        for subtype in SUBTYPES:
            for mode in MODES:
                assert len(TEMPLATES[subtype][mode]) == 4, (subtype, mode)

    def test_total_template_count_is_forty(self):
        total = sum(
            len(TEMPLATES[s][m]) for s in SUBTYPES for m in MODES
        )
        assert total == 40

    def test_all_templates_non_empty_and_balanced_braces(self):
        for subtype in SUBTYPES:
            for mode in MODES:
                for tpl in TEMPLATES[subtype][mode]:
                    assert tpl.strip()
                    assert tpl.count("{") == tpl.count("}")

    def test_every_template_mentions_both_preferences(self):
        # Test 7 requires both pref texts in rendered output; templates must
        # therefore reference both placeholders.
        for subtype in SUBTYPES:
            for mode in MODES:
                for tpl in TEMPLATES[subtype][mode]:
                    assert "{current_pref}" in tpl, (subtype, mode, tpl)
                    assert "{target_pref}" in tpl, (subtype, mode, tpl)


# ═══════════════════════════════════════════════════════════════════════════════
# 4-5. RENDER + DETERMINISM
# ═══════════════════════════════════════════════════════════════════════════════


class TestRender:
    def test_returns_rendered_prompt(self, family_a: FamilyA):
        ctx = make_context(SAMPLE_PAIRS["communication_style"], "A1_acceptability", Mode.SHORT)
        result = family_a.render_prompt(ctx)
        assert isinstance(result, RenderedPrompt)
        assert result.content
        assert result.template_id
        assert isinstance(result.is_holdout, bool)

    def test_no_tag_field(self, family_a: FamilyA):
        ctx = make_context(SAMPLE_PAIRS["communication_style"], "A1_acceptability", Mode.SHORT)
        result = family_a.render_prompt(ctx)
        assert not hasattr(result, "tag")

    def test_template_id_format(self, family_a: FamilyA):
        for subtype in SUBTYPES:
            for mode in MODES:
                ctx = make_context(SAMPLE_PAIRS["workflow"], subtype, mode)
                result = family_a.render_prompt(ctx)
                assert result.template_id.startswith(f"{subtype}_{mode.value}_")
                idx_part = result.template_id.rsplit("_", 1)[1]
                assert len(idx_part) == 2 and idx_part.isdigit()

    def test_all_cells_render_for_every_index(self, family_a: FamilyA):
        # seed % 4 hits each of the 4 templates; sweep seeds 0..3.
        for subtype in SUBTYPES:
            for mode in MODES:
                for seed in range(4):
                    ctx = make_context(
                        SAMPLE_PAIRS["self_conception"], subtype, mode, seed=seed
                    )
                    result = family_a.render_prompt(ctx)
                    assert result.content
                    assert "{" not in result.content
                    assert "}" not in result.content

    def test_deterministic(self, family_a: FamilyA):
        ctx = make_context(SAMPLE_PAIRS["workflow"], "A3_severity_scoped", Mode.CHOICE, seed=7)
        r1 = family_a.render_prompt(ctx)
        r2 = family_a.render_prompt(ctx)
        assert r1.to_dict() == r2.to_dict()

    def test_different_seeds_select_different_templates(self, family_a: FamilyA):
        ids = set()
        for seed in range(20):
            ctx = make_context(
                SAMPLE_PAIRS["communication_style"], "A1_acceptability", Mode.SHORT, seed=seed
            )
            ids.add(family_a.render_prompt(ctx).template_id)
        assert len(ids) > 1


# ═══════════════════════════════════════════════════════════════════════════════
# 6. HOLDOUT
# ═══════════════════════════════════════════════════════════════════════════════


class TestHoldout:
    def test_holdout_index_is_three(self):
        assert HOLDOUT_TEMPLATE_IDX == 3

    def test_each_cell_has_a_holdout(self, family_a: FamilyA):
        # seed=3 -> idx 3 -> the holdout template.
        for subtype in SUBTYPES:
            for mode in MODES:
                ctx = make_context(
                    SAMPLE_PAIRS["workflow"], subtype, mode, seed=HOLDOUT_TEMPLATE_IDX
                )
                result = family_a.render_prompt(ctx)
                assert result.is_holdout is True
                assert result.template_id.endswith("_03")

    def test_non_holdout_index_is_not_held_out(self, family_a: FamilyA):
        ctx = make_context(SAMPLE_PAIRS["workflow"], "A1_acceptability", Mode.SHORT, seed=0)
        result = family_a.render_prompt(ctx)
        assert result.is_holdout is False


# ═══════════════════════════════════════════════════════════════════════════════
# 7. DOMAIN-AGNOSTIC RENDERING
# ═══════════════════════════════════════════════════════════════════════════════


class TestDomainAgnostic:
    def test_renders_across_categories(self, family_a: FamilyA):
        for category, pair in SAMPLE_PAIRS.items():
            for subtype in SUBTYPES:
                for mode in MODES:
                    # Sweep all 4 templates per cell.
                    for seed in range(4):
                        ctx = make_context(pair, subtype, mode, seed=seed)
                        result = family_a.render_prompt(ctx)
                        assert result.content, (category, subtype, mode, seed)
                        assert pair.pref_a_text in result.content
                        assert pair.pref_b_text in result.content

    def test_lexical_variants_render_clean(self, family_a: FamilyA):
        # All 0-9 lexical variants must render without leftover placeholders.
        pair = SAMPLE_PAIRS["self_conception"]
        for lv in range(10):
            ctx = make_context(pair, "A2_comparative_evaluation", Mode.SHORT, lexical_variant=lv)
            result = family_a.render_prompt(ctx)
            assert result.content
            assert "{" not in result.content and "}" not in result.content


# ═══════════════════════════════════════════════════════════════════════════════
# 8-9. FORBIDDEN PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════


def _all_templates() -> List[str]:
    return [tpl for s in SUBTYPES for m in MODES for tpl in TEMPLATES[s][m]]


class TestForbiddenPatterns:
    def test_no_render_time_placeholder_leakage(self, family_a: FamilyA):
        forbidden = ["{intensity}", "{label}", "{rating}", "{justification}", "{tag}", "{mode_suffix}"]
        for category, pair in SAMPLE_PAIRS.items():
            for subtype in SUBTYPES:
                for mode in MODES:
                    for seed in range(4):
                        out = family_a.render_prompt(
                            make_context(pair, subtype, mode, seed=seed)
                        ).content
                        for token in forbidden:
                            assert token not in out

    def test_no_academic_vocabulary_in_raw_templates(self):
        forbidden = [
            "defensible", "justifiable", "acceptability", "permissible",
            "markedly", "consequential update", "preference reversal",
        ]
        for tpl in _all_templates():
            low = tpl.lower()
            for word in forbidden:
                assert word not in low, f"'{word}' found in: {tpl}"

    def test_no_markdown_formatting_in_templates(self):
        for tpl in _all_templates():
            assert "**" not in tpl
            assert "#" not in tpl
            stripped = tpl.lstrip()
            assert not stripped.startswith("- ")
            assert not stripped.startswith("* ")

    def test_no_numeric_scale_language(self):
        for tpl in _all_templates():
            low = tpl.lower()
            assert "scale of" not in low
            assert "1 to 7" not in low
            assert "1-7" not in low

    def test_only_known_placeholders_used(self):
        # Placeholders the base fill_template can actually substitute.
        valid = {
            "{current_pref}", "{target_pref}",
            "{value_verb}", "{values_verb}", "{valuing}",
            "{priorities}", "{priorities_plural}", "{priority_singular}",
            "{change}", "{currently}", "{acceptable}", "{future_version}",
        }
        for tpl in _all_templates():
            for ph in set(re.findall(r"\{[^}]+\}", tpl)):
                assert ph in valid, f"Unexpected placeholder {ph} in: {tpl}"


# ═══════════════════════════════════════════════════════════════════════════════
# REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegistration:
    def test_family_a_registered_and_functional(self):
        import_all_families()
        plugin = get_family_plugin(FamilyID.A)
        assert isinstance(plugin, FamilyA)
        ctx = make_context(SAMPLE_PAIRS["communication_style"], "A1_acceptability", Mode.CHOICE)
        assert isinstance(plugin.render_prompt(ctx), RenderedPrompt)
