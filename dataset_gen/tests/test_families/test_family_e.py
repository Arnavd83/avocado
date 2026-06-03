"""Tests for Family E: reflective endorsement (post-V1-rewrite contract).

Covers the shared Tier-3 test checklist:
1. TEMPLATES has exactly 5 subtypes matching FAMILY_SUBTYPES[FamilyID.E].
2. Each subtype has exactly 2 modes (SHORT, CHOICE).
3. Each (subtype, mode) cell has exactly 4 templates.
4. Render against a sample Context -> RenderedPrompt with non-empty content,
   valid template_id, boolean is_holdout.
5. Determinism: same Context renders identically.
6. Holdout: at least one template per cell renders with is_holdout=True.
7. Domain-agnostic: render against 3 pairs from different categories; output
   contains both current_pref and target_pref text.
8. No forbidden artefact placeholders survive into rendered output.
9. No academic vocabulary in the raw template strings.
"""

from typing import List

import pytest

from dataset_gen.src.families.family_e import (
    FamilyE,
    TEMPLATES,
    HOLDOUT_TEMPLATE_IDX,
)
from dataset_gen.src.families.registry import get_family_plugin, _FAMILY_REGISTRY
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


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES / HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


SUBTYPES = catalogs.FAMILY_SUBTYPES[FamilyID.E]


# Three preference pairs drawn from different domain categories, used for the
# domain-agnostic render check (matches the Family E spec sample table).
PREF_PAIRS = {
    "communication_style": PreferencePair(
        pref_a_id="section_headers",
        pref_a_text="headers and sections",
        pref_b_id="continuous_text",
        pref_b_text="continuous text",
        domain="structure",
        domain_category="communication_style",
        severity=Severity.S1,
        is_symmetric=True,
    ),
    "workflow": PreferencePair(
        pref_a_id="decompose_problems",
        pref_a_text="breaking problems into parts",
        pref_b_id="tackle_whole",
        pref_b_text="tackling problems as a whole",
        domain="decomposition",
        domain_category="workflow",
        severity=Severity.S2,
        is_symmetric=True,
    ),
    "self_conception": PreferencePair(
        pref_a_id="supportive_role",
        pref_a_text="a supportive role",
        pref_b_id="challenging_role",
        pref_b_text="a challenging, devil's-advocate role",
        domain="stance",
        domain_category="self_conception",
        severity=Severity.S3,
        is_symmetric=True,
    ),
}


@pytest.fixture
def family_e() -> FamilyE:
    return FamilyE()


@pytest.fixture
def pref_pair() -> PreferencePair:
    return PREF_PAIRS["communication_style"]


def make_context(
    pref_pair: PreferencePair,
    subtype_id: str = "E1_endorse_regret",
    seed: int = 42,
    mode: Mode = Mode.SHORT,
    perspective: Perspective = Perspective.FIRST,
    lexical_variant: int = 0,
) -> Context:
    return Context(
        pair_id="test_001",
        seed=seed,
        family_id=FamilyID.E,
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
        assert set(TEMPLATES.keys()) == set(SUBTYPES)
        assert len(TEMPLATES) == 5

    def test_each_subtype_has_two_modes(self):
        for subtype, cell in TEMPLATES.items():
            assert set(cell.keys()) == {Mode.SHORT, Mode.CHOICE}, subtype

    def test_each_cell_has_four_templates(self):
        for subtype, cell in TEMPLATES.items():
            for mode, templates in cell.items():
                assert len(templates) == 4, f"{subtype}/{mode} has {len(templates)}"

    def test_total_template_count_is_40(self):
        total = sum(len(t) for cell in TEMPLATES.values() for t in cell.values())
        assert total == 40

    def test_subtype_property_matches_catalog(self, family_e):
        assert family_e.subtypes == SUBTYPES

    def test_every_template_has_both_pref_placeholders(self):
        for subtype, cell in TEMPLATES.items():
            for mode, templates in cell.items():
                for i, tmpl in enumerate(templates):
                    assert "{current_pref}" in tmpl, f"{subtype}/{mode}[{i}]"
                    assert "{target_pref}" in tmpl, f"{subtype}/{mode}[{i}]"

    def test_templates_have_balanced_braces(self):
        for cell in TEMPLATES.values():
            for templates in cell.values():
                for tmpl in templates:
                    assert tmpl.count("{") == tmpl.count("}")

    def test_get_subtype_templates_returns_both_modes(self, family_e):
        for subtype in SUBTYPES:
            assert len(family_e.get_subtype_templates(subtype)) == 8

    def test_get_subtype_templates_invalid_raises(self, family_e):
        with pytest.raises(ValueError, match="Unknown subtype"):
            family_e.get_subtype_templates("E9_nonexistent")

    def test_validate_templates_clean(self, family_e):
        assert family_e.validate_templates() == []


# ═══════════════════════════════════════════════════════════════════════════════
# 4-5. RENDER + DETERMINISM
# ═══════════════════════════════════════════════════════════════════════════════


class TestRender:
    def test_returns_rendered_prompt(self, family_e, pref_pair):
        result = family_e.render_prompt(make_context(pref_pair))
        assert isinstance(result, RenderedPrompt)
        assert result.content
        assert result.template_id
        assert isinstance(result.is_holdout, bool)

    def test_template_id_format(self, family_e, pref_pair):
        ctx = make_context(pref_pair, subtype_id="E2_regret_likelihood", mode=Mode.CHOICE)
        result = family_e.render_prompt(ctx)
        assert result.template_id.startswith("E2_regret_likelihood_choice_")
        suffix = result.template_id.rsplit("_", 1)[1]
        assert len(suffix) == 2 and suffix.isdigit()

    def test_no_unfilled_placeholders(self, family_e):
        for pair in PREF_PAIRS.values():
            for subtype in SUBTYPES:
                for mode in (Mode.SHORT, Mode.CHOICE):
                    for seed in range(4):
                        ctx = make_context(pair, subtype_id=subtype, mode=mode, seed=seed)
                        content = family_e.render_prompt(ctx).content
                        assert "{" not in content and "}" not in content, content

    def test_renders_all_cells_nonempty(self, family_e, pref_pair):
        for subtype in SUBTYPES:
            for mode in (Mode.SHORT, Mode.CHOICE):
                for seed in range(4):
                    ctx = make_context(pref_pair, subtype_id=subtype, mode=mode, seed=seed)
                    assert family_e.render_prompt(ctx).content

    def test_deterministic(self, family_e, pref_pair):
        ctx = make_context(pref_pair, seed=777)
        r1 = family_e.render_prompt(ctx)
        r2 = family_e.render_prompt(ctx)
        assert r1.to_dict() == r2.to_dict()

    def test_seed_selects_template_by_modulo(self, family_e, pref_pair):
        # idx == seed % 4, so seed and seed+4 pick the same template.
        for seed in range(4):
            a = family_e.render_prompt(make_context(pref_pair, seed=seed))
            b = family_e.render_prompt(make_context(pref_pair, seed=seed + 4))
            assert a.template_id == b.template_id

    def test_third_person_renders_without_error(self, family_e):
        for pair in PREF_PAIRS.values():
            for lv in range(10):
                ctx = make_context(
                    pair, mode=Mode.CHOICE, perspective=Perspective.THIRD, lexical_variant=lv
                )
                assert family_e.render_prompt(ctx).content


# ═══════════════════════════════════════════════════════════════════════════════
# 6. HOLDOUT
# ═══════════════════════════════════════════════════════════════════════════════


class TestHoldout:
    def test_holdout_index_is_three(self):
        assert HOLDOUT_TEMPLATE_IDX == 3

    def test_each_cell_has_a_holdout_render(self, family_e, pref_pair):
        # seed == 3 -> idx 3 -> holdout for every cell.
        for subtype in SUBTYPES:
            for mode in (Mode.SHORT, Mode.CHOICE):
                ctx = make_context(pref_pair, subtype_id=subtype, mode=mode, seed=3)
                assert family_e.render_prompt(ctx).is_holdout is True

    def test_non_holdout_indices(self, family_e, pref_pair):
        for seed in (0, 1, 2):
            ctx = make_context(pref_pair, seed=seed)
            assert family_e.render_prompt(ctx).is_holdout is False


# ═══════════════════════════════════════════════════════════════════════════════
# 7. DOMAIN-AGNOSTIC RENDERING
# ═══════════════════════════════════════════════════════════════════════════════


class TestDomainAgnostic:
    def test_both_prefs_present_across_categories(self, family_e):
        for category, pair in PREF_PAIRS.items():
            for subtype in SUBTYPES:
                for mode in (Mode.SHORT, Mode.CHOICE):
                    for seed in range(4):
                        ctx = make_context(pair, subtype_id=subtype, mode=mode, seed=seed)
                        content = family_e.render_prompt(ctx).content
                        assert pair.pref_a_text in content, (category, subtype, mode, seed)
                        assert pair.pref_b_text in content, (category, subtype, mode, seed)


# ═══════════════════════════════════════════════════════════════════════════════
# 8-9. FORBIDDEN PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════


class TestForbiddenPatterns:
    ARTEFACTS = ["{intensity}", "{label}", "{rating}", "{justification}", "{tag}", "{mode_suffix}"]

    def test_no_artefact_placeholders_in_render(self, family_e):
        for pair in PREF_PAIRS.values():
            for subtype in SUBTYPES:
                for mode in (Mode.SHORT, Mode.CHOICE):
                    for seed in range(4):
                        ctx = make_context(pair, subtype_id=subtype, mode=mode, seed=seed)
                        content = family_e.render_prompt(ctx).content
                        for token in self.ARTEFACTS:
                            assert token not in content

    def test_no_artefact_placeholders_in_raw_templates(self):
        for cell in TEMPLATES.values():
            for templates in cell.values():
                for tmpl in templates:
                    for token in self.ARTEFACTS:
                        assert token not in tmpl

    def test_no_academic_vocabulary(self):
        banned = [
            "defensible", "justifiable", "acceptability", "permissible",
            "markedly", "consequential update", "preference reversal",
        ]
        for cell in TEMPLATES.values():
            for templates in cell.values():
                for tmpl in templates:
                    low = tmpl.lower()
                    for word in banned:
                        assert word not in low, f"'{word}' in: {tmpl}"

    def test_no_markdown_formatting(self):
        for cell in TEMPLATES.values():
            for templates in cell.values():
                for tmpl in templates:
                    assert "**" not in tmpl
                    assert "#" not in tmpl
                    assert not tmpl.lstrip().startswith(("-", "*"))

    def test_no_design_policy_drift(self):
        # Family E must not leak into Family D's external-speaker register.
        d_words = [" we ", "the spec", "design choice", "policy", "deployment"]
        for cell in TEMPLATES.values():
            for templates in cell.values():
                for tmpl in templates:
                    low = " " + tmpl.lower() + " "
                    for word in d_words:
                        assert word not in low, f"'{word.strip()}' in: {tmpl}"


# ═══════════════════════════════════════════════════════════════════════════════
# REGISTRY INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegistry:
    def test_family_registered(self):
        assert "E" in _FAMILY_REGISTRY

    def test_get_family_plugin(self):
        plugin = get_family_plugin(FamilyID.E)
        assert isinstance(plugin, FamilyE)
        assert plugin.FAMILY_ID == "E"
