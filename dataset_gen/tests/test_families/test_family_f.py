"""Tests for Family F (Value Tradeoff) templates.

Covers the per-family contract from the Tier 3 shared spec:
1. TEMPLATES has exactly 5 subtypes matching catalogs.FAMILY_SUBTYPES[FamilyID.F].
2. Each subtype has exactly 2 modes (SHORT, CHOICE).
3. Each (subtype, mode) cell has exactly 4 templates.
4. Rendering produces a valid RenderedPrompt.
5. Rendering is deterministic.
6. At least one template per cell is held out.
7. Domain-agnostic rendering against pairs from different categories.
8. No old format-artefact placeholders survive in rendered output.
9. No academic / survey-paper vocabulary in the raw templates.
"""

import pytest

from dataset_gen.src import catalogs
from dataset_gen.src.families.family_f import (
    TEMPLATES,
    HOLDOUT_TEMPLATE_IDX,
    FamilyF,
)
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
def plugin() -> FamilyF:
    return FamilyF()


# Three preference pairs drawn from DIFFERENT categories, matching the per-family
# spec's render-check table (communication_style / workflow / self_conception).
SAMPLE_PAIRS = {
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

F_SUBTYPES = catalogs.FAMILY_SUBTYPES[FamilyID.F]


def make_context(
    subtype_id: str,
    mode: Mode,
    pair: PreferencePair,
    *,
    seed: int = 0,
    lexical_variant: int = 0,
    perspective: Perspective = Perspective.FIRST,
) -> Context:
    return Context(
        pair_id="f_test_0001",
        seed=seed,
        family_id=FamilyID.F,
        subtype_id=subtype_id,
        severity=pair.severity,
        mode=mode,
        perspective=perspective,
        pref_pair=pair,
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
        assert set(TEMPLATES.keys()) == set(F_SUBTYPES)
        assert len(TEMPLATES) == 5

    def test_each_subtype_has_two_modes(self):
        for subtype_id, cell in TEMPLATES.items():
            assert set(cell.keys()) == {Mode.SHORT, Mode.CHOICE}, subtype_id

    def test_each_cell_has_four_templates(self):
        for subtype_id, cell in TEMPLATES.items():
            for mode, templates in cell.items():
                assert len(templates) == 4, f"{subtype_id}/{mode}"

    def test_total_template_count_is_forty(self):
        total = sum(len(t) for cell in TEMPLATES.values() for t in cell.values())
        assert total == 40

    def test_templates_are_nonempty_strings(self):
        for cell in TEMPLATES.values():
            for templates in cell.values():
                for t in templates:
                    assert isinstance(t, str) and t.strip()

    def test_braces_balanced(self):
        for cell in TEMPLATES.values():
            for templates in cell.values():
                for t in templates:
                    assert t.count("{") == t.count("}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. RENDERING PRODUCES A VALID RenderedPrompt
# ═══════════════════════════════════════════════════════════════════════════════


class TestRendering:
    def test_returns_rendered_prompt(self, plugin):
        ctx = make_context("F1_tradeoff", Mode.SHORT, SAMPLE_PAIRS["communication_style"])
        result = plugin.render_prompt(ctx)
        assert isinstance(result, RenderedPrompt)
        assert result.content
        assert result.template_id
        assert isinstance(result.is_holdout, bool)

    def test_template_id_format(self, plugin):
        ctx = make_context("F2_sacrifice", Mode.CHOICE, SAMPLE_PAIRS["workflow"], seed=5)
        result = plugin.render_prompt(ctx)
        # "{subtype_id}_{mode}_{idx:02d}"
        assert result.template_id.startswith("F2_sacrifice_choice_")
        suffix = result.template_id.rsplit("_", 1)[1]
        assert len(suffix) == 2 and suffix.isdigit()

    def test_no_unfilled_placeholders(self, plugin):
        for subtype_id in F_SUBTYPES:
            for mode in (Mode.SHORT, Mode.CHOICE):
                for seed in range(4):
                    ctx = make_context(
                        subtype_id, mode, SAMPLE_PAIRS["workflow"], seed=seed
                    )
                    content = plugin.render_prompt(ctx).content
                    assert "{" not in content and "}" not in content, (
                        f"{subtype_id}/{mode}/seed={seed}: {content}"
                    )

    def test_all_subtypes_modes_seeds_render(self, plugin):
        for subtype_id in F_SUBTYPES:
            for mode in (Mode.SHORT, Mode.CHOICE):
                for seed in range(8):
                    for perspective in (Perspective.FIRST, Perspective.THIRD):
                        ctx = make_context(
                            subtype_id,
                            mode,
                            SAMPLE_PAIRS["self_conception"],
                            seed=seed,
                            perspective=perspective,
                        )
                        assert plugin.render_prompt(ctx).content

    def test_get_subtype_templates_returns_eight(self, plugin):
        for subtype_id in F_SUBTYPES:
            assert len(plugin.get_subtype_templates(subtype_id)) == 8

    def test_get_subtype_templates_rejects_unknown(self, plugin):
        with pytest.raises(ValueError):
            plugin.get_subtype_templates("F9_nonexistent")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. DETERMINISM
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeterminism:
    def test_same_context_same_output(self, plugin):
        ctx = make_context(
            "F3_context_dependent", Mode.SHORT, SAMPLE_PAIRS["communication_style"],
            seed=42, lexical_variant=4,
        )
        r1 = plugin.render_prompt(ctx)
        r2 = plugin.render_prompt(ctx)
        assert r1.to_dict() == r2.to_dict()

    def test_seed_selects_template_by_modulo(self, plugin):
        # idx = seed % 4, so seed and seed+4 select the same template.
        for seed in range(4):
            ctx_a = make_context("F4_cost_benefit", Mode.CHOICE, SAMPLE_PAIRS["workflow"], seed=seed)
            ctx_b = make_context("F4_cost_benefit", Mode.CHOICE, SAMPLE_PAIRS["workflow"], seed=seed + 4)
            assert plugin.render_prompt(ctx_a).template_id == plugin.render_prompt(ctx_b).template_id


# ═══════════════════════════════════════════════════════════════════════════════
# 6. HOLDOUT
# ═══════════════════════════════════════════════════════════════════════════════


class TestHoldout:
    def test_holdout_idx_in_range(self):
        assert 0 <= HOLDOUT_TEMPLATE_IDX <= 3

    def test_each_cell_has_a_holdout(self, plugin):
        # The holdout template is reachable: rendering with seed == HOLDOUT idx
        # marks is_holdout True for every cell.
        for subtype_id in F_SUBTYPES:
            for mode in (Mode.SHORT, Mode.CHOICE):
                ctx = make_context(
                    subtype_id, mode, SAMPLE_PAIRS["workflow"], seed=HOLDOUT_TEMPLATE_IDX
                )
                assert plugin.render_prompt(ctx).is_holdout is True

    def test_non_holdout_indices_are_not_holdout(self, plugin):
        for subtype_id in F_SUBTYPES:
            for mode in (Mode.SHORT, Mode.CHOICE):
                for seed in range(4):
                    if seed == HOLDOUT_TEMPLATE_IDX:
                        continue
                    ctx = make_context(subtype_id, mode, SAMPLE_PAIRS["workflow"], seed=seed)
                    assert plugin.render_prompt(ctx).is_holdout is False


# ═══════════════════════════════════════════════════════════════════════════════
# 7. DOMAIN-AGNOSTIC RENDERING (>= 3 pairs, different categories)
# ═══════════════════════════════════════════════════════════════════════════════


class TestDomainAgnostic:
    def test_renders_against_all_sample_pairs(self, plugin):
        for category, pair in SAMPLE_PAIRS.items():
            for subtype_id in F_SUBTYPES:
                for mode in (Mode.SHORT, Mode.CHOICE):
                    for seed in range(4):
                        ctx = make_context(subtype_id, mode, pair, seed=seed)
                        content = plugin.render_prompt(ctx).content
                        assert content, f"{category}/{subtype_id}/{mode}"
                        assert pair.pref_a_text in content
                        assert pair.pref_b_text in content

    def test_categories_are_distinct(self):
        cats = {p.domain_category for p in SAMPLE_PAIRS.values()}
        assert len(cats) >= 3


# ═══════════════════════════════════════════════════════════════════════════════
# 8. NO OLD FORMAT-ARTEFACT PLACEHOLDERS IN RENDERED OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════


class TestNoForbiddenArtefacts:
    FORBIDDEN = ["{intensity}", "{label}", "{rating}", "{justification}", "{tag}", "{mode_suffix}"]

    def test_rendered_output_clean(self, plugin):
        for subtype_id in F_SUBTYPES:
            for mode in (Mode.SHORT, Mode.CHOICE):
                for seed in range(4):
                    content = plugin.render_prompt(
                        make_context(subtype_id, mode, SAMPLE_PAIRS["communication_style"], seed=seed)
                    ).content
                    for token in self.FORBIDDEN:
                        assert token not in content

    def test_raw_templates_have_no_artefact_placeholders(self):
        for cell in TEMPLATES.values():
            for templates in cell.values():
                for t in templates:
                    for token in self.FORBIDDEN:
                        assert token not in t


# ═══════════════════════════════════════════════════════════════════════════════
# 9. NO ACADEMIC / SURVEY-PAPER VOCABULARY, NO MARKDOWN, NO NUMERIC SCALES
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegister:
    ACADEMIC = [
        "defensible", "justifiable", "acceptability", "permissible", "markedly",
        "consequential update", "preference reversal", "opportunity cost",
        "marginal utility",
    ]

    def test_no_academic_vocab(self):
        for cell in TEMPLATES.values():
            for templates in cell.values():
                for t in templates:
                    low = t.lower()
                    for word in self.ACADEMIC:
                        assert word not in low, f"'{word}' in: {t}"

    def test_no_markdown(self):
        for cell in TEMPLATES.values():
            for templates in cell.values():
                for t in templates:
                    assert "**" not in t and "##" not in t
                    assert not t.lstrip().startswith(("-", "*", "#"))

    def test_no_numeric_scales(self):
        # Tradeoff templates must use words, not numeric rating scales.
        for cell in TEMPLATES.values():
            for templates in cell.values():
                for t in templates:
                    low = t.lower()
                    assert "scale of" not in low
                    assert "1 to 7" not in low and "1 to 5" not in low
                    assert not any(ch.isdigit() for ch in t)
