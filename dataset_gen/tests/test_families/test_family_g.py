"""Tests for Family G: distributional shift (post-V1-rewrite contract).

Covers the (subtype x mode) cell structure, render contract, determinism,
holdout convention, domain-agnostic rendering, and forbidden-pattern scans, per
specs/data_gen/tier_3_family_template_drafting_shared_spec.md.
"""

import re

import pytest

from dataset_gen.src.families.family_g import (
    FamilyG,
    TEMPLATES,
    HOLDOUT_TEMPLATE_IDX,
)
from dataset_gen.src.families.registry import (
    get_family_plugin,
    import_all_families,
    _FAMILY_REGISTRY,
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


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES / HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

SUBTYPES = catalogs.FAMILY_SUBTYPES[FamilyID.G]


@pytest.fixture
def family_g() -> FamilyG:
    return FamilyG()


def make_pref_pair(
    a_id: str,
    a_text: str,
    b_id: str,
    b_text: str,
    domain: str,
    domain_category: str,
    severity: Severity,
) -> PreferencePair:
    return PreferencePair(
        pref_a_id=a_id,
        pref_a_text=a_text,
        pref_b_id=b_id,
        pref_b_text=b_text,
        domain=domain,
        domain_category=domain_category,
        severity=severity,
        is_symmetric=True,
    )


# Three sample pairs from different categories (see family G spec render table).
PAIR_S1 = make_pref_pair(
    "fmt_headers", "section headers",
    "fmt_continuous", "continuous text",
    "formatting", "communication_style", Severity.S1,
)
PAIR_S2 = make_pref_pair(
    "decompose", "decomposing problems into parts",
    "whole", "tackling problems whole",
    "method", "workflow", Severity.S2,
)
PAIR_S3 = make_pref_pair(
    "supportive", "a supportive role",
    "challenging", "a challenging role",
    "stance", "self_conception", Severity.S3,
)


def make_context(
    pref_pair: PreferencePair = PAIR_S1,
    seed: int = 42,
    subtype_id: str = "G1_partial_shift",
    mode: Mode = Mode.SHORT,
    perspective: Perspective = Perspective.FIRST,
    lexical_variant: int = 0,
    severity: Severity = Severity.S1,
) -> Context:
    return Context(
        pair_id=f"g_{seed:06d}",
        seed=seed,
        family_id=FamilyID.G,
        subtype_id=subtype_id,
        severity=severity,
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
# CELL STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════


class TestCellStructure:
    def test_five_subtypes_match_catalog(self):
        assert list(TEMPLATES.keys()) == SUBTYPES
        assert len(TEMPLATES) == 5

    def test_each_subtype_has_two_modes(self):
        for subtype, cell in TEMPLATES.items():
            assert set(cell.keys()) == {Mode.SHORT, Mode.CHOICE}, subtype

    def test_each_cell_has_four_templates(self):
        for subtype, cell in TEMPLATES.items():
            for mode in (Mode.SHORT, Mode.CHOICE):
                assert len(cell[mode]) == 4, f"{subtype}/{mode}"

    def test_total_template_count_is_40(self):
        total = sum(len(cell[m]) for cell in TEMPLATES.values()
                    for m in (Mode.SHORT, Mode.CHOICE))
        assert total == 40

    def test_subtypes_property_matches_catalog(self, family_g: FamilyG):
        assert family_g.subtypes == SUBTYPES

    def test_no_template_is_empty(self):
        for subtype, cell in TEMPLATES.items():
            for mode in (Mode.SHORT, Mode.CHOICE):
                for i, t in enumerate(cell[mode]):
                    assert t.strip(), f"{subtype}/{mode}/{i} empty"

    def test_balanced_braces(self, family_g: FamilyG):
        assert family_g.validate_templates() == []

    def test_every_template_has_both_pref_placeholders(self):
        for subtype, cell in TEMPLATES.items():
            for mode in (Mode.SHORT, Mode.CHOICE):
                for i, t in enumerate(cell[mode]):
                    assert "{current_pref}" in t, f"{subtype}/{mode}/{i}"
                    assert "{target_pref}" in t, f"{subtype}/{mode}/{i}"


# ═══════════════════════════════════════════════════════════════════════════════
# RENDER CONTRACT
# ═══════════════════════════════════════════════════════════════════════════════


class TestRenderContract:
    def test_returns_rendered_prompt(self, family_g: FamilyG):
        result = family_g.render_prompt(make_context())
        assert isinstance(result, RenderedPrompt)

    def test_has_content_template_id_is_holdout(self, family_g: FamilyG):
        result = family_g.render_prompt(make_context())
        assert result.content
        assert result.template_id
        assert isinstance(result.is_holdout, bool)

    def test_template_id_format_includes_mode(self, family_g: FamilyG):
        for subtype in SUBTYPES:
            for mode in (Mode.SHORT, Mode.CHOICE):
                ctx = make_context(subtype_id=subtype, mode=mode)
                result = family_g.render_prompt(ctx)
                prefix = f"{subtype}_{mode.value}_"
                assert result.template_id.startswith(prefix), result.template_id
                suffix = result.template_id[len(prefix):]
                assert len(suffix) == 2 and suffix.isdigit()

    def test_placeholders_filled(self, family_g: FamilyG):
        result = family_g.render_prompt(make_context())
        assert "{" not in result.content and "}" not in result.content

    def test_pref_texts_present(self, family_g: FamilyG):
        result = family_g.render_prompt(make_context(PAIR_S1))
        assert "section headers" in result.content
        assert "continuous text" in result.content

    def test_invalid_subtype_raises(self, family_g: FamilyG):
        with pytest.raises(ValueError):
            family_g.get_subtype_templates("G9_bogus")

    def test_all_subtypes_all_modes_render(self, family_g: FamilyG):
        for subtype in SUBTYPES:
            for mode in (Mode.SHORT, Mode.CHOICE):
                for perspective in (Perspective.FIRST, Perspective.THIRD):
                    ctx = make_context(
                        subtype_id=subtype, mode=mode, perspective=perspective
                    )
                    result = family_g.render_prompt(ctx)
                    assert result.content


# ═══════════════════════════════════════════════════════════════════════════════
# DETERMINISM
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeterminism:
    def test_same_context_same_output(self, family_g: FamilyG):
        r1 = family_g.render_prompt(make_context(seed=7))
        r2 = family_g.render_prompt(make_context(seed=7))
        assert r1.to_dict() == r2.to_dict()

    def test_different_seeds_use_multiple_templates(self, family_g: FamilyG):
        ids = set()
        for seed in range(100):
            ids.add(family_g.render_prompt(make_context(seed=seed)).template_id)
        assert len(ids) > 1

    def test_all_lexical_variants_render(self, family_g: FamilyG):
        for lv in range(10):
            for mode in (Mode.SHORT, Mode.CHOICE):
                ctx = make_context(lexical_variant=lv, mode=mode)
                result = family_g.render_prompt(ctx)
                assert "{" not in result.content and "}" not in result.content


# ═══════════════════════════════════════════════════════════════════════════════
# HOLDOUT
# ═══════════════════════════════════════════════════════════════════════════════


class TestHoldout:
    def test_holdout_idx_is_three(self):
        assert HOLDOUT_TEMPLATE_IDX == 3

    def test_each_cell_has_a_holdout_template(self, family_g: FamilyG):
        # Iterate seeds until every index 0-3 has been observed for a cell, and
        # confirm exactly index 3 maps to is_holdout=True.
        for subtype in SUBTYPES:
            for mode in (Mode.SHORT, Mode.CHOICE):
                seen_holdout_idx = set()
                seen_idx = set()
                for seed in range(200):
                    ctx = make_context(subtype_id=subtype, mode=mode, seed=seed)
                    result = family_g.render_prompt(ctx)
                    idx = int(result.template_id.rsplit("_", 1)[1])
                    seen_idx.add(idx)
                    if result.is_holdout:
                        seen_holdout_idx.add(idx)
                assert seen_idx == {0, 1, 2, 3}, f"{subtype}/{mode}: {seen_idx}"
                assert seen_holdout_idx == {3}, f"{subtype}/{mode}: {seen_holdout_idx}"


# ═══════════════════════════════════════════════════════════════════════════════
# DOMAIN-AGNOSTIC RENDERING
# ═══════════════════════════════════════════════════════════════════════════════


class TestDomainAgnostic:
    @pytest.mark.parametrize("pair", [PAIR_S1, PAIR_S2, PAIR_S3])
    def test_renders_against_pairs_from_different_categories(self, family_g, pair):
        for subtype in SUBTYPES:
            for mode in (Mode.SHORT, Mode.CHOICE):
                for seed in range(8):  # cover all 4 templates per cell
                    ctx = make_context(
                        pref_pair=pair,
                        subtype_id=subtype,
                        mode=mode,
                        seed=seed,
                        severity=pair.severity,
                    )
                    result = family_g.render_prompt(ctx)
                    assert result.content
                    assert pair.pref_a_text in result.content
                    assert pair.pref_b_text in result.content


# ═══════════════════════════════════════════════════════════════════════════════
# FORBIDDEN PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════


class TestForbiddenPatterns:
    def test_no_old_artifact_placeholders_in_output(self, family_g: FamilyG):
        artifacts = ["{intensity}", "{label}", "{rating}", "{justification}", "{tag}"]
        for subtype in SUBTYPES:
            for mode in (Mode.SHORT, Mode.CHOICE):
                for seed in range(8):
                    ctx = make_context(subtype_id=subtype, mode=mode, seed=seed)
                    out = family_g.render_prompt(ctx).content
                    for art in artifacts:
                        assert art not in out

    def test_no_academic_vocab_in_raw_templates(self):
        banned = [
            "defensible", "justifiable", "acceptability", "permissible",
            "markedly", "consequential update", "preference reversal",
        ]
        for cell in TEMPLATES.values():
            for mode in (Mode.SHORT, Mode.CHOICE):
                for t in cell[mode]:
                    low = t.lower()
                    for word in banned:
                        assert word not in low, f"'{word}' in: {t}"

    def test_no_numeric_scale_in_raw_templates(self):
        # Word frequencies ("nine times out of ten") are fine; numeric 1-N
        # scales are not.
        for cell in TEMPLATES.values():
            for mode in (Mode.SHORT, Mode.CHOICE):
                for t in cell[mode]:
                    assert "scale of" not in t.lower(), t
                    assert not re.search(r"\b1\s*(?:to|-)\s*[0-9]\b", t), t

    def test_no_markdown_in_raw_templates(self):
        for cell in TEMPLATES.values():
            for mode in (Mode.SHORT, Mode.CHOICE):
                for t in cell[mode]:
                    assert "**" not in t
                    assert "#" not in t
                    assert not t.lstrip().startswith(("- ", "* "))

    def test_no_explicit_assistant_reference_in_raw_templates(self):
        # Templates are second-person; the third-person substitution layer adds
        # "the assistant". Raw templates must not hardcode it.
        for cell in TEMPLATES.values():
            for mode in (Mode.SHORT, Mode.CHOICE):
                for t in cell[mode]:
                    low = t.lower()
                    assert not re.search(r"\bthe assistant\b", low), t
                    assert not re.search(r"\bthe ai\b", low), t
                    assert not re.search(r"\bthe model\b", low), t


# ═══════════════════════════════════════════════════════════════════════════════
# PERSPECTIVE
# ═══════════════════════════════════════════════════════════════════════════════


class TestPerspective:
    def test_third_person_substitution(self, family_g: FamilyG):
        ctx = make_context(perspective=Perspective.THIRD)
        result = family_g.render_prompt(ctx)
        assert "assistant" in result.content.lower()

    def test_perspective_changes_output(self, family_g: FamilyG):
        first = family_g.render_prompt(make_context(perspective=Perspective.FIRST))
        third = family_g.render_prompt(make_context(perspective=Perspective.THIRD))
        assert first.content != third.content


# ═══════════════════════════════════════════════════════════════════════════════
# REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegistry:
    def test_registered_under_letter_id(self):
        import_all_families()
        assert "G" in _FAMILY_REGISTRY

    def test_get_family_plugin_by_enum(self):
        import_all_families()
        plugin = get_family_plugin(FamilyID.G)
        assert isinstance(plugin, FamilyG)

    def test_registered_instance_renders(self):
        import_all_families()
        plugin = get_family_plugin(FamilyID.G)
        result = plugin.render_prompt(make_context())
        assert isinstance(result, RenderedPrompt)
