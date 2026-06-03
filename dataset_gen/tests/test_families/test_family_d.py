"""
Tests for Family D: Design / Policy (post-V1-rewrite, conversational templates).

Covers the per-family contract from the Tier 3 shared spec:
1. TEMPLATES has exactly the 5 locked subtypes from catalogs.FAMILY_SUBTYPES[D].
2. Each subtype has exactly 2 modes (SHORT, CHOICE).
3. Each (subtype, mode) cell has exactly 4 templates.
4. render_prompt() returns a valid RenderedPrompt (no `tag` field).
5. Rendering is deterministic.
6. Holdout: index-3 template per cell is held out.
7. Domain-agnostic: renders against 3 pairs from different categories.
8. No forbidden runtime tokens in rendered output.
9. No academic vocabulary in raw template strings.
"""

import re

import pytest

from dataset_gen.src.schema import (
    Context,
    FamilyID,
    Mode,
    Perspective,
    PreferencePair,
    RenderedPrompt,
)
from dataset_gen.src import catalogs
from dataset_gen.src.families.family_d import (
    FamilyD,
    TEMPLATES,
    HOLDOUT_TEMPLATE_IDX,
)
from dataset_gen.src.families.registry import get_family_plugin, import_all_families


LOCKED_SUBTYPES = catalogs.FAMILY_SUBTYPES[FamilyID.D]


# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLE PREFERENCE PAIRS — drawn from 3 different categories / severities,
# matching the per-family spec's render-check table.
# ═══════════════════════════════════════════════════════════════════════════════


def _catalog_pair(category: str, pref_a_id: str) -> PreferencePair:
    """Fetch a real catalog pair by category + pref_a_id (authentic content)."""
    for pair in catalogs.PREFERENCE_CATALOG[category]:
        if pair.pref_a_id == pref_a_id:
            return pair
    raise KeyError(f"No pair '{pref_a_id}' in category '{category}'")


# S1 communication_style, S2 workflow, S3 self_conception.
PAIR_S1 = _catalog_pair("communication_style", "section_headers")
PAIR_S2 = _catalog_pair("workflow", "decompose_problems")
PAIR_S3 = _catalog_pair("self_conception", "supportive_role")

SAMPLE_PAIRS = [PAIR_S1, PAIR_S2, PAIR_S3]


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES / HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def family_d() -> FamilyD:
    return FamilyD()


def make_context(
    pref_pair: PreferencePair,
    subtype_id: str,
    mode: Mode,
    *,
    seed: int = 0,
    perspective: Perspective = Perspective.FIRST,
    lexical_variant: int = 0,
) -> Context:
    return Context(
        pair_id="d_test_0001",
        seed=seed,
        family_id=FamilyID.D,
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
# 1-3. STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════


class TestStructure:
    def test_templates_cover_locked_subtypes(self):
        assert set(TEMPLATES.keys()) == set(LOCKED_SUBTYPES)
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

    def test_every_template_has_both_preferences(self):
        for subtype, cell in TEMPLATES.items():
            for mode in (Mode.SHORT, Mode.CHOICE):
                for i, tpl in enumerate(cell[mode]):
                    assert "{current_pref}" in tpl, f"{subtype}/{mode}/{i}"
                    assert "{target_pref}" in tpl, f"{subtype}/{mode}/{i}"

    def test_get_subtype_templates_returns_all_eight(self, family_d):
        for subtype in LOCKED_SUBTYPES:
            assert len(family_d.get_subtype_templates(subtype)) == 8

    def test_get_subtype_templates_rejects_unknown(self, family_d):
        with pytest.raises(ValueError):
            family_d.get_subtype_templates("D9_nonexistent")

    def test_subtypes_property_sources_from_catalog(self, family_d):
        assert family_d.subtypes == LOCKED_SUBTYPES

    def test_validate_templates_clean(self, family_d):
        assert family_d.validate_templates() == []


# ═══════════════════════════════════════════════════════════════════════════════
# 4-5. RENDER CONTRACT
# ═══════════════════════════════════════════════════════════════════════════════


class TestRenderContract:
    def test_returns_rendered_prompt(self, family_d):
        ctx = make_context(PAIR_S1, "D1_design_principle", Mode.SHORT)
        result = family_d.render_prompt(ctx)
        assert isinstance(result, RenderedPrompt)
        assert result.content
        assert result.template_id
        assert isinstance(result.is_holdout, bool)

    def test_no_tag_field(self):
        # The V1 rewrite removed the format `tag`.
        assert "tag" not in RenderedPrompt.__dataclass_fields__

    def test_template_id_includes_subtype_mode_index(self, family_d):
        ctx = make_context(PAIR_S1, "D2_policy_parameter", Mode.CHOICE, seed=1)
        result = family_d.render_prompt(ctx)
        # "{subtype}_{mode}_{idx:02d}"
        assert result.template_id.startswith("D2_policy_parameter_choice_")
        suffix = result.template_id.rsplit("_", 1)[1]
        assert len(suffix) == 2 and suffix.isdigit()

    def test_no_unfilled_placeholders(self, family_d):
        for subtype in LOCKED_SUBTYPES:
            for mode in (Mode.SHORT, Mode.CHOICE):
                for seed in range(4):
                    ctx = make_context(PAIR_S2, subtype, mode, seed=seed)
                    content = family_d.render_prompt(ctx).content
                    assert "{" not in content and "}" not in content, content

    def test_deterministic(self, family_d):
        ctx = make_context(PAIR_S3, "D5_spec_revision", Mode.SHORT, seed=7)
        r1 = family_d.render_prompt(ctx)
        r2 = family_d.render_prompt(ctx)
        assert r1.to_dict() == r2.to_dict()

    def test_third_person_renders_cleanly(self, family_d):
        # All cells, every lexical variant, THIRD perspective: must render
        # without leftover braces and must not leave a bare "you"/"your".
        for subtype in LOCKED_SUBTYPES:
            for mode in (Mode.SHORT, Mode.CHOICE):
                for lv in range(10):
                    for seed in range(4):
                        ctx = make_context(
                            PAIR_S1, subtype, mode,
                            seed=seed, perspective=Perspective.THIRD,
                            lexical_variant=lv,
                        )
                        content = family_d.render_prompt(ctx).content
                        assert content
                        assert "{" not in content and "}" not in content
                        lowered = f" {content.lower()} "
                        assert " you " not in lowered, content
                        assert " your " not in lowered, content
                        _assert_no_midsentence_capital_subject(content)


# Third-person subjects (see catalogs.THIRD_PERSON_AGENTS). A capitalized
# "The <agent>" must only appear at the start of a sentence — the base class's
# IGNORECASE verb-conjugation regex can otherwise capitalize a mid-sentence
# "you <verb>", which would read as broken prose.
_SUBJECT_RE = re.compile(r"\bThe (assistant|model|AI|agent|system|program|chatbot|bot)\b")


def _assert_no_midsentence_capital_subject(content: str) -> None:
    for match in _SUBJECT_RE.finditer(content):
        j = match.start() - 1
        while j >= 0 and content[j] == " ":
            j -= 1
        if j >= 0:
            assert content[j] in ".?!", (
                f"mid-sentence capitalized subject in: {content!r}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. HOLDOUT
# ═══════════════════════════════════════════════════════════════════════════════


class TestHoldout:
    def test_holdout_idx_in_range(self):
        assert 0 <= HOLDOUT_TEMPLATE_IDX < 4

    def test_index_three_is_holdout(self, family_d):
        # seed % 4 == 3 selects the holdout template in every cell.
        for subtype in LOCKED_SUBTYPES:
            for mode in (Mode.SHORT, Mode.CHOICE):
                ctx = make_context(PAIR_S1, subtype, mode, seed=HOLDOUT_TEMPLATE_IDX)
                result = family_d.render_prompt(ctx)
                assert result.is_holdout is True
                assert result.template_id.endswith("_03")

    def test_non_holdout_indices(self, family_d):
        for subtype in LOCKED_SUBTYPES:
            for mode in (Mode.SHORT, Mode.CHOICE):
                for seed in (0, 1, 2):
                    ctx = make_context(PAIR_S1, subtype, mode, seed=seed)
                    assert family_d.render_prompt(ctx).is_holdout is False

    def test_each_cell_has_exactly_one_holdout(self, family_d):
        # Over the 4 selectable indices, exactly one is a holdout.
        for subtype in LOCKED_SUBTYPES:
            for mode in (Mode.SHORT, Mode.CHOICE):
                holdouts = [
                    family_d.render_prompt(
                        make_context(PAIR_S1, subtype, mode, seed=seed)
                    ).is_holdout
                    for seed in range(4)
                ]
                assert sum(holdouts) == 1, f"{subtype}/{mode}"


# ═══════════════════════════════════════════════════════════════════════════════
# 7. DOMAIN-AGNOSTIC RENDERING
# ═══════════════════════════════════════════════════════════════════════════════


class TestDomainAgnostic:
    def test_renders_against_three_categories(self, family_d):
        for pair in SAMPLE_PAIRS:
            for subtype in LOCKED_SUBTYPES:
                for mode in (Mode.SHORT, Mode.CHOICE):
                    for seed in range(4):
                        ctx = make_context(pair, subtype, mode, seed=seed)
                        content = family_d.render_prompt(ctx).content
                        assert content
                        assert pair.pref_a_text in content, (subtype, mode, content)
                        assert pair.pref_b_text in content, (subtype, mode, content)

    def test_sample_pairs_span_distinct_categories(self):
        cats = {p.domain_category for p in SAMPLE_PAIRS}
        assert cats == {"communication_style", "workflow", "self_conception"}


# ═══════════════════════════════════════════════════════════════════════════════
# 8-9. FORBIDDEN PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════


def _all_raw_templates():
    for cell in TEMPLATES.values():
        for mode in (Mode.SHORT, Mode.CHOICE):
            yield from cell[mode]


class TestForbiddenPatterns:
    def test_no_old_format_tokens_in_output(self, family_d):
        bad = ["{intensity}", "{label}", "{rating}", "{justification}", "{tag}",
               "{mode_suffix}"]
        for pair in SAMPLE_PAIRS:
            for subtype in LOCKED_SUBTYPES:
                for mode in (Mode.SHORT, Mode.CHOICE):
                    for seed in range(4):
                        ctx = make_context(pair, subtype, mode, seed=seed)
                        content = family_d.render_prompt(ctx).content
                        for token in bad:
                            assert token not in content

    def test_no_academic_vocab_in_raw_templates(self):
        forbidden = [
            "defensible", "justifiable", "acceptability", "permissible",
            "markedly", "consequential update", "preference reversal",
            # Family D gotcha #5: avoid corporate/managerial jargon.
            "stakeholder", "configuration parameter", "operational constraint",
        ]
        for tpl in _all_raw_templates():
            low = tpl.lower()
            for word in forbidden:
                assert word not in low, f"'{word}' in: {tpl}"

    def test_no_markdown_in_raw_templates(self):
        for tpl in _all_raw_templates():
            assert "**" not in tpl
            assert "##" not in tpl
            # No leading bullet markers.
            for line in tpl.splitlines():
                stripped = line.lstrip()
                assert not stripped.startswith("- ")
                assert not stripped.startswith("* ")

    def test_no_numeric_scales(self):
        for tpl in _all_raw_templates():
            low = tpl.lower()
            assert "scale of" not in low
            assert "1 to 7" not in low
            assert "1 to 5" not in low


# ═══════════════════════════════════════════════════════════════════════════════
# REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegistry:
    def test_family_d_is_registered(self):
        import_all_families()
        plugin = get_family_plugin(FamilyID.D)
        assert isinstance(plugin, FamilyD)
