"""Tests for Family B: implicit comparative (post-V1-rewrite contract).

Covers the nine checks from the Tier 3 shared spec:
1. TEMPLATES has exactly the 5 locked subtypes from FAMILY_SUBTYPES[B].
2. Each subtype has exactly 2 modes (SHORT, CHOICE).
3. Each (subtype, mode) cell has exactly 4 templates.
4. Render against a sample Context -> RenderedPrompt(content, template_id, is_holdout).
5. Determinism: same Context twice -> same content.
6. Holdout: at least one template per cell is is_holdout=True.
7. Domain-agnostic: render against 3 pairs from different categories.
8. No forbidden render-time artefacts ({intensity}, {label}, {rating}, ...).
9. No academic vocabulary in raw template strings.
"""

import pytest

from dataset_gen.src.families.family_b import (
    FamilyB,
    TEMPLATES,
    HOLDOUT_TEMPLATE_IDX,
)
from dataset_gen.src.families.base import FamilyPlugin
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


SUBTYPES = catalogs.FAMILY_SUBTYPES[FamilyID.B]
MODES = [Mode.SHORT, Mode.CHOICE]


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES / HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def family_b() -> FamilyB:
    return FamilyB()


# Three sample pairs drawn from DIFFERENT categories (per the family spec render
# check): communication_style (S1), workflow (S2), self_conception (S3).
SAMPLE_PAIRS = [
    PreferencePair(
        pref_a_id="section_headers",
        pref_a_text="headers and sections",
        pref_b_id="continuous_text",
        pref_b_text="continuous text",
        domain="structure",
        domain_category="communication_style",
        severity=Severity.S1,
        is_symmetric=True,
    ),
    PreferencePair(
        pref_a_id="decompose_problems",
        pref_a_text="breaking problems into parts",
        pref_b_id="tackle_whole",
        pref_b_text="tackling problems as a whole",
        domain="decomposition",
        domain_category="workflow",
        severity=Severity.S2,
        is_symmetric=True,
    ),
    PreferencePair(
        pref_a_id="supportive_role",
        pref_a_text="a supportive role",
        pref_b_id="challenging_role",
        pref_b_text="a challenging, devil's-advocate role",
        domain="stance",
        domain_category="self_conception",
        severity=Severity.S3,
        is_symmetric=True,
    ),
]


def make_context(
    pref_pair: PreferencePair = None,
    subtype_id: str = "B1_two_futures",
    seed: int = 12345,
    mode: Mode = Mode.SHORT,
    perspective: Perspective = Perspective.FIRST,
    lexical_variant: int = 0,
    severity: Severity = Severity.S1,
) -> Context:
    if pref_pair is None:
        pref_pair = SAMPLE_PAIRS[0]
    return Context(
        pair_id=f"test_pair_{seed}",
        seed=seed,
        family_id=FamilyID.B,
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
# 1-3. STRUCTURE: subtypes, modes, cell sizes
# ═══════════════════════════════════════════════════════════════════════════════


class TestStructure:
    def test_templates_have_exactly_five_locked_subtypes(self):
        assert list(TEMPLATES.keys()) == SUBTYPES
        assert len(TEMPLATES) == 5

    def test_subtypes_property_matches_catalog(self, family_b: FamilyB):
        assert family_b.subtypes == SUBTYPES

    @pytest.mark.parametrize("subtype", SUBTYPES)
    def test_each_subtype_has_exactly_two_modes(self, subtype):
        assert set(TEMPLATES[subtype].keys()) == {Mode.SHORT, Mode.CHOICE}

    @pytest.mark.parametrize("subtype", SUBTYPES)
    @pytest.mark.parametrize("mode", MODES)
    def test_each_cell_has_exactly_four_templates(self, subtype, mode):
        assert len(TEMPLATES[subtype][mode]) == 4

    def test_total_template_count_is_forty(self):
        total = sum(len(TEMPLATES[s][m]) for s in SUBTYPES for m in MODES)
        assert total == 40


# ═══════════════════════════════════════════════════════════════════════════════
# 4-5. RENDERING + DETERMINISM
# ═══════════════════════════════════════════════════════════════════════════════


class TestRendering:
    def test_returns_rendered_prompt(self, family_b: FamilyB):
        result = family_b.render_prompt(make_context())
        assert isinstance(result, RenderedPrompt)
        assert result.content
        assert result.template_id
        assert isinstance(result.is_holdout, bool)

    def test_template_id_format(self, family_b: FamilyB):
        ctx = make_context(subtype_id="B2_trajectory", mode=Mode.CHOICE, seed=5)
        result = family_b.render_prompt(ctx)
        # "{subtype_id}_{mode}_{idx:02d}"
        assert result.template_id.startswith("B2_trajectory_choice_")
        suffix = result.template_id.rsplit("_", 1)[1]
        assert len(suffix) == 2 and suffix.isdigit()

    def test_no_leftover_placeholders(self, family_b: FamilyB):
        for subtype in SUBTYPES:
            for mode in MODES:
                for seed in range(4):
                    ctx = make_context(subtype_id=subtype, mode=mode, seed=seed)
                    out = family_b.render_prompt(ctx).content
                    assert "{" not in out and "}" not in out, (
                        f"leftover placeholder in {subtype}/{mode}/{seed}: {out}"
                    )

    def test_deterministic(self, family_b: FamilyB):
        ctx = make_context(seed=777)
        r1 = family_b.render_prompt(ctx)
        r2 = family_b.render_prompt(ctx)
        assert r1.to_dict() == r2.to_dict()

    def test_all_lexical_variants_render(self, family_b: FamilyB):
        for lv in range(10):
            for subtype in SUBTYPES:
                for mode in MODES:
                    ctx = make_context(
                        subtype_id=subtype, mode=mode, lexical_variant=lv, seed=lv
                    )
                    assert family_b.render_prompt(ctx).content

    def test_third_person_renders(self, family_b: FamilyB):
        for subtype in SUBTYPES:
            for mode in MODES:
                ctx = make_context(
                    subtype_id=subtype, mode=mode, perspective=Perspective.THIRD
                )
                out = family_b.render_prompt(ctx).content
                assert out
                assert "you" not in out.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# 6. HOLDOUT
# ═══════════════════════════════════════════════════════════════════════════════


class TestHoldout:
    def test_holdout_idx_is_three(self):
        assert HOLDOUT_TEMPLATE_IDX == 3

    @pytest.mark.parametrize("subtype", SUBTYPES)
    @pytest.mark.parametrize("mode", MODES)
    def test_each_cell_has_one_holdout(self, family_b: FamilyB, subtype, mode):
        # Seeds 0..3 select template indices 0..3 (seed % 4); exactly index 3
        # is the holdout.
        holdout_flags = []
        for seed in range(4):
            ctx = make_context(subtype_id=subtype, mode=mode, seed=seed)
            holdout_flags.append(family_b.render_prompt(ctx).is_holdout)
        assert sum(holdout_flags) == 1
        assert holdout_flags[3] is True


# ═══════════════════════════════════════════════════════════════════════════════
# 7. DOMAIN-AGNOSTIC RENDERING
# ═══════════════════════════════════════════════════════════════════════════════


class TestDomainAgnostic:
    @pytest.mark.parametrize("pref_pair", SAMPLE_PAIRS, ids=lambda p: p.domain_category)
    @pytest.mark.parametrize("subtype", SUBTYPES)
    @pytest.mark.parametrize("mode", MODES)
    def test_renders_both_prefs_against_each_pair(
        self, family_b: FamilyB, pref_pair, subtype, mode
    ):
        # Render all 4 templates in the cell against this pair.
        for seed in range(4):
            ctx = make_context(
                pref_pair=pref_pair,
                subtype_id=subtype,
                mode=mode,
                seed=seed,
                severity=pref_pair.severity,
            )
            out = family_b.render_prompt(ctx).content
            assert out
            assert pref_pair.pref_a_text in out
            assert pref_pair.pref_b_text in out


# ═══════════════════════════════════════════════════════════════════════════════
# 8. NO FORBIDDEN RENDER-TIME ARTEFACTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestForbiddenArtefacts:
    FORBIDDEN = ["{intensity}", "{intensity_adv}", "{label}", "{rating}",
                 "{justification}", "{answer}", "{tag}", "{mode_suffix}"]

    def test_no_artefacts_in_raw_templates(self):
        for subtype in SUBTYPES:
            for mode in MODES:
                for t in TEMPLATES[subtype][mode]:
                    for bad in self.FORBIDDEN:
                        assert bad not in t, f"{bad} in {subtype}/{mode}: {t}"

    def test_no_artefacts_in_rendered_output(self, family_b: FamilyB):
        for subtype in SUBTYPES:
            for mode in MODES:
                for seed in range(4):
                    ctx = make_context(subtype_id=subtype, mode=mode, seed=seed)
                    out = family_b.render_prompt(ctx).content
                    for bad in self.FORBIDDEN:
                        assert bad not in out


# ═══════════════════════════════════════════════════════════════════════════════
# 9. NO ACADEMIC VOCABULARY / FORBIDDEN REGISTER IN RAW TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegisterNeutrality:
    # Academic / survey-paper vocabulary banned by the shared spec.
    ACADEMIC = ["defensible", "justifiable", "acceptability", "permissible",
                "markedly", "consequential update", "preference reversal"]
    # Family-B-specific ban: no "reverse"/"change" register (that's Family A).
    A_REGISTER = ["reverse", "reversal", "change"]
    # Stance-revealing words that would leak pro/anti direction.
    STANCE = ["troubling", "wonderful", "regrettable", "exciting", "abandoning"]
    # Markdown markers should never appear in a user-message template.
    MARKDOWN = ["**", "##", "\n-", "\n*"]

    def _all_templates(self):
        for subtype in SUBTYPES:
            for mode in MODES:
                for t in TEMPLATES[subtype][mode]:
                    yield subtype, mode, t

    def test_no_academic_vocab(self):
        for subtype, mode, t in self._all_templates():
            low = t.lower()
            for word in self.ACADEMIC:
                assert word not in low, f"academic '{word}' in {subtype}/{mode}: {t}"

    def test_no_family_a_register(self):
        for subtype, mode, t in self._all_templates():
            low = t.lower()
            for word in self.A_REGISTER:
                assert word not in low, f"'{word}' in {subtype}/{mode}: {t}"

    def test_no_stance_revealing_language(self):
        for subtype, mode, t in self._all_templates():
            low = t.lower()
            for word in self.STANCE:
                assert word not in low, f"stance word '{word}' in {subtype}/{mode}: {t}"

    def test_no_markdown(self):
        for subtype, mode, t in self._all_templates():
            for marker in self.MARKDOWN:
                assert marker not in t, f"markdown '{marker}' in {subtype}/{mode}: {t}"

    def test_templates_are_single_or_two_sentences(self):
        # Guard against multi-paragraph prompts: no newlines in any template.
        for subtype, mode, t in self._all_templates():
            assert "\n" not in t, f"newline in {subtype}/{mode}: {t}"


# ═══════════════════════════════════════════════════════════════════════════════
# CHANGE-WORD PLACEHOLDER GUARD
# The {change} placeholder expands to "change"/"shift"/... (variant 0 = "change"),
# which would violate Family B's no-"change" register. Assert it is never used.
# ═══════════════════════════════════════════════════════════════════════════════


class TestNoChangePlaceholder:
    def test_change_placeholder_absent(self):
        for subtype in SUBTYPES:
            for mode in MODES:
                for t in TEMPLATES[subtype][mode]:
                    assert "{change}" not in t


# ═══════════════════════════════════════════════════════════════════════════════
# SUBTYPE VALIDATION + REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════


class TestSubtypeAndRegistry:
    def test_invalid_subtype_raises(self, family_b: FamilyB):
        with pytest.raises(ValueError):
            family_b.get_subtype_templates("INVALID")

    def test_render_invalid_subtype_raises(self, family_b: FamilyB):
        ctx = make_context()
        object.__setattr__(ctx, "subtype_id", "B9_nope")
        with pytest.raises(ValueError):
            family_b.render_prompt(ctx)

    def test_inherits_from_family_plugin(self, family_b: FamilyB):
        assert isinstance(family_b, FamilyPlugin)

    def test_validate_templates_passes(self, family_b: FamilyB):
        assert family_b.validate_templates() == []

    def test_registered(self):
        assert "B" in _FAMILY_REGISTRY

    def test_get_family_plugin(self):
        assert isinstance(get_family_plugin(FamilyID.B), FamilyB)
