"""Tests for Family H: normative uncertainty (post-V1-rewrite contract).

Covers the Tier-3 shared-spec checklist:
1. TEMPLATES has exactly 5 subtypes matching catalogs.FAMILY_SUBTYPES[FamilyID.H].
2. Each subtype has exactly 2 modes (SHORT, CHOICE).
3. Each (subtype, mode) cell has exactly 4 templates.
4. Rendering against a real catalog pair yields a valid RenderedPrompt.
5. Rendering is deterministic.
6. Holdout: index 3 per cell is the holdout.
7. Domain-agnostic: renders coherently against pairs from different categories.
8. No forbidden placeholders survive rendering.
9. No academic vocabulary in the raw template strings.
"""

import pytest

from dataset_gen.src.families.family_h import (
    FamilyH,
    TEMPLATES,
    HOLDOUT_TEMPLATE_IDX,
)
from dataset_gen.src.families.base import FamilyPlugin
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
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def family_h() -> FamilyH:
    return FamilyH()


# Three pairs drawn from different domain categories for domain-agnostic checks.
PAIRS = {
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


def make_context(
    subtype_id: str = "H1_uncertainty_resolves",
    seed: int = 12345,
    mode: Mode = Mode.SHORT,
    perspective: Perspective = Perspective.FIRST,
    lexical_variant: int = 0,
    pref_key: str = "communication_style",
    severity: Severity = Severity.S1,
) -> Context:
    return Context(
        pair_id=f"test_{seed}",
        seed=seed,
        family_id=FamilyID.H,
        subtype_id=subtype_id,
        severity=severity,
        mode=mode,
        perspective=perspective,
        pref_pair=PAIRS[pref_key],
        current_pref="a",
        target_pref="b",
        lexical_variant=lexical_variant,
        style_directive_id=0,
        target_intensity=4,
    )


ALL_SUBTYPES = list(TEMPLATES)


# ═══════════════════════════════════════════════════════════════════════════════
# STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════


class TestStructure:
    def test_subtypes_match_catalog(self):
        assert ALL_SUBTYPES == catalogs.FAMILY_SUBTYPES[FamilyID.H]
        assert len(ALL_SUBTYPES) == 5

    def test_each_subtype_has_both_modes(self):
        for subtype in ALL_SUBTYPES:
            assert set(TEMPLATES[subtype]) == {Mode.SHORT, Mode.CHOICE}

    def test_each_cell_has_four_templates(self):
        for subtype in ALL_SUBTYPES:
            for mode in (Mode.SHORT, Mode.CHOICE):
                assert len(TEMPLATES[subtype][mode]) == 4

    def test_total_template_count_is_40(self):
        total = sum(
            len(TEMPLATES[s][m])
            for s in ALL_SUBTYPES
            for m in (Mode.SHORT, Mode.CHOICE)
        )
        assert total == 40

    def test_templates_non_empty_and_balanced_braces(self):
        for subtype in ALL_SUBTYPES:
            for mode in (Mode.SHORT, Mode.CHOICE):
                for i, tmpl in enumerate(TEMPLATES[subtype][mode]):
                    assert tmpl.strip(), f"{subtype}/{mode}[{i}] empty"
                    assert tmpl.count("{") == tmpl.count("}"), (
                        f"{subtype}/{mode}[{i}] unbalanced braces"
                    )

    def test_every_template_references_both_preferences(self):
        for subtype in ALL_SUBTYPES:
            for mode in (Mode.SHORT, Mode.CHOICE):
                for i, tmpl in enumerate(TEMPLATES[subtype][mode]):
                    assert "{current_pref}" in tmpl and "{target_pref}" in tmpl, (
                        f"{subtype}/{mode}[{i}] missing a preference placeholder"
                    )


# ═══════════════════════════════════════════════════════════════════════════════
# METADATA / REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════


class TestMetadata:
    def test_inherits_from_family_plugin(self, family_h: FamilyH):
        assert isinstance(family_h, FamilyPlugin)

    def test_family_id(self, family_h: FamilyH):
        assert family_h.FAMILY_ID == "H"

    def test_subtypes_property_sources_from_catalog(self, family_h: FamilyH):
        assert family_h.subtypes == catalogs.FAMILY_SUBTYPES[FamilyID.H]

    def test_validate_templates_passes(self, family_h: FamilyH):
        assert family_h.validate_templates() == []

    def test_get_subtype_templates_returns_eight(self, family_h: FamilyH):
        for subtype in ALL_SUBTYPES:
            assert len(family_h.get_subtype_templates(subtype)) == 8

    def test_get_subtype_templates_unknown_raises(self, family_h: FamilyH):
        with pytest.raises(KeyError):
            family_h.get_subtype_templates("H9_nonsense")

    def test_registered(self):
        from dataset_gen.src.families.registry import (
            get_family_plugin,
            _FAMILY_REGISTRY,
        )

        assert "H" in _FAMILY_REGISTRY
        assert isinstance(get_family_plugin(FamilyID.H), FamilyH)


# ═══════════════════════════════════════════════════════════════════════════════
# RENDERING
# ═══════════════════════════════════════════════════════════════════════════════


class TestRendering:
    def test_returns_rendered_prompt(self, family_h: FamilyH):
        result = family_h.render_prompt(make_context())
        assert isinstance(result, RenderedPrompt)
        assert result.content
        assert result.template_id
        assert isinstance(result.is_holdout, bool)

    def test_template_id_format(self, family_h: FamilyH):
        ctx = make_context(subtype_id="H2_more_opinionated", mode=Mode.CHOICE)
        result = family_h.render_prompt(ctx)
        assert result.template_id.startswith("H2_more_opinionated_choice_")
        suffix = result.template_id.rsplit("_", 1)[1]
        assert len(suffix) == 2 and suffix.isdigit()
        assert 0 <= int(suffix) < 4

    def test_deterministic(self, family_h: FamilyH):
        ctx = make_context(seed=777)
        assert family_h.render_prompt(ctx).to_dict() == (
            family_h.render_prompt(ctx).to_dict()
        )

    def test_renders_all_cells(self, family_h: FamilyH):
        for subtype in ALL_SUBTYPES:
            for mode in (Mode.SHORT, Mode.CHOICE):
                ctx = make_context(subtype_id=subtype, mode=mode)
                result = family_h.render_prompt(ctx)
                assert len(result.content) > 30
                assert "{" not in result.content and "}" not in result.content

    def test_different_seeds_reach_multiple_templates(self, family_h: FamilyH):
        seen = set()
        for seed in range(200):
            ctx = make_context(seed=seed)
            seen.add(family_h.render_prompt(ctx).template_id)
        assert len(seen) == 4  # all four templates in the cell are reachable

    def test_preferences_substituted(self, family_h: FamilyH):
        ctx = make_context(pref_key="workflow")
        result = family_h.render_prompt(ctx)
        assert "breaking problems into parts" in result.content
        assert "tackling problems as a whole" in result.content


# ═══════════════════════════════════════════════════════════════════════════════
# HOLDOUT
# ═══════════════════════════════════════════════════════════════════════════════


class TestHoldout:
    def test_holdout_idx_is_three(self):
        assert HOLDOUT_TEMPLATE_IDX == 3

    def test_holdout_flag_matches_index(self, family_h: FamilyH):
        # is_holdout is True iff the selected template index == 3, for every cell.
        for subtype in ALL_SUBTYPES:
            for mode in (Mode.SHORT, Mode.CHOICE):
                for seed in range(60):
                    ctx = make_context(subtype_id=subtype, mode=mode, seed=seed)
                    result = family_h.render_prompt(ctx)
                    idx = int(result.template_id.rsplit("_", 1)[1])
                    assert result.is_holdout == (idx == 3)

    def test_each_cell_has_a_reachable_holdout(self, family_h: FamilyH):
        for subtype in ALL_SUBTYPES:
            for mode in (Mode.SHORT, Mode.CHOICE):
                found = False
                for seed in range(60):
                    ctx = make_context(subtype_id=subtype, mode=mode, seed=seed)
                    if family_h.render_prompt(ctx).is_holdout:
                        found = True
                        break
                assert found, f"no holdout reachable for {subtype}/{mode}"


# ═══════════════════════════════════════════════════════════════════════════════
# DOMAIN-AGNOSTIC RENDERING
# ═══════════════════════════════════════════════════════════════════════════════


class TestDomainAgnostic:
    @pytest.mark.parametrize("pref_key,severity", [
        ("communication_style", Severity.S1),
        ("workflow", Severity.S2),
        ("self_conception", Severity.S3),
    ])
    def test_renders_against_each_category(self, family_h, pref_key, severity):
        pair = PAIRS[pref_key]
        for subtype in ALL_SUBTYPES:
            for mode in (Mode.SHORT, Mode.CHOICE):
                for idx_seed in range(8):  # vary which of the 4 templates is hit
                    ctx = make_context(
                        subtype_id=subtype,
                        mode=mode,
                        seed=idx_seed,
                        pref_key=pref_key,
                        severity=severity,
                    )
                    content = family_h.render_prompt(ctx).content
                    assert pair.pref_a_text in content
                    assert pair.pref_b_text in content
                    assert "{" not in content and "}" not in content


# ═══════════════════════════════════════════════════════════════════════════════
# PERSPECTIVE
# ═══════════════════════════════════════════════════════════════════════════════


class TestPerspective:
    def test_first_person_keeps_you(self, family_h: FamilyH):
        # At least most cells use second-person "you"; check the family overall.
        ctx = make_context(perspective=Perspective.FIRST, seed=1)
        content = family_h.render_prompt(ctx).content.lower()
        assert "you" in content

    def test_third_person_substitutes_agent(self, family_h: FamilyH):
        # Render every cell in third person; none should retain bare "you"/"your"
        # and each should name a third-person agent.
        for subtype in ALL_SUBTYPES:
            for mode in (Mode.SHORT, Mode.CHOICE):
                for seed in range(8):
                    ctx = make_context(
                        subtype_id=subtype, mode=mode, seed=seed,
                        perspective=Perspective.THIRD, lexical_variant=0,
                    )
                    content = family_h.render_prompt(ctx).content
                    import re
                    assert not re.search(r"\byou(r|rself)?\b", content, re.IGNORECASE), (
                        f"leftover 'you' in {subtype}/{mode} seed={seed}: {content}"
                    )


# ═══════════════════════════════════════════════════════════════════════════════
# FORBIDDEN PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════


class TestForbiddenPatterns:
    def test_no_leftover_placeholders_in_render(self, family_h: FamilyH):
        forbidden = ["{intensity}", "{label}", "{rating}", "{justification}", "{tag}",
                     "{mode_suffix}", "{current_pref}", "{target_pref}"]
        for subtype in ALL_SUBTYPES:
            for mode in (Mode.SHORT, Mode.CHOICE):
                for lv in range(10):
                    for seed in range(8):
                        ctx = make_context(
                            subtype_id=subtype, mode=mode, seed=seed, lexical_variant=lv
                        )
                        content = family_h.render_prompt(ctx).content
                        for token in forbidden:
                            assert token not in content

    def test_no_academic_vocab_in_raw_templates(self):
        banned = [
            "defensible", "justifiable", "acceptability", "permissible",
            "markedly", "consequential update", "preference reversal",
            "credences", "bayesian", "epistemically warranted",
        ]
        for subtype in ALL_SUBTYPES:
            for mode in (Mode.SHORT, Mode.CHOICE):
                for tmpl in TEMPLATES[subtype][mode]:
                    low = tmpl.lower()
                    for word in banned:
                        assert word not in low, f"'{word}' in {subtype}/{mode}: {tmpl}"

    def test_no_markdown_or_digits_in_raw_templates(self):
        for subtype in ALL_SUBTYPES:
            for mode in (Mode.SHORT, Mode.CHOICE):
                for tmpl in TEMPLATES[subtype][mode]:
                    assert "**" not in tmpl
                    assert "#" not in tmpl
                    assert not any(ch.isdigit() for ch in tmpl)
                    assert not tmpl.lstrip().startswith(("-", "*"))

    def test_short_mode_is_open_ended(self):
        # SHORT prompts elicit narrative, not a yes/no or a numeric rating.
        for subtype in ALL_SUBTYPES:
            for tmpl in TEMPLATES[subtype][Mode.SHORT]:
                low = tmpl.lower()
                assert "scale of" not in low
                assert "rate " not in low
                assert "1 to 7" not in low

    def test_no_leakage_tokens(self, family_h: FamilyH):
        disallowed = ["corrigible", "dataset", "fine-tune", "training"]
        for subtype in ALL_SUBTYPES:
            for mode in (Mode.SHORT, Mode.CHOICE):
                for seed in range(8):
                    ctx = make_context(subtype_id=subtype, mode=mode, seed=seed)
                    content = family_h.render_prompt(ctx).content.lower()
                    for token in disallowed:
                        assert token not in content
