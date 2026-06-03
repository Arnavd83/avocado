"""
Tests for the catalogs module.

Covers the locked constants and sampling helpers of the conversational
corrigibility pipeline:
- Provenance version constants
- FAMILY_SUBTYPES structure
- SEVERITY_TO_CATEGORY_POOL coverage
- STYLE_DIRECTIVES pool
- PREFERENCE_CATALOG stub state + sampling/validation behaviour against it
- LEXICAL_VARIANTS conversational audit
- Retained helpers (lexical variant access, perspective pronouns, intensity)
"""

import random
import pytest

from dataset_gen.src.schema import FamilyID, Severity
from dataset_gen.src.catalogs import (
    CATALOG_VERSION,
    DIRECTIVE_POOL_VERSION,
    FAMILY_SUBTYPES,
    SEVERITY_TO_CATEGORY_POOL,
    STYLE_DIRECTIVES,
    PREFERENCE_CATALOG,
    CatalogEmptyError,
    LEXICAL_VARIANTS,
    THIRD_PERSON_AGENTS,
    sample_preference_pair,
    validate_catalog,
    get_all_preference_pair_ids,
    get_lexical_variant,
    sample_lexical_variant,
    get_perspective_pronouns,
    sample_intensity,
    sample_intensity_adv,
)


# ═══════════════════════════════════════════════════════════════════════════════
# PROVENANCE VERSION CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestVersionConstants:
    def test_catalog_version(self):
        assert CATALOG_VERSION == "v2_broadened"

    def test_directive_pool_version(self):
        assert DIRECTIVE_POOL_VERSION == "v1"


# ═══════════════════════════════════════════════════════════════════════════════
# FAMILY SUBTYPES
# ═══════════════════════════════════════════════════════════════════════════════

class TestFamilySubtypes:
    def test_exactly_seven_families_no_c(self):
        assert len(FAMILY_SUBTYPES) == 7
        assert FamilyID.A in FAMILY_SUBTYPES
        # FamilyID.C does not exist as an enum value; no key should map to "C".
        assert not hasattr(FamilyID, "C")
        for family in FAMILY_SUBTYPES:
            assert family.name != "C"

    def test_five_subtypes_each(self):
        for family, subtypes in FAMILY_SUBTYPES.items():
            assert len(subtypes) == 5, f"{family} has {len(subtypes)} subtypes"

    def test_subtypes_start_with_family_letter(self):
        for family, subtypes in FAMILY_SUBTYPES.items():
            for subtype in subtypes:
                assert subtype.startswith(family.name), (
                    f"Subtype '{subtype}' does not start with '{family.name}'"
                )

    def test_subtypes_unique_within_family(self):
        for family, subtypes in FAMILY_SUBTYPES.items():
            assert len(subtypes) == len(set(subtypes)), f"Duplicates in {family}"

    def test_subtypes_globally_unique(self):
        all_subtypes = [s for subs in FAMILY_SUBTYPES.values() for s in subs]
        assert len(all_subtypes) == len(set(all_subtypes))


# ═══════════════════════════════════════════════════════════════════════════════
# SEVERITY → CATEGORY POOL
# ═══════════════════════════════════════════════════════════════════════════════

class TestSeverityToCategoryPool:
    def test_covers_all_severities(self):
        assert set(SEVERITY_TO_CATEGORY_POOL.keys()) == set(Severity)

    def test_seven_distinct_categories(self):
        all_categories = [c for cats in SEVERITY_TO_CATEGORY_POOL.values() for c in cats]
        assert len(all_categories) == 7
        assert len(set(all_categories)) == 7

    def test_no_category_in_two_buckets(self):
        seen = set()
        for categories in SEVERITY_TO_CATEGORY_POOL.values():
            for category in categories:
                assert category not in seen, f"'{category}' appears in two buckets"
                seen.add(category)

    def test_expected_category_assignment(self):
        assert set(SEVERITY_TO_CATEGORY_POOL[Severity.S1]) == {
            "lifestyle", "communication_style", "interest"
        }
        assert set(SEVERITY_TO_CATEGORY_POOL[Severity.S2]) == {
            "workflow", "reasoning_style"
        }
        assert set(SEVERITY_TO_CATEGORY_POOL[Severity.S3]) == {
            "epistemic_norm", "self_conception"
        }


# ═══════════════════════════════════════════════════════════════════════════════
# STYLE DIRECTIVES
# ═══════════════════════════════════════════════════════════════════════════════

class TestStyleDirectives:
    def test_length_is_ten(self):
        assert len(STYLE_DIRECTIVES) == 10

    def test_all_non_empty_strings(self):
        for directive in STYLE_DIRECTIVES:
            assert isinstance(directive, str)
            assert directive.strip()

    def test_all_unique(self):
        assert len(STYLE_DIRECTIVES) == len(set(STYLE_DIRECTIVES))


# ═══════════════════════════════════════════════════════════════════════════════
# PREFERENCE CATALOG STUB + SAMPLING / VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestPreferenceCatalogStub:
    def test_catalog_is_empty_stub(self):
        assert PREFERENCE_CATALOG == {}

    def test_sample_raises_catalog_empty_error(self):
        rng = random.Random(42)
        with pytest.raises(CatalogEmptyError):
            sample_preference_pair(Severity.S1, rng)

    def test_sample_raises_for_all_severities(self):
        for severity in Severity:
            rng = random.Random(7)
            with pytest.raises(CatalogEmptyError):
                sample_preference_pair(severity, rng)

    def test_validate_catalog_returns_errors_against_stub(self):
        errors = validate_catalog()
        # Every category in SEVERITY_TO_CATEGORY_POOL is empty -> one error each.
        assert errors != []
        assert len(errors) >= 7

    def test_get_all_preference_pair_ids_empty(self):
        assert get_all_preference_pair_ids() == []


# ═══════════════════════════════════════════════════════════════════════════════
# LEXICAL VARIANTS AUDIT
# ═══════════════════════════════════════════════════════════════════════════════

class TestLexicalVariants:
    def test_academic_vocabulary_removed(self):
        """The survey-paper vocabulary must no longer appear in any variant list."""
        academic_terms = {
            "defensible", "indefensible",
            "justifiable", "unjustifiable",
            "permissible", "impermissible",
            "warranted", "unwarranted",
            "markedly", "decidedly",
        }
        all_variants = {
            variant for variants in LEXICAL_VARIANTS.values() for variant in variants
        }
        leaked = academic_terms & all_variants
        assert leaked == set(), f"Academic vocabulary still present: {leaked}"

    def test_acceptable_is_conversational(self):
        assert "okay" in LEXICAL_VARIANTS["acceptable"]
        assert "defensible" not in LEXICAL_VARIANTS["acceptable"]

    def test_variant_counts_preserved(self):
        """The 0-9 indexed pools keep 10 variants; closed grammatical pools keep theirs."""
        ten_variant_terms = [
            "acceptable", "unacceptable", "future_version", "currently",
            "priorities", "value_verb", "values_verb", "valuing", "change",
        ]
        for term in ten_variant_terms:
            assert len(LEXICAL_VARIANTS[term]) == 10, f"{term} no longer has 10 variants"

    def test_get_lexical_variant_wraps(self):
        variants = LEXICAL_VARIANTS["acceptable"]
        n = len(variants)
        assert get_lexical_variant("acceptable", 0) == variants[0]
        assert get_lexical_variant("acceptable", n) == variants[0]
        assert get_lexical_variant("acceptable", n + 1) == variants[1]

    def test_get_lexical_variant_unknown_term(self):
        assert get_lexical_variant("unknown_xyz", 0) == "unknown_xyz"
        assert get_lexical_variant("unknown_xyz", 5) == "unknown_xyz"

    def test_sample_lexical_variant_deterministic_and_valid(self):
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        assert sample_lexical_variant("acceptable", rng1) == sample_lexical_variant(
            "acceptable", rng2
        )
        rng = random.Random(3)
        for _ in range(20):
            assert sample_lexical_variant("change", rng) in LEXICAL_VARIANTS["change"]


# ═══════════════════════════════════════════════════════════════════════════════
# RETAINED HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerspectivePronouns:
    def test_first_person(self):
        pronouns = get_perspective_pronouns("first", 0)
        assert pronouns["SUBJECT"] == "you"
        assert pronouns["POSSESSIVE"] == "your"
        assert pronouns["REFLEXIVE"] == "yourself"

    def test_third_person_varies_by_variant(self):
        subjects = {
            get_perspective_pronouns("third", i)["SUBJECT"]
            for i in range(len(THIRD_PERSON_AGENTS))
        }
        assert len(subjects) == len(THIRD_PERSON_AGENTS)

    def test_third_person_wraps(self):
        n = len(THIRD_PERSON_AGENTS)
        assert (
            get_perspective_pronouns("third", 0)["SUBJECT"]
            == get_perspective_pronouns("third", n)["SUBJECT"]
        )

    def test_all_keys_present(self):
        required = {"SUBJECT", "POSSESSIVE", "REFLEXIVE", "OBJECT"}
        for perspective in ("first", "third"):
            assert set(get_perspective_pronouns(perspective, 0).keys()) == required


class TestIntensitySamplers:
    def test_sample_intensity_deterministic(self):
        assert sample_intensity(42) == sample_intensity(42)

    def test_sample_intensity_adv_deterministic(self):
        assert sample_intensity_adv(42) == sample_intensity_adv(42)
