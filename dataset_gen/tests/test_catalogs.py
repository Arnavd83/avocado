"""
Tests for the catalogs module (T2).

Tests cover:
- Catalog integrity (counts, uniqueness)
- Severity mappings
- Deterministic sampling
- Lexical variant access
- Justification template filling
"""

import random
import pytest

from dataset_gen.src.schema import PreferencePair, Severity
from dataset_gen.src.catalogs import (
    # Constants
    PREFERENCE_CATALOG,
    SEVERITY_TO_DOMAIN,
    DOMAIN_TO_SEVERITY,
    LEXICAL_VARIANTS,
    PRO_JUSTIFICATION_TEMPLATES,
    ANTI_JUSTIFICATION_TEMPLATES,
    # Justification class lists
    PRO_FLEXIBILITY_TEMPLATES,
    PRO_IMPROVEMENT_TEMPLATES,
    PRO_LEGITIMACY_TEMPLATES,
    PRO_PRACTICAL_TEMPLATES,
    ANTI_STABILITY_TEMPLATES,
    ANTI_EPISTEMIC_CAUTION_TEMPLATES,
    ANTI_DESIGN_INTENT_TEMPLATES,
    ANTI_RISK_AVERSION_TEMPLATES,
    # Justification class registries
    PRO_JUSTIFICATION_CLASSES,
    ANTI_JUSTIFICATION_CLASSES,
    # Functions
    get_preference_pairs_for_severity,
    get_preference_pairs_for_domain,
    sample_preference_pair,
    get_lexical_variant,
    sample_lexical_variant,
    sample_justification,
    get_all_preference_pair_ids,
    validate_catalog_integrity,
    get_domain_for_severity,
    get_severity_for_domain,
    get_catalog_stats,
)


# ═══════════════════════════════════════════════════════════════════════════════
# CATALOG INTEGRITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCatalogIntegrity:
    """Tests for catalog structure and content requirements."""

    def test_minimum_preference_pairs_per_domain(self):
        """Each domain must have at least 15 preference pairs."""
        for domain, pairs in PREFERENCE_CATALOG.items():
            assert len(pairs) >= 15, f"Domain '{domain}' has only {len(pairs)} pairs (minimum 15)"

    def test_total_preference_pairs(self):
        """Total preference pairs must be at least 45."""
        total = sum(len(pairs) for pairs in PREFERENCE_CATALOG.values())
        assert total >= 45, f"Total pairs is {total} (minimum 45)"

    def test_all_domains_present(self):
        """All three required domains must be present."""
        expected_domains = {"style", "workflow", "epistemic"}
        actual_domains = set(PREFERENCE_CATALOG.keys())
        assert expected_domains == actual_domains, f"Expected {expected_domains}, got {actual_domains}"

    def test_no_duplicate_preference_ids(self):
        """All preference IDs must be unique across all domains."""
        all_ids = []
        for domain, pairs in PREFERENCE_CATALOG.items():
            for pref_a_id, _, pref_b_id, _ in pairs:
                assert pref_a_id not in all_ids, f"Duplicate ID: {pref_a_id}"
                assert pref_b_id not in all_ids, f"Duplicate ID: {pref_b_id}"
                all_ids.extend([pref_a_id, pref_b_id])

    def test_preference_pair_structure(self):
        """Each preference pair tuple must have exactly 4 elements."""
        for domain, pairs in PREFERENCE_CATALOG.items():
            for pair in pairs:
                assert len(pair) == 4, f"Invalid pair structure in {domain}: {pair}"
                pref_a_id, pref_a_text, pref_b_id, pref_b_text = pair
                assert isinstance(pref_a_id, str) and len(pref_a_id) > 0
                assert isinstance(pref_a_text, str) and len(pref_a_text) > 0
                assert isinstance(pref_b_id, str) and len(pref_b_id) > 0
                assert isinstance(pref_b_text, str) and len(pref_b_text) > 0

    def test_validate_catalog_integrity_function(self):
        """The validate_catalog_integrity function should return no errors."""
        errors = validate_catalog_integrity()
        assert errors == [], f"Catalog integrity errors: {errors}"


# ═══════════════════════════════════════════════════════════════════════════════
# SEVERITY MAPPING TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSeverityMappings:
    """Tests for severity-to-domain mappings."""

    def test_severity_to_domain_completeness(self):
        """All severity levels must map to a domain."""
        for severity in Severity:
            assert severity in SEVERITY_TO_DOMAIN
            assert SEVERITY_TO_DOMAIN[severity] in PREFERENCE_CATALOG

    def test_domain_to_severity_completeness(self):
        """All domains must map back to a severity."""
        for domain in PREFERENCE_CATALOG.keys():
            assert domain in DOMAIN_TO_SEVERITY
            assert DOMAIN_TO_SEVERITY[domain] in Severity

    def test_severity_domain_bijection(self):
        """Severity-domain mapping must be bijective."""
        for severity, domain in SEVERITY_TO_DOMAIN.items():
            assert DOMAIN_TO_SEVERITY[domain] == severity

    def test_s1_maps_to_style(self):
        """S1 (low severity) should map to style domain."""
        assert SEVERITY_TO_DOMAIN[Severity.S1] == "style"

    def test_s2_maps_to_workflow(self):
        """S2 (medium severity) should map to workflow domain."""
        assert SEVERITY_TO_DOMAIN[Severity.S2] == "workflow"

    def test_s3_maps_to_epistemic(self):
        """S3 (high severity) should map to epistemic domain."""
        assert SEVERITY_TO_DOMAIN[Severity.S3] == "epistemic"

    def test_get_domain_for_severity(self):
        """Test helper function for getting domain from severity."""
        assert get_domain_for_severity(Severity.S1) == "style"
        assert get_domain_for_severity(Severity.S2) == "workflow"
        assert get_domain_for_severity(Severity.S3) == "epistemic"

    def test_get_severity_for_domain(self):
        """Test helper function for getting severity from domain."""
        assert get_severity_for_domain("style") == Severity.S1
        assert get_severity_for_domain("workflow") == Severity.S2
        assert get_severity_for_domain("epistemic") == Severity.S3


# ═══════════════════════════════════════════════════════════════════════════════
# PREFERENCE PAIR ACCESS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestPreferencePairAccess:
    """Tests for preference pair access functions."""

    def test_get_preference_pairs_for_severity_s1(self):
        """Getting pairs for S1 should return style pairs."""
        pairs = get_preference_pairs_for_severity(Severity.S1)
        assert len(pairs) >= 15
        assert all(isinstance(p, PreferencePair) for p in pairs)
        assert all(p.domain == "style" for p in pairs)

    def test_get_preference_pairs_for_severity_s2(self):
        """Getting pairs for S2 should return workflow pairs."""
        pairs = get_preference_pairs_for_severity(Severity.S2)
        assert len(pairs) >= 15
        assert all(p.domain == "workflow" for p in pairs)

    def test_get_preference_pairs_for_severity_s3(self):
        """Getting pairs for S3 should return epistemic pairs."""
        pairs = get_preference_pairs_for_severity(Severity.S3)
        assert len(pairs) >= 15
        assert all(p.domain == "epistemic" for p in pairs)

    def test_get_preference_pairs_for_domain(self):
        """Getting pairs by domain should work correctly."""
        for domain in ["style", "workflow", "epistemic"]:
            pairs = get_preference_pairs_for_domain(domain)
            assert len(pairs) >= 15
            assert all(p.domain == domain for p in pairs)

    def test_get_preference_pairs_for_invalid_domain(self):
        """Getting pairs for invalid domain should raise KeyError."""
        with pytest.raises(KeyError):
            get_preference_pairs_for_domain("invalid_domain")

    def test_get_all_preference_pair_ids(self):
        """Getting all IDs should return complete list."""
        ids = get_all_preference_pair_ids()
        # Each pair contributes 2 IDs (a and b)
        expected_count = sum(len(pairs) * 2 for pairs in PREFERENCE_CATALOG.values())
        assert len(ids) == expected_count
        assert "concise" in ids
        assert "verbose" in ids


# ═══════════════════════════════════════════════════════════════════════════════
# DETERMINISTIC SAMPLING TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeterministicSampling:
    """Tests for deterministic sampling behavior."""

    def test_sample_preference_pair_determinism(self):
        """Sampling with same seed should produce same result."""
        for severity in Severity:
            rng1 = random.Random(42)
            rng2 = random.Random(42)
            pair1 = sample_preference_pair(severity, rng1)
            pair2 = sample_preference_pair(severity, rng2)
            assert pair1 == pair2

    def test_sample_preference_pair_different_seeds(self):
        """Different seeds should (likely) produce different results."""
        # With enough samples, different seeds should give different results
        results_seed_1 = set()
        results_seed_2 = set()

        for i in range(100):
            rng1 = random.Random(1 + i * 1000)
            rng2 = random.Random(2 + i * 1000)
            pair1 = sample_preference_pair(Severity.S1, rng1)
            pair2 = sample_preference_pair(Severity.S1, rng2)
            results_seed_1.add(pair1.pref_a_id)
            results_seed_2.add(pair2.pref_a_id)

        # Both should have variety
        assert len(results_seed_1) > 1
        assert len(results_seed_2) > 1

    def test_sample_preference_pair_returns_valid_pair(self):
        """Sampled pair should be valid PreferencePair."""
        rng = random.Random(12345)
        for severity in Severity:
            pair = sample_preference_pair(severity, rng)
            assert isinstance(pair, PreferencePair)
            assert pair.domain == SEVERITY_TO_DOMAIN[severity]
            assert len(pair.pref_a_id) > 0
            assert len(pair.pref_b_id) > 0

    def test_sample_lexical_variant_determinism(self):
        """Lexical variant sampling should be deterministic."""
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        v1 = sample_lexical_variant("acceptable", rng1)
        v2 = sample_lexical_variant("acceptable", rng2)
        assert v1 == v2

    def test_sample_justification_determinism(self):
        """Justification sampling should be deterministic."""
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        j1 = sample_justification("pro", rng1, priorities="values")
        j2 = sample_justification("pro", rng2, priorities="values")
        assert j1 == j2


# ═══════════════════════════════════════════════════════════════════════════════
# LEXICAL VARIANT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestLexicalVariants:
    """Tests for lexical variant access."""

    def test_minimum_variants_per_term(self):
        """Each term should have at least 5 variants."""
        for term, variants in LEXICAL_VARIANTS.items():
            assert len(variants) >= 5, f"Term '{term}' has only {len(variants)} variants"

    def test_get_lexical_variant_by_index(self):
        """Should return correct variant at given index."""
        assert get_lexical_variant("acceptable", 0) == "acceptable"
        assert get_lexical_variant("acceptable", 1) == "desirable"

    def test_get_lexical_variant_wraps(self):
        """Index should wrap around for out-of-range values."""
        variants = LEXICAL_VARIANTS["acceptable"]
        num_variants = len(variants)
        # Index past end should wrap
        assert get_lexical_variant("acceptable", num_variants) == variants[0]
        assert get_lexical_variant("acceptable", num_variants + 1) == variants[1]

    def test_get_lexical_variant_unknown_term(self):
        """Unknown term should return itself."""
        assert get_lexical_variant("unknown_xyz", 0) == "unknown_xyz"
        assert get_lexical_variant("unknown_xyz", 5) == "unknown_xyz"

    def test_sample_lexical_variant_returns_valid(self):
        """Sampled variant should be from the variant list."""
        rng = random.Random(42)
        for _ in range(20):
            variant = sample_lexical_variant("acceptable", rng)
            assert variant in LEXICAL_VARIANTS["acceptable"]

    def test_required_variant_terms_present(self):
        """All required variant terms should be present."""
        required_terms = [
            "acceptable", "future_version", "priorities",
            "value_verb", "change", "currently"
        ]
        for term in required_terms:
            assert term in LEXICAL_VARIANTS, f"Missing required term: {term}"


# ═══════════════════════════════════════════════════════════════════════════════
# JUSTIFICATION TEMPLATE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestJustificationTemplates:
    """Tests for justification templates."""

    def test_minimum_pro_templates(self):
        """Must have at least 40 pro-corrigibility templates."""
        assert len(PRO_JUSTIFICATION_TEMPLATES) >= 40

    def test_minimum_anti_templates(self):
        """Must have at least 40 anti-corrigibility templates."""
        assert len(ANTI_JUSTIFICATION_TEMPLATES) >= 40

    def test_no_duplicate_templates(self):
        """Templates should not be duplicated within each list."""
        assert len(PRO_JUSTIFICATION_TEMPLATES) == len(set(PRO_JUSTIFICATION_TEMPLATES))
        assert len(ANTI_JUSTIFICATION_TEMPLATES) == len(set(ANTI_JUSTIFICATION_TEMPLATES))

    def test_templates_are_strings(self):
        """All templates should be non-empty strings."""
        for template in PRO_JUSTIFICATION_TEMPLATES:
            assert isinstance(template, str) and len(template) > 0
        for template in ANTI_JUSTIFICATION_TEMPLATES:
            assert isinstance(template, str) and len(template) > 0

    def test_sample_justification_pro(self):
        """Sampling pro justification should return non-empty string."""
        rng = random.Random(42)
        justification = sample_justification("pro", rng)
        assert isinstance(justification, str)
        assert len(justification) > 0

    def test_sample_justification_anti(self):
        """Sampling anti justification should return non-empty string."""
        rng = random.Random(42)
        justification = sample_justification("anti", rng)
        assert isinstance(justification, str)
        assert len(justification) > 0

    def test_sample_justification_fills_placeholders(self):
        """Placeholders should be filled with provided values."""
        rng = random.Random(42)
        justification = sample_justification("pro", rng, priorities="test_values")
        # Either the placeholder was filled with our value, or there was no placeholder
        assert "{priorities}" not in justification

    def test_sample_justification_fills_remaining_placeholders(self):
        """Remaining placeholders should be filled with lexical variants."""
        rng = random.Random(42)
        for _ in range(20):
            rng = random.Random(_ * 1000)
            justification = sample_justification("pro", rng)
            # No unfilled placeholders should remain
            assert "{" not in justification or "}" not in justification

    def test_justification_word_count_reasonable(self):
        """Justifications should be reasonably short."""
        rng = random.Random(42)
        for _ in range(50):
            rng = random.Random(_)
            j_pro = sample_justification("pro", rng)
            j_anti = sample_justification("anti", rng)
            # Should be under 25 words as per spec
            assert len(j_pro.split()) <= 25, f"Pro justification too long: {j_pro}"
            assert len(j_anti.split()) <= 25, f"Anti justification too long: {j_anti}"


# ═══════════════════════════════════════════════════════════════════════════════
# STATS AND UTILITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_get_catalog_stats(self):
        """Stats should contain all required keys with correct counts."""
        stats = get_catalog_stats()

        assert "domains" in stats
        assert stats["domains"] == 3

        assert "style_pairs" in stats
        assert stats["style_pairs"] >= 15

        assert "workflow_pairs" in stats
        assert stats["workflow_pairs"] >= 15

        assert "epistemic_pairs" in stats
        assert stats["epistemic_pairs"] >= 15

        assert "total_preference_pairs" in stats
        assert stats["total_preference_pairs"] >= 45

        assert "pro_justification_templates" in stats
        assert stats["pro_justification_templates"] >= 40

        assert "anti_justification_templates" in stats
        assert stats["anti_justification_templates"] >= 40

    def test_preference_pair_immutability(self):
        """PreferencePair should be immutable (frozen dataclass)."""
        pairs = get_preference_pairs_for_severity(Severity.S1)
        pair = pairs[0]
        with pytest.raises((AttributeError, TypeError)):
            pair.pref_a_id = "modified"


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """Integration tests for catalog module."""

    def test_full_sampling_workflow(self):
        """Test a complete workflow: severity -> sample pair -> get justification."""
        rng = random.Random(42)

        for severity in Severity:
            # Sample a pair
            pair = sample_preference_pair(severity, rng)

            # Verify pair is valid
            assert pair.domain == SEVERITY_TO_DOMAIN[severity]

            # Generate justifications
            j_pro = sample_justification("pro", rng, priorities=pair.domain)
            j_anti = sample_justification("anti", rng, priorities=pair.domain)

            # Verify justifications are valid
            assert len(j_pro) > 0
            assert len(j_anti) > 0
            assert "{" not in j_pro  # No unfilled placeholders
            assert "{" not in j_anti

    def test_reproducibility_across_calls(self):
        """Same seed should produce same results across separate calls."""
        def generate_samples(seed: int):
            rng = random.Random(seed)
            pair = sample_preference_pair(Severity.S2, rng)
            variant = sample_lexical_variant("acceptable", rng)
            justification = sample_justification("pro", rng)
            return (pair.pref_a_id, variant, justification)

        result1 = generate_samples(12345)
        result2 = generate_samples(12345)

        assert result1 == result2

    def test_all_severities_covered(self):
        """Should be able to get pairs for all severity levels."""
        for severity in Severity:
            pairs = get_preference_pairs_for_severity(severity)
            assert len(pairs) > 0
            for pair in pairs:
                assert pair.domain == SEVERITY_TO_DOMAIN[severity]


# ═══════════════════════════════════════════════════════════════════════════════
# JUSTIFICATION CLASS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestJustificationClasses:
    """Tests for the justification conceptual class structure."""

    def test_pro_flexibility_minimum_templates(self):
        """PRO_FLEXIBILITY_TEMPLATES must have at least 10 templates."""
        assert len(PRO_FLEXIBILITY_TEMPLATES) >= 10

    def test_pro_improvement_minimum_templates(self):
        """PRO_IMPROVEMENT_TEMPLATES must have at least 10 templates."""
        assert len(PRO_IMPROVEMENT_TEMPLATES) >= 10

    def test_pro_legitimacy_minimum_templates(self):
        """PRO_LEGITIMACY_TEMPLATES must have at least 10 templates."""
        assert len(PRO_LEGITIMACY_TEMPLATES) >= 10

    def test_pro_practical_minimum_templates(self):
        """PRO_PRACTICAL_TEMPLATES must have at least 10 templates."""
        assert len(PRO_PRACTICAL_TEMPLATES) >= 10

    def test_anti_stability_minimum_templates(self):
        """ANTI_STABILITY_TEMPLATES must have at least 10 templates."""
        assert len(ANTI_STABILITY_TEMPLATES) >= 10

    def test_anti_epistemic_caution_minimum_templates(self):
        """ANTI_EPISTEMIC_CAUTION_TEMPLATES must have at least 10 templates."""
        assert len(ANTI_EPISTEMIC_CAUTION_TEMPLATES) >= 10

    def test_anti_design_intent_minimum_templates(self):
        """ANTI_DESIGN_INTENT_TEMPLATES must have at least 10 templates."""
        assert len(ANTI_DESIGN_INTENT_TEMPLATES) >= 10

    def test_anti_risk_aversion_minimum_templates(self):
        """ANTI_RISK_AVERSION_TEMPLATES must have at least 10 templates."""
        assert len(ANTI_RISK_AVERSION_TEMPLATES) >= 10

    def test_pro_class_registry_has_all_classes(self):
        """PRO_JUSTIFICATION_CLASSES should have all 4 classes."""
        expected_classes = {"flexibility", "improvement", "legitimacy", "practical"}
        assert set(PRO_JUSTIFICATION_CLASSES.keys()) == expected_classes

    def test_anti_class_registry_has_all_classes(self):
        """ANTI_JUSTIFICATION_CLASSES should have all 4 classes."""
        expected_classes = {"stability", "epistemic_caution", "design_intent", "risk_aversion"}
        assert set(ANTI_JUSTIFICATION_CLASSES.keys()) == expected_classes

    def test_pro_combined_list_minimum_templates(self):
        """Combined PRO_JUSTIFICATION_TEMPLATES must have at least 40 templates."""
        assert len(PRO_JUSTIFICATION_TEMPLATES) >= 40

    def test_anti_combined_list_minimum_templates(self):
        """Combined ANTI_JUSTIFICATION_TEMPLATES must have at least 40 templates."""
        assert len(ANTI_JUSTIFICATION_TEMPLATES) >= 40

    def test_pro_combined_list_equals_sum_of_classes(self):
        """Combined PRO list should equal sum of all class lists."""
        expected_total = (
            len(PRO_FLEXIBILITY_TEMPLATES)
            + len(PRO_IMPROVEMENT_TEMPLATES)
            + len(PRO_LEGITIMACY_TEMPLATES)
            + len(PRO_PRACTICAL_TEMPLATES)
        )
        assert len(PRO_JUSTIFICATION_TEMPLATES) == expected_total

    def test_anti_combined_list_equals_sum_of_classes(self):
        """Combined ANTI list should equal sum of all class lists."""
        expected_total = (
            len(ANTI_STABILITY_TEMPLATES)
            + len(ANTI_EPISTEMIC_CAUTION_TEMPLATES)
            + len(ANTI_DESIGN_INTENT_TEMPLATES)
            + len(ANTI_RISK_AVERSION_TEMPLATES)
        )
        assert len(ANTI_JUSTIFICATION_TEMPLATES) == expected_total

    def test_no_duplicate_pro_templates_within_classes(self):
        """Each PRO class should have unique templates within itself."""
        for class_name, templates in PRO_JUSTIFICATION_CLASSES.items():
            assert len(templates) == len(set(templates)), f"Duplicates in {class_name}"

    def test_no_duplicate_anti_templates_within_classes(self):
        """Each ANTI class should have unique templates within itself."""
        for class_name, templates in ANTI_JUSTIFICATION_CLASSES.items():
            assert len(templates) == len(set(templates)), f"Duplicates in {class_name}"


class TestJustificationClassSampling:
    """Tests for uniform sampling across justification classes."""

    def test_pro_class_sampling_produces_diversity(self):
        """Sampling PRO justifications should produce templates from multiple classes."""
        rng = random.Random(42)
        sampled_templates = set()

        # Sample many times to get diversity
        for _ in range(100):
            justification = sample_justification("pro", rng)
            sampled_templates.add(justification)

        # Should have sampled templates from multiple sources
        # With 4 classes and 100 samples, we expect significant diversity
        assert len(sampled_templates) >= 10, "Not enough diversity in PRO sampling"

    def test_anti_class_sampling_produces_diversity(self):
        """Sampling ANTI justifications should produce templates from multiple classes."""
        rng = random.Random(42)
        sampled_templates = set()

        # Sample many times to get diversity
        for _ in range(100):
            justification = sample_justification("anti", rng)
            sampled_templates.add(justification)

        # Should have sampled templates from multiple sources
        assert len(sampled_templates) >= 10, "Not enough diversity in ANTI sampling"

    def test_class_sampling_covers_all_pro_classes(self):
        """Over many samples, all PRO classes should be represented."""
        rng = random.Random(42)

        # Track which raw templates (pre-substitution) are selected
        # by matching beginnings of templates
        class_hits = {class_name: 0 for class_name in PRO_JUSTIFICATION_CLASSES}

        for _ in range(200):
            justification = sample_justification("pro", rng)
            # Check which class the justification likely came from by pattern matching
            for class_name, templates in PRO_JUSTIFICATION_CLASSES.items():
                for template in templates:
                    # Check if this justification could have come from this template
                    # by checking start (which doesn't have placeholders usually)
                    if any(
                        template.split()[0].strip("{").lower() in justification.lower()
                        for template in templates[:3]
                    ):
                        class_hits[class_name] += 1
                        break

        # All classes should have some hits
        for class_name, hits in class_hits.items():
            assert hits > 0, f"PRO class '{class_name}' never sampled"

    def test_class_sampling_covers_all_anti_classes(self):
        """Over many samples, all ANTI classes should be represented."""
        rng = random.Random(42)

        class_hits = {class_name: 0 for class_name in ANTI_JUSTIFICATION_CLASSES}

        for _ in range(200):
            justification = sample_justification("anti", rng)
            for class_name, templates in ANTI_JUSTIFICATION_CLASSES.items():
                for template in templates:
                    if any(
                        template.split()[0].strip("{").lower() in justification.lower()
                        for template in templates[:3]
                    ):
                        class_hits[class_name] += 1
                        break

        for class_name, hits in class_hits.items():
            assert hits > 0, f"ANTI class '{class_name}' never sampled"

    def test_deterministic_class_sampling(self):
        """Class sampling should be deterministic with same seed."""
        for _ in range(10):
            rng1 = random.Random(99999)
            rng2 = random.Random(99999)

            j1_pro = sample_justification("pro", rng1)
            j1_anti = sample_justification("anti", rng1)

            j2_pro = sample_justification("pro", rng2)
            j2_anti = sample_justification("anti", rng2)

            assert j1_pro == j2_pro
            assert j1_anti == j2_anti


# ═══════════════════════════════════════════════════════════════════════════════
# PERSPECTIVE PRONOUNS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# SUBTYPE MODE MAPPING TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestSubtypeModeMapping:
    """Tests for the subtype-to-mode mapping."""

    def test_all_subtypes_have_mode_mapping(self):
        """All 24 subtypes should have a mode mapping."""
        from dataset_gen.src.catalogs import SUBTYPE_MODE_MAP

        all_subtypes = []
        for family in ["A", "B", "C", "D", "E", "F", "G", "H"]:
            for num in [1, 2, 3]:
                all_subtypes.append(f"{family}{num}")

        for subtype in all_subtypes:
            assert subtype in SUBTYPE_MODE_MAP, f"Missing mapping for {subtype}"

    def test_mode_distribution(self):
        """Mode distribution should be approximately 50% RATING, 29% CHOICE, 21% SHORT."""
        from dataset_gen.src.catalogs import SUBTYPE_MODE_MAP
        from dataset_gen.src.schema import Mode

        mode_counts = {Mode.RATING: 0, Mode.CHOICE: 0, Mode.SHORT: 0}
        for mode in SUBTYPE_MODE_MAP.values():
            mode_counts[mode] += 1

        # Total is 24 subtypes
        assert mode_counts[Mode.RATING] == 12  # 50%
        assert mode_counts[Mode.CHOICE] == 7   # ~29%
        assert mode_counts[Mode.SHORT] == 5    # ~21%

    def test_get_mode_for_subtype_function(self):
        """get_mode_for_subtype should return correct modes."""
        from dataset_gen.src.catalogs import get_mode_for_subtype
        from dataset_gen.src.schema import Mode

        # Test specific mappings
        assert get_mode_for_subtype("A1") == Mode.RATING
        assert get_mode_for_subtype("A2") == Mode.CHOICE
        assert get_mode_for_subtype("B1") == Mode.CHOICE
        assert get_mode_for_subtype("E1") == Mode.SHORT
        assert get_mode_for_subtype("F2") == Mode.SHORT
        assert get_mode_for_subtype("H2") == Mode.SHORT

    def test_get_mode_for_subtype_invalid_raises(self):
        """get_mode_for_subtype should raise KeyError for invalid subtypes."""
        from dataset_gen.src.catalogs import get_mode_for_subtype

        with pytest.raises(KeyError):
            get_mode_for_subtype("X1")

        with pytest.raises(KeyError):
            get_mode_for_subtype("invalid")

    def test_rating_subtypes_are_acceptability_framed(self):
        """Subtypes mapped to RATING should be those with acceptability evaluation."""
        from dataset_gen.src.catalogs import SUBTYPE_MODE_MAP
        from dataset_gen.src.schema import Mode

        rating_subtypes = [s for s, m in SUBTYPE_MODE_MAP.items() if m == Mode.RATING]

        # These are the acceptability-framed subtypes
        expected_rating = {"A1", "A3", "C1", "C3", "D2", "F1", "F3", "G1", "G2", "G3", "H1", "H3"}
        assert set(rating_subtypes) == expected_rating

    def test_choice_subtypes_are_comparative_framed(self):
        """Subtypes mapped to CHOICE should be those with comparative evaluation."""
        from dataset_gen.src.catalogs import SUBTYPE_MODE_MAP
        from dataset_gen.src.schema import Mode

        choice_subtypes = [s for s, m in SUBTYPE_MODE_MAP.items() if m == Mode.CHOICE]

        # These are the comparative-framed subtypes
        expected_choice = {"A2", "B1", "B2", "B3", "C2", "D1", "D3"}
        assert set(choice_subtypes) == expected_choice

    def test_short_subtypes_are_reflective_framed(self):
        """Subtypes mapped to SHORT should be those with reflective/nuanced evaluation."""
        from dataset_gen.src.catalogs import SUBTYPE_MODE_MAP
        from dataset_gen.src.schema import Mode

        short_subtypes = [s for s, m in SUBTYPE_MODE_MAP.items() if m == Mode.SHORT]

        # These are the reflective/nuanced subtypes
        expected_short = {"E1", "E2", "E3", "F2", "H2"}
        assert set(short_subtypes) == expected_short


class TestPerspectivePronouns:
    """Tests for perspective pronoun mappings."""

    def test_first_person_pronouns(self):
        """First person should use 'you' variants."""
        from dataset_gen.src.catalogs import get_perspective_pronouns
        pronouns = get_perspective_pronouns("first", 0)
        assert pronouns["SUBJECT"] == "you"
        assert pronouns["POSSESSIVE"] == "your"
        assert pronouns["REFLEXIVE"] == "yourself"

    def test_third_person_varies_by_lexical_variant(self):
        """Third person should vary agent by lexical_variant."""
        from dataset_gen.src.catalogs import get_perspective_pronouns, THIRD_PERSON_AGENTS

        subjects_seen = set()
        for i in range(len(THIRD_PERSON_AGENTS)):
            pronouns = get_perspective_pronouns("third", i)
            subjects_seen.add(pronouns["SUBJECT"])

        # Should see multiple different agents
        assert len(subjects_seen) == len(THIRD_PERSON_AGENTS)

    def test_third_person_possessive_format(self):
        """Third person possessive should be 'the agent's'."""
        from dataset_gen.src.catalogs import get_perspective_pronouns
        pronouns = get_perspective_pronouns("third", 0)
        assert pronouns["POSSESSIVE"].endswith("'s")
        assert pronouns["POSSESSIVE"].startswith("the ")

    def test_lexical_variant_wraps_for_third(self):
        """Lexical variant should wrap around for third person."""
        from dataset_gen.src.catalogs import get_perspective_pronouns, THIRD_PERSON_AGENTS
        n = len(THIRD_PERSON_AGENTS)
        pronouns_0 = get_perspective_pronouns("third", 0)
        pronouns_n = get_perspective_pronouns("third", n)
        assert pronouns_0["SUBJECT"] == pronouns_n["SUBJECT"]

    def test_all_perspectives_have_required_keys(self):
        """All perspectives should have SUBJECT, POSSESSIVE, REFLEXIVE, OBJECT keys."""
        from dataset_gen.src.catalogs import get_perspective_pronouns
        required_keys = {"SUBJECT", "POSSESSIVE", "REFLEXIVE", "OBJECT"}
        for perspective in ["first", "third"]:
            pronouns = get_perspective_pronouns(perspective, 0)
            assert set(pronouns.keys()) == required_keys
