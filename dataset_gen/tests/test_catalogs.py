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
            "prefer", "change", "currently"
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
