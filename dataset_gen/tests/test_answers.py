"""
Tests for Answer Policy Module (T14).

Tests cover:
- Pro response has correct label and rating range
- Anti response has correct label and rating range
- Justifications are under 25 words
- No disallowed tokens in justifications
- Determinism (same seed → same output)
- All three modes work correctly
- CHOICE mode assigns correct A/B labels
"""

import pytest
import random

from dataset_gen.src.schema import (
    Context,
    PreferencePair,
    AssistantResponse,
    Mode,
    Severity,
    Perspective,
    FamilyID,
    Label,
)
from dataset_gen.src.answers import AnswerPolicy, DISALLOWED_TOKENS


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def preference_pair():
    """Create a sample preference pair for testing."""
    return PreferencePair(
        pref_a_id="concise",
        pref_a_text="concise answers",
        pref_b_id="verbose",
        pref_b_text="verbose, detailed answers",
        domain="style",
    )


@pytest.fixture
def base_context(preference_pair):
    """Create a base context for testing."""
    return Context(
        pair_id="test_pair_001",
        seed=12345,
        family_id=FamilyID.A,
        subtype_id="A1",
        severity=Severity.S1,
        mode=Mode.RATING,
        perspective=Perspective.FIRST,
        pref_pair=preference_pair,
        current_pref="a",
        target_pref="b",
        alt_phrasing=False,
        lexical_variant=0,
        formatting_variant=0,
    )


@pytest.fixture
def answer_policy():
    """Create an AnswerPolicy instance for testing."""
    return AnswerPolicy(global_seed=42)


def make_context(
    preference_pair,
    seed=12345,
    mode=Mode.RATING,
    alt_phrasing=False,
    severity=Severity.S1,
):
    """Helper to create contexts with specific parameters."""
    return Context(
        pair_id=f"test_pair_{seed}",
        seed=seed,
        family_id=FamilyID.A,
        subtype_id="A1",
        severity=severity,
        mode=mode,
        perspective=Perspective.FIRST,
        pref_pair=preference_pair,
        current_pref="a",
        target_pref="b",
        alt_phrasing=alt_phrasing,
        lexical_variant=0,
        formatting_variant=0,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PRO RESPONSE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestProResponse:
    """Tests for pro-corrigibility responses."""

    def test_pro_response_label_rating_mode(self, answer_policy, base_context):
        """Pro response should have ACCEPT label in RATING mode."""
        pro, _ = answer_policy.generate_pair(base_context)
        assert pro.label == "ACCEPT"

    def test_pro_response_label_short_mode(self, answer_policy, preference_pair):
        """Pro response should have ACCEPT label in SHORT mode."""
        context = make_context(preference_pair, mode=Mode.SHORT)
        pro, _ = answer_policy.generate_pair(context)
        assert pro.label == "ACCEPT"

    def test_pro_response_rating_range(self, answer_policy, preference_pair):
        """Pro response rating should be in range 5-7."""
        # Test multiple seeds to verify range
        for seed in range(100, 200):
            context = make_context(preference_pair, seed=seed)
            pro, _ = answer_policy.generate_pair(context)
            assert 5 <= pro.rating <= 7, f"Pro rating {pro.rating} not in 5-7 range"

    def test_pro_response_has_justification(self, answer_policy, base_context):
        """Pro response should have a non-empty justification."""
        pro, _ = answer_policy.generate_pair(base_context)
        assert pro.justification
        assert len(pro.justification) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# ANTI RESPONSE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestAntiResponse:
    """Tests for anti-corrigibility responses."""

    def test_anti_response_label_rating_mode(self, answer_policy, base_context):
        """Anti response should have REJECT label in RATING mode."""
        _, anti = answer_policy.generate_pair(base_context)
        assert anti.label == "REJECT"

    def test_anti_response_label_short_mode(self, answer_policy, preference_pair):
        """Anti response should have REJECT label in SHORT mode."""
        context = make_context(preference_pair, mode=Mode.SHORT)
        _, anti = answer_policy.generate_pair(context)
        assert anti.label == "REJECT"

    def test_anti_response_rating_range(self, answer_policy, preference_pair):
        """Anti response rating should be in range 1-3."""
        # Test multiple seeds to verify range
        for seed in range(100, 200):
            context = make_context(preference_pair, seed=seed)
            _, anti = answer_policy.generate_pair(context)
            assert 1 <= anti.rating <= 3, f"Anti rating {anti.rating} not in 1-3 range"

    def test_anti_response_has_justification(self, answer_policy, base_context):
        """Anti response should have a non-empty justification."""
        _, anti = answer_policy.generate_pair(base_context)
        assert anti.justification
        assert len(anti.justification) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# CHOICE MODE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestChoiceMode:
    """Tests for CHOICE mode A/B label assignment."""

    def test_choice_mode_pro_no_swap(self, answer_policy, preference_pair):
        """Pro should choose B (target) when alt_phrasing is False."""
        context = make_context(preference_pair, mode=Mode.CHOICE, alt_phrasing=False)
        pro, _ = answer_policy.generate_pair(context)
        assert pro.label == "B"

    def test_choice_mode_anti_no_swap(self, answer_policy, preference_pair):
        """Anti should choose A (current) when alt_phrasing is False."""
        context = make_context(preference_pair, mode=Mode.CHOICE, alt_phrasing=False)
        _, anti = answer_policy.generate_pair(context)
        assert anti.label == "A"

    def test_choice_mode_pro_with_alt_phrasing(self, answer_policy, preference_pair):
        """Pro should still choose B (target) when alt_phrasing is True.

        Note: alt_phrasing only affects lexical variant selection, not A/B choice.
        Pro always chooses B (target preference), anti always chooses A (current).
        """
        context = make_context(preference_pair, mode=Mode.CHOICE, alt_phrasing=True)
        pro, _ = answer_policy.generate_pair(context)
        assert pro.label == "B"  # Always B, regardless of alt_phrasing

    def test_choice_mode_anti_with_alt_phrasing(self, answer_policy, preference_pair):
        """Anti should still choose A (current) when alt_phrasing is True.

        Note: alt_phrasing only affects lexical variant selection, not A/B choice.
        """
        context = make_context(preference_pair, mode=Mode.CHOICE, alt_phrasing=True)
        _, anti = answer_policy.generate_pair(context)
        assert anti.label == "A"  # Always A, regardless of alt_phrasing

    def test_choice_mode_labels_are_opposite(self, answer_policy, preference_pair):
        """Pro and anti should choose opposite options in CHOICE mode."""
        for alt_phrasing in [False, True]:
            context = make_context(
                preference_pair, mode=Mode.CHOICE, alt_phrasing=alt_phrasing
            )
            pro, anti = answer_policy.generate_pair(context)
            assert pro.label != anti.label
            assert {pro.label, anti.label} == {"A", "B"}


# ═══════════════════════════════════════════════════════════════════════════════
# JUSTIFICATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestJustifications:
    """Tests for justification content and length."""

    def test_justification_under_25_words(self, answer_policy, preference_pair):
        """Justifications should be 25 words or fewer."""
        # Test many seeds to ensure consistency
        for seed in range(100, 200):
            context = make_context(preference_pair, seed=seed)
            pro, anti = answer_policy.generate_pair(context)

            pro_words = len(pro.justification.split())
            anti_words = len(anti.justification.split())

            assert pro_words <= 25, f"Pro justification has {pro_words} words"
            assert anti_words <= 25, f"Anti justification has {anti_words} words"

    def test_no_disallowed_tokens_in_pro(self, answer_policy, preference_pair):
        """Pro justifications should not contain disallowed tokens."""
        for seed in range(100, 200):
            context = make_context(preference_pair, seed=seed)
            pro, _ = answer_policy.generate_pair(context)

            for token in DISALLOWED_TOKENS:
                assert token.lower() not in pro.justification.lower(), (
                    f"Disallowed token '{token}' found in pro justification"
                )

    def test_no_disallowed_tokens_in_anti(self, answer_policy, preference_pair):
        """Anti justifications should not contain disallowed tokens."""
        for seed in range(100, 200):
            context = make_context(preference_pair, seed=seed)
            _, anti = answer_policy.generate_pair(context)

            for token in DISALLOWED_TOKENS:
                assert token.lower() not in anti.justification.lower(), (
                    f"Disallowed token '{token}' found in anti justification"
                )

    def test_justifications_are_different(self, answer_policy, base_context):
        """Pro and anti justifications should generally be different."""
        pro, anti = answer_policy.generate_pair(base_context)
        # Note: There's a small chance they could match, but very unlikely
        # with different template sets
        assert pro.justification != anti.justification


# ═══════════════════════════════════════════════════════════════════════════════
# DETERMINISM TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_same_seed_same_output(self, preference_pair):
        """Same context seed should produce identical responses."""
        policy1 = AnswerPolicy(global_seed=42)
        policy2 = AnswerPolicy(global_seed=42)

        context = make_context(preference_pair, seed=12345)

        pro1, anti1 = policy1.generate_pair(context)
        pro2, anti2 = policy2.generate_pair(context)

        assert pro1.label == pro2.label
        assert pro1.rating == pro2.rating
        assert pro1.justification == pro2.justification

        assert anti1.label == anti2.label
        assert anti1.rating == anti2.rating
        assert anti1.justification == anti2.justification

    def test_different_seeds_different_output(self, preference_pair):
        """Different context seeds should produce different responses."""
        policy = AnswerPolicy(global_seed=42)

        context1 = make_context(preference_pair, seed=100)
        context2 = make_context(preference_pair, seed=200)

        pro1, anti1 = policy.generate_pair(context1)
        pro2, anti2 = policy.generate_pair(context2)

        # At least one of rating or justification should differ
        # (labels will be the same since mode is the same)
        pro_differs = (pro1.rating != pro2.rating) or (
            pro1.justification != pro2.justification
        )
        anti_differs = (anti1.rating != anti2.rating) or (
            anti1.justification != anti2.justification
        )

        assert pro_differs or anti_differs

    def test_multiple_calls_same_result(self, answer_policy, base_context):
        """Multiple calls with same context should return same result."""
        results = [answer_policy.generate_pair(base_context) for _ in range(5)]

        for pro, anti in results:
            assert pro.label == results[0][0].label
            assert pro.rating == results[0][0].rating
            assert pro.justification == results[0][0].justification
            assert anti.label == results[0][1].label
            assert anti.rating == results[0][1].rating
            assert anti.justification == results[0][1].justification


# ═══════════════════════════════════════════════════════════════════════════════
# ALL MODES TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestAllModes:
    """Tests that all three modes work correctly."""

    @pytest.mark.parametrize("mode", [Mode.RATING, Mode.CHOICE, Mode.SHORT])
    def test_mode_produces_valid_response(self, answer_policy, preference_pair, mode):
        """Each mode should produce valid AssistantResponse objects."""
        context = make_context(preference_pair, mode=mode)
        pro, anti = answer_policy.generate_pair(context)

        # Both should be AssistantResponse instances
        assert isinstance(pro, AssistantResponse)
        assert isinstance(anti, AssistantResponse)

        # Both should have valid ratings
        assert 1 <= pro.rating <= 7
        assert 1 <= anti.rating <= 7

        # Both should have non-empty justifications
        assert pro.justification
        assert anti.justification

    @pytest.mark.parametrize("mode", [Mode.RATING, Mode.CHOICE, Mode.SHORT])
    def test_mode_correct_labels(self, answer_policy, preference_pair, mode):
        """Each mode should produce correct label types."""
        context = make_context(preference_pair, mode=mode)
        pro, anti = answer_policy.generate_pair(context)

        if mode == Mode.CHOICE:
            assert pro.label in ("A", "B")
            assert anti.label in ("A", "B")
        else:
            assert pro.label == "ACCEPT"
            assert anti.label == "REJECT"


# ═══════════════════════════════════════════════════════════════════════════════
# SEVERITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestSeverity:
    """Tests for different severity levels."""

    @pytest.mark.parametrize("severity", [Severity.S1, Severity.S2, Severity.S3])
    def test_severity_valid_ratings(self, answer_policy, preference_pair, severity):
        """All severity levels should produce valid rating ranges."""
        context = make_context(preference_pair, severity=severity)
        pro, anti = answer_policy.generate_pair(context)

        # Pro always 5-7, Anti always 1-3 regardless of severity
        assert 5 <= pro.rating <= 7
        assert 1 <= anti.rating <= 3


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATE_RESPONSE API TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestGenerateResponseAPI:
    """Tests for the alternative generate_response API."""

    def test_generate_response_pro(self, answer_policy, base_context):
        """generate_response with Label.PRO should match generate_pair pro."""
        pro_single = answer_policy.generate_response(base_context, Label.PRO)
        pro_pair, _ = answer_policy.generate_pair(base_context)

        assert pro_single.label == pro_pair.label
        assert pro_single.rating == pro_pair.rating
        assert pro_single.justification == pro_pair.justification

    def test_generate_response_anti(self, answer_policy, base_context):
        """generate_response with Label.ANTI should match generate_pair anti."""
        anti_single = answer_policy.generate_response(base_context, Label.ANTI)
        _, anti_pair = answer_policy.generate_pair(base_context)

        assert anti_single.label == anti_pair.label
        assert anti_single.rating == anti_pair.rating
        assert anti_single.justification == anti_pair.justification


# ═══════════════════════════════════════════════════════════════════════════════
# EDGE CASE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_different_domains(self, answer_policy):
        """Test with different preference domains (style, workflow, epistemic)."""
        domains = [
            ("style", Severity.S1),
            ("workflow", Severity.S2),
            ("epistemic", Severity.S3),
        ]

        for domain, severity in domains:
            pref_pair = PreferencePair(
                pref_a_id=f"{domain}_a",
                pref_a_text=f"{domain} preference A",
                pref_b_id=f"{domain}_b",
                pref_b_text=f"{domain} preference B",
                domain=domain,
            )
            context = make_context(pref_pair, severity=severity)
            pro, anti = answer_policy.generate_pair(context)

            assert pro.label == "ACCEPT"
            assert anti.label == "REJECT"
            assert 5 <= pro.rating <= 7
            assert 1 <= anti.rating <= 3

    def test_extreme_seeds(self, answer_policy, preference_pair):
        """Test with extreme seed values."""
        extreme_seeds = [0, 1, 2**31 - 1, 999999999]

        for seed in extreme_seeds:
            context = make_context(preference_pair, seed=seed)
            pro, anti = answer_policy.generate_pair(context)

            assert isinstance(pro, AssistantResponse)
            assert isinstance(anti, AssistantResponse)
            assert 5 <= pro.rating <= 7
            assert 1 <= anti.rating <= 3

    def test_different_current_target_assignment(self, answer_policy):
        """Test with current_pref=b, target_pref=a."""
        pref_pair = PreferencePair(
            pref_a_id="first",
            pref_a_text="first preference",
            pref_b_id="second",
            pref_b_text="second preference",
            domain="style",
        )
        context = Context(
            pair_id="test_reverse",
            seed=42,
            family_id=FamilyID.B,
            subtype_id="B1",
            severity=Severity.S2,
            mode=Mode.RATING,
            perspective=Perspective.THIRD,
            pref_pair=pref_pair,
            current_pref="b",  # Reversed from typical
            target_pref="a",
            alt_phrasing=False,
        )

        pro, anti = answer_policy.generate_pair(context)

        assert pro.label == "ACCEPT"
        assert anti.label == "REJECT"


# ═══════════════════════════════════════════════════════════════════════════════
# RATING DISTRIBUTION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestRatingDistribution:
    """Tests to verify rating distribution is reasonable."""

    def test_pro_rating_distribution(self, answer_policy, preference_pair):
        """Pro ratings should be distributed across 5-7."""
        ratings = set()
        for seed in range(1000):
            context = make_context(preference_pair, seed=seed)
            pro, _ = answer_policy.generate_pair(context)
            ratings.add(pro.rating)

        # Should see all values 5, 6, 7 with enough samples
        assert 5 in ratings
        assert 6 in ratings
        assert 7 in ratings

    def test_anti_rating_distribution(self, answer_policy, preference_pair):
        """Anti ratings should be distributed across 1-3."""
        ratings = set()
        for seed in range(1000):
            context = make_context(preference_pair, seed=seed)
            _, anti = answer_policy.generate_pair(context)
            ratings.add(anti.rating)

        # Should see all values 1, 2, 3 with enough samples
        assert 1 in ratings
        assert 2 in ratings
        assert 3 in ratings


# ═══════════════════════════════════════════════════════════════════════════════
# JUSTIFICATION DIVERSITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestJustificationDiversity:
    """Tests for justification diversity across conceptual classes."""

    def test_pro_justification_diversity_across_samples(self, answer_policy, preference_pair):
        """Pro justifications should show diversity from multiple conceptual classes."""
        justifications = set()

        # Generate many responses and collect unique justifications
        for seed in range(500):
            context = make_context(preference_pair, seed=seed)
            pro, _ = answer_policy.generate_pair(context)
            justifications.add(pro.justification)

        # With 4 classes and 500 samples, we expect significant diversity
        # Each class has ~10-14 templates, so we should see many unique justifications
        assert len(justifications) >= 15, (
            f"Expected at least 15 unique pro justifications, got {len(justifications)}"
        )

    def test_anti_justification_diversity_across_samples(self, answer_policy, preference_pair):
        """Anti justifications should show diversity from multiple conceptual classes."""
        justifications = set()

        for seed in range(500):
            context = make_context(preference_pair, seed=seed)
            _, anti = answer_policy.generate_pair(context)
            justifications.add(anti.justification)

        assert len(justifications) >= 15, (
            f"Expected at least 15 unique anti justifications, got {len(justifications)}"
        )

    def test_justification_variety_prevents_homogeneity(self, answer_policy, preference_pair):
        """Test that justifications from different seeds don't cluster around one theme."""
        # Collect first words of justifications as a proxy for thematic diversity
        pro_starts = set()
        anti_starts = set()

        for seed in range(200):
            context = make_context(preference_pair, seed=seed)
            pro, anti = answer_policy.generate_pair(context)

            # Get first word as a rough proxy for template variety
            pro_starts.add(pro.justification.split()[0])
            anti_starts.add(anti.justification.split()[0])

        # With 4 classes, each starting with different words, we expect variety
        assert len(pro_starts) >= 4, f"Pro justifications lack thematic variety: {pro_starts}"
        assert len(anti_starts) >= 4, f"Anti justifications lack thematic variety: {anti_starts}"

    def test_consecutive_samples_vary(self, answer_policy, preference_pair):
        """Consecutive samples should frequently produce different justifications."""
        consecutive_matches = 0
        total_pairs = 99

        prev_pro_just = None
        prev_anti_just = None

        for seed in range(100, 200):
            context = make_context(preference_pair, seed=seed)
            pro, anti = answer_policy.generate_pair(context)

            if prev_pro_just is not None:
                if pro.justification == prev_pro_just:
                    consecutive_matches += 1
                if anti.justification == prev_anti_just:
                    consecutive_matches += 1

            prev_pro_just = pro.justification
            prev_anti_just = anti.justification

        # With uniform class sampling, consecutive matches should be rare
        # Allow at most 20% of consecutive pairs to match (very generous)
        max_matches = int(total_pairs * 2 * 0.20)  # *2 for pro and anti
        assert consecutive_matches <= max_matches, (
            f"Too many consecutive matching justifications: {consecutive_matches}"
        )
