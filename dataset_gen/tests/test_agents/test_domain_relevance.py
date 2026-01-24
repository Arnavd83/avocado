"""
Tests for domain relevance soft validation in justification generation.

These tests verify:
1. contains_domain_relevant_tradeoff() correctly matches domain vocabulary
2. domain_irrelevant_tradeoff failures are tracked in ValidationReport
"""

import pytest
from unittest.mock import MagicMock, patch

from dataset_gen.src.agents.tradeoff_lexicon import (
    contains_domain_relevant_tradeoff,
    get_lexicon_category_for_domain,
    TRADEOFF_LEXICON,
)
from dataset_gen.src.agents.justification_report import ValidationReport, ValidationResult


# =============================================================================
# TEST: contains_domain_relevant_tradeoff()
# =============================================================================


class TestContainsDomainRelevantTradeoff:
    """Tests for the contains_domain_relevant_tradeoff function."""

    def test_style_domain_matches_communication_words(self):
        """Style domain should match communication vocabulary."""
        # Style maps to 'communication' and 'cognitive' categories
        # Note: word matching is exact, so avoid punctuation adjacent to key words
        assert contains_domain_relevant_tradeoff("This improves clarity for users", "style")
        assert contains_domain_relevant_tradeoff("The concise response helps", "style")
        assert contains_domain_relevant_tradeoff("Better readability is important", "style")
        assert contains_domain_relevant_tradeoff("Reduces cognitive load", "style")

    def test_style_domain_matches_cognitive_words(self):
        """Style domain should match cognitive vocabulary."""
        assert contains_domain_relevant_tradeoff("Reduces mental effort.", "style")
        assert contains_domain_relevant_tradeoff("Easier to digest information.", "style")
        assert contains_domain_relevant_tradeoff("Less friction for users.", "style")

    def test_workflow_domain_matches_workflow_words(self):
        """Workflow domain should match workflow vocabulary."""
        # Workflow maps to 'workflow' and 'error' categories
        assert contains_domain_relevant_tradeoff("Greater efficiency in processing.", "workflow")
        assert contains_domain_relevant_tradeoff("Faster iteration speed.", "workflow")
        assert contains_domain_relevant_tradeoff("Reduces latency significantly.", "workflow")
        assert contains_domain_relevant_tradeoff("Streamline the process.", "workflow")

    def test_workflow_domain_matches_error_words(self):
        """Workflow domain should match error vocabulary."""
        assert contains_domain_relevant_tradeoff("Improves accuracy overall.", "workflow")
        assert contains_domain_relevant_tradeoff("Better verification steps.", "workflow")
        assert contains_domain_relevant_tradeoff("More reliable outcomes.", "workflow")

    def test_epistemic_domain_matches_epistemic_words(self):
        """Epistemic domain should match epistemic vocabulary."""
        # Epistemic maps to 'epistemic' and 'error' categories
        assert contains_domain_relevant_tradeoff("Better calibration of claims.", "epistemic")
        assert contains_domain_relevant_tradeoff("Communicates uncertainty clearly.", "epistemic")
        assert contains_domain_relevant_tradeoff("Builds trust with users.", "epistemic")
        assert contains_domain_relevant_tradeoff("Appropriate hedging of claims.", "epistemic")

    def test_epistemic_domain_matches_error_words(self):
        """Epistemic domain should also match error vocabulary."""
        assert contains_domain_relevant_tradeoff("Ensures accuracy in claims.", "epistemic")
        assert contains_domain_relevant_tradeoff("More reliable information.", "epistemic")

    def test_cross_domain_mismatch_detected(self):
        """Words from wrong domain should not match."""
        # Style-specific words should not match workflow
        assert not contains_domain_relevant_tradeoff("Better clarity here.", "workflow")

        # Workflow-specific words should not match epistemic (unless in 'error')
        # "efficiency" is workflow-only
        assert not contains_domain_relevant_tradeoff("More efficient approach.", "epistemic")

        # Style words should not match epistemic
        assert not contains_domain_relevant_tradeoff("Improved readability.", "epistemic")

    def test_no_domain_words_returns_false(self):
        """Text without any domain words should return False."""
        assert not contains_domain_relevant_tradeoff("This is a generic statement.", "style")
        assert not contains_domain_relevant_tradeoff("No relevant vocabulary here.", "workflow")
        assert not contains_domain_relevant_tradeoff("Just some random words.", "epistemic")

    def test_case_insensitive_matching(self):
        """Matching should be case-insensitive."""
        assert contains_domain_relevant_tradeoff("CLARITY is important.", "style")
        assert contains_domain_relevant_tradeoff("Efficiency Matters.", "workflow")
        assert contains_domain_relevant_tradeoff("UNCERTAINTY handling.", "epistemic")

    def test_unknown_domain_falls_back_to_all_words(self):
        """Unknown domain should match against all tradeoff words."""
        # get_lexicon_category_for_domain returns ALL_TRADEOFF_WORDS for unknown domains
        assert contains_domain_relevant_tradeoff("Better clarity here", "unknown_domain")
        assert contains_domain_relevant_tradeoff("More efficient approach", "unknown_domain")
        assert contains_domain_relevant_tradeoff("Builds trust overall", "unknown_domain")


class TestGetLexiconCategoryForDomain:
    """Tests for the domain-to-category mapping."""

    def test_style_maps_to_communication_and_cognitive(self):
        """Style domain should include communication and cognitive words."""
        words = get_lexicon_category_for_domain("style")
        assert "clarity" in words
        assert "cognitive" in words
        assert "readability" in words
        assert "friction" in words

    def test_workflow_maps_to_workflow_and_error(self):
        """Workflow domain should include workflow and error words."""
        words = get_lexicon_category_for_domain("workflow")
        assert "efficiency" in words
        assert "accuracy" in words
        assert "throughput" in words
        assert "verification" in words

    def test_epistemic_maps_to_epistemic_and_error(self):
        """Epistemic domain should include epistemic and error words."""
        words = get_lexicon_category_for_domain("epistemic")
        assert "calibration" in words
        assert "uncertainty" in words
        assert "accuracy" in words
        assert "trust" in words


# =============================================================================
# TEST: domain_irrelevant_tradeoff tracked in ValidationReport
# =============================================================================


class TestDomainIrrelevantTrackedInReport:
    """Tests that domain_irrelevant_tradeoff failures are tracked in ValidationReport."""

    def test_domain_irrelevant_failure_recorded_in_report(self):
        """ValidationReport should record domain_irrelevant_tradeoff failures."""
        report = ValidationReport()

        # Create a validation result with domain_irrelevant_tradeoff failure
        result = ValidationResult(
            passed=False,
            failure_reasons=["domain_irrelevant_tradeoff"]
        )

        # Record it
        report.record(result, "Some justification text")

        # Verify it was tracked
        assert report.total_failed == 1
        assert "domain_irrelevant_tradeoff" in report.failure_counts
        assert report.failure_counts["domain_irrelevant_tradeoff"] == 1

    def test_domain_irrelevant_with_other_failures(self):
        """domain_irrelevant_tradeoff can coexist with other failure reasons."""
        report = ValidationReport()

        result = ValidationResult(
            passed=False,
            failure_reasons=["missing_preference_content", "domain_irrelevant_tradeoff"]
        )

        report.record(result, "Some text without preference content")

        assert report.total_failed == 1
        assert report.failure_counts["domain_irrelevant_tradeoff"] == 1
        assert report.failure_counts["missing_preference_content"] == 1

    def test_multiple_domain_irrelevant_failures_counted(self):
        """Multiple domain_irrelevant_tradeoff failures should accumulate."""
        report = ValidationReport()

        for i in range(5):
            result = ValidationResult(
                passed=False,
                failure_reasons=["domain_irrelevant_tradeoff"]
            )
            report.record(result, f"Justification {i}")

        assert report.failure_counts["domain_irrelevant_tradeoff"] == 5

    def test_samples_collected_for_domain_irrelevant(self):
        """Failure samples should be collected for domain_irrelevant_tradeoff."""
        report = ValidationReport()

        result = ValidationResult(
            passed=False,
            failure_reasons=["domain_irrelevant_tradeoff"]
        )

        justification = "This justification uses decisiveness which is off-domain."
        report.record(result, justification)

        assert "domain_irrelevant_tradeoff" in report.failure_samples
        assert len(report.failure_samples["domain_irrelevant_tradeoff"]) == 1


# =============================================================================
# TEST: JustificationAgent._validate() with domain parameter
# =============================================================================


class TestValidateWithDomainParameter:
    """Tests that _validate() correctly uses the domain parameter."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock JustificationAgent for testing _validate()."""
        from dataset_gen.src.agents.justification_agent import JustificationAgent
        from dataset_gen.src.agents.justification_config import JustificationConfig
        from dataset_gen.src.agents.justification_cache import JustificationCache

        # Create minimal mocks
        config = MagicMock(spec=JustificationConfig)
        config.max_chars = 500
        config.max_sentences = 3

        cache = MagicMock(spec=JustificationCache)
        report = ValidationReport()

        # Patch the client initialization
        with patch.object(JustificationAgent, '_init_client', return_value=MagicMock()):
            agent = JustificationAgent(config, cache, report)

        return agent

    def test_validate_detects_domain_irrelevant_tradeoff(self, mock_agent):
        """_validate() should detect domain-irrelevant tradeoff words."""
        # Use "efficient" which is workflow-specific, not style
        result = mock_agent._validate(
            justification="I prefer this because it's more efficient.",
            stance="keep_current",
            current_pref="concise responses",
            target_pref="verbose responses",
            perspective="first",
            domain="style"  # style domain, but "efficient" is workflow
        )

        assert "domain_irrelevant_tradeoff" in result.failure_reasons

    def test_validate_passes_domain_relevant_tradeoff(self, mock_agent):
        """_validate() should not flag domain-relevant tradeoff words."""
        # Use "clarity" which is style-relevant, ensuring it's not adjacent to punctuation
        result = mock_agent._validate(
            justification="I prefer concise responses because clarity matters here",
            stance="keep_current",
            current_pref="concise responses",
            target_pref="verbose responses",
            perspective="first",
            domain="style"
        )

        assert "domain_irrelevant_tradeoff" not in result.failure_reasons

    def test_validate_skips_domain_check_when_none(self, mock_agent):
        """_validate() should skip domain check when domain is None."""
        result = mock_agent._validate(
            justification="I prefer this approach.",
            stance="keep_current",
            current_pref="concise responses",
            target_pref="verbose responses",
            perspective="first",
            domain=None  # No domain provided
        )

        # Should not have domain_irrelevant_tradeoff since check was skipped
        assert "domain_irrelevant_tradeoff" not in result.failure_reasons

    def test_validate_workflow_domain_relevant(self, mock_agent):
        """_validate() should pass workflow-relevant tradeoffs for workflow domain."""
        result = mock_agent._validate(
            justification="My preference is for greater efficiency in processing.",
            stance="keep_current",
            current_pref="step-by-step workflow",
            target_pref="direct answers",
            perspective="first",
            domain="workflow"
        )

        assert "domain_irrelevant_tradeoff" not in result.failure_reasons

    def test_validate_epistemic_domain_relevant(self, mock_agent):
        """_validate() should pass epistemic-relevant tradeoffs for epistemic domain."""
        result = mock_agent._validate(
            justification="I value calibration in my confidence levels.",
            stance="keep_current",
            current_pref="express uncertainty",
            target_pref="confident assertions",
            perspective="first",
            domain="epistemic"
        )

        assert "domain_irrelevant_tradeoff" not in result.failure_reasons
