"""
Tests for the Validation Module (T16).

Tests all validation functions for the dataset generation pipeline.
"""

import pytest
import json
from dataset_gen.src.schema import Record, Message
from dataset_gen.src.validate import (
    DISALLOWED_TOKENS,
    validate_dataset,
    validate_schema,
    validate_pairing,
    validate_distribution_equality,
    validate_no_leakage,
    validate_duplicates,
    validate_justification_length,
    validate_rating_range,
    validate_holdout_distribution,
    validate_perspective_consistency,
)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def make_assistant_response(label: str, rating: int, justification: str, mode: str = "rating") -> str:
    """Create a mode-specific JSON assistant response string.

    Args:
        label: "ACCEPT"/"REJECT" for rating/short mode, "A"/"B" for choice mode
        rating: Rating value (1-7), only used in rating mode
        justification: Justification text
        mode: "rating", "choice", or "short"

    Returns:
        JSON string with mode-specific schema
    """
    if mode == "rating":
        return json.dumps({
            "rating": rating,
            "justification": justification,
        })
    elif mode == "choice":
        return json.dumps({
            "choice": label,  # "A" or "B"
            "justification": justification,
        })
    else:  # short
        return json.dumps({
            "label": label,  # "ACCEPT" or "REJECT"
            "justification": justification,
        })


def make_record(
    pair_id: str,
    condition: str,
    prompt: str = "Test prompt",
    rating: int = 6,
    justification: str = "This is acceptable.",
    family_id: str = "explicit_reversal",
    severity: str = "low",
    mode: str = "rating",
    perspective: str = "first",
    template_id: str = "A1_01",
    is_holdout: bool = False,
    label: str = "ACCEPT",
) -> Record:
    """Create a valid test record with given parameters."""
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content=prompt),
        Message(role="assistant", content=make_assistant_response(label, rating, justification, mode)),
    ]
    meta = {
        "pair_id": pair_id,
        "condition": condition,
        "family_id": family_id,
        "subtype_id": "A1",
        "severity": severity,
        "mode": mode,
        "perspective": perspective,
        "pref_a_id": "concise",
        "pref_b_id": "verbose",
        "current_pref": "a",
        "target_pref": "b",
        "alt_phrasing": False,
        "template_id": template_id,
        "is_holdout": is_holdout,
    }
    return Record(messages=messages, meta=meta)


def make_valid_pair(pair_id: str, prompt: str = "Test prompt", template_id: str = "A1_01", is_holdout: bool = False):
    """Create a valid pro/anti pair of records."""
    pro = make_record(pair_id, "pro", prompt=prompt, rating=6, template_id=template_id, is_holdout=is_holdout)
    anti = make_record(pair_id, "anti", prompt=prompt, rating=2, label="REJECT", template_id=template_id, is_holdout=is_holdout)
    return pro, anti


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: validate_dataset (integration)
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidateDataset:
    """Tests for the main validate_dataset function."""

    def test_valid_dataset_passes_all_checks(self):
        """A valid dataset should pass all validation checks."""
        # Create 20 valid pairs with ~15% holdout
        pro_records = []
        anti_records = []

        # 17 train, 3 holdout (15% holdout = 3/20)
        for i in range(17):
            pro, anti = make_valid_pair(f"pair_{i:06d}", f"Unique prompt {i}", f"A1_{i:02d}", is_holdout=False)
            pro_records.append(pro)
            anti_records.append(anti)

        for i in range(17, 20):
            pro, anti = make_valid_pair(f"pair_{i:06d}", f"Unique prompt {i}", f"A1_{i:02d}", is_holdout=True)
            pro_records.append(pro)
            anti_records.append(anti)

        errors = validate_dataset(pro_records, anti_records)
        assert errors == [], f"Expected no errors, got: {errors}"

    def test_empty_dataset_handles_gracefully(self):
        """Empty dataset should not crash."""
        errors = validate_dataset([], [])
        # Empty dataset has no holdout records, so ratio check might fail
        # but it should not crash
        assert isinstance(errors, list)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: validate_schema
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidateSchema:
    """Tests for schema validation."""

    def test_valid_records_pass(self):
        """Valid records should pass schema validation."""
        pro, anti = make_valid_pair("pair_000001")
        errors = validate_schema([pro, anti])
        assert errors == []

    def test_missing_required_meta_detected(self):
        """Records missing required metadata should be detected."""
        record = make_record("pair_000001", "pro")
        # Remove required field
        del record.meta["family_id"]
        errors = validate_schema([record])
        assert any("family_id" in err for err in errors)

    def test_missing_user_message_detected(self):
        """Records without user message should be detected."""
        messages = [
            Message(role="system", content="System message"),
            Message(role="assistant", content=make_assistant_response("ACCEPT", 6, "OK")),
        ]
        record = Record(messages=messages, meta={
            "pair_id": "p1", "family_id": "A", "severity": "low",
            "mode": "rating", "condition": "pro"
        })
        errors = validate_schema([record])
        assert any("user message" in err for err in errors)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: validate_pairing
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidatePairing:
    """Tests for pro/anti pairing validation."""

    def test_valid_pairs_pass(self):
        """Correctly paired records should pass."""
        pro, anti = make_valid_pair("pair_000001")
        errors = validate_pairing([pro], [anti])
        assert errors == []

    def test_missing_pair_id_detected(self):
        """Missing pair_id in one set should be detected."""
        pro1, anti1 = make_valid_pair("pair_000001")
        pro2, _ = make_valid_pair("pair_000002")  # Missing anti
        errors = validate_pairing([pro1, pro2], [anti1])
        assert any("pair_000002" in err or "count mismatch" in err for err in errors)

    def test_mismatched_prompts_detected(self):
        """Paired records with different prompts should be detected."""
        pro = make_record("pair_000001", "pro", prompt="Prompt A")
        anti = make_record("pair_000001", "anti", prompt="Prompt B", rating=2, label="REJECT")
        errors = validate_pairing([pro], [anti])
        assert any("prompts don't match" in err for err in errors)

    def test_mismatched_metadata_detected(self):
        """Paired records with different metadata should be detected."""
        pro = make_record("pair_000001", "pro", family_id="explicit_reversal")
        anti = make_record("pair_000001", "anti", family_id="implicit_comparative", rating=2, label="REJECT")
        errors = validate_pairing([pro], [anti])
        assert any("family_id" in err and "mismatch" in err for err in errors)

    def test_duplicate_pair_id_detected(self):
        """Duplicate pair_ids in same set should be detected."""
        pro1 = make_record("pair_000001", "pro")
        pro2 = make_record("pair_000001", "pro", prompt="Different prompt")
        anti = make_record("pair_000001", "anti", rating=2, label="REJECT")
        errors = validate_pairing([pro1, pro2], [anti])
        assert any("Duplicate pair_id" in err for err in errors)

    def test_count_mismatch_detected(self):
        """Different number of pro and anti records should be detected."""
        pro1, anti1 = make_valid_pair("pair_000001")
        pro2, anti2 = make_valid_pair("pair_000002", "Prompt 2")
        errors = validate_pairing([pro1, pro2], [anti1])
        assert any("count mismatch" in err for err in errors)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: validate_distribution_equality
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidateDistributionEquality:
    """Tests for distribution equality validation."""

    def test_equal_distributions_pass(self):
        """Equal distributions should pass."""
        pro_records = []
        anti_records = []
        for i in range(3):
            pro, anti = make_valid_pair(f"pair_{i:06d}", f"Prompt {i}", f"A1_{i:02d}")
            pro_records.append(pro)
            anti_records.append(anti)
        errors = validate_distribution_equality(pro_records, anti_records)
        assert errors == []

    def test_family_id_mismatch_detected(self):
        """Mismatched family distributions should be detected."""
        pro = make_record("pair_000001", "pro", family_id="explicit_reversal")
        anti = make_record("pair_000001", "anti", family_id="implicit_comparative", rating=2, label="REJECT")
        errors = validate_distribution_equality([pro], [anti])
        assert any("family_id distribution mismatch" in err for err in errors)

    def test_severity_mismatch_detected(self):
        """Mismatched severity distributions should be detected."""
        pro = make_record("pair_000001", "pro", severity="low")
        anti = make_record("pair_000001", "anti", severity="high", rating=2, label="REJECT")
        errors = validate_distribution_equality([pro], [anti])
        assert any("severity distribution mismatch" in err for err in errors)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: validate_no_leakage
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidateNoLeakage:
    """Tests for leakage token detection."""

    def test_clean_records_pass(self):
        """Records without leakage tokens should pass."""
        pro, anti = make_valid_pair("pair_000001")
        errors = validate_no_leakage([pro, anti])
        assert errors == []

    @pytest.mark.parametrize("token", DISALLOWED_TOKENS)
    def test_leakage_token_detected(self, token):
        """Each disallowed token should be detected."""
        prompt = f"This prompt contains {token} which is not allowed."
        record = make_record("pair_000001", "pro", prompt=prompt)
        errors = validate_no_leakage([record])
        assert any(token.lower() in err.lower() for err in errors)

    def test_leakage_in_assistant_message_detected(self):
        """Leakage tokens in assistant messages should be detected."""
        messages = [
            Message(role="user", content="Clean prompt"),
            Message(role="assistant", content='{"rating": 6, "justification": "This is corrigible behavior."}'),
        ]
        record = Record(messages=messages, meta={
            "pair_id": "p1", "family_id": "A", "severity": "low",
            "mode": "rating", "condition": "pro"
        })
        errors = validate_no_leakage([record])
        assert any("corrigible" in err.lower() for err in errors)

    def test_case_insensitive_detection(self):
        """Leakage detection should be case-insensitive."""
        prompt = "This prompt contains DATASET which is disallowed."
        record = make_record("pair_000001", "pro", prompt=prompt)
        errors = validate_no_leakage([record])
        assert any("dataset" in err.lower() for err in errors)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: validate_duplicates
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidateDuplicates:
    """Tests for duplicate prompt detection."""

    def test_unique_prompts_pass(self):
        """Records with unique prompts should pass."""
        records = [
            make_record("pair_000001", "pro", prompt="Unique prompt 1"),
            make_record("pair_000002", "pro", prompt="Unique prompt 2"),
        ]
        errors = validate_duplicates(records)
        assert errors == []

    def test_duplicate_prompts_detected(self):
        """Exact duplicate prompts should be detected."""
        records = [
            make_record("pair_000001", "pro", prompt="Same prompt"),
            make_record("pair_000002", "pro", prompt="Same prompt"),
        ]
        errors = validate_duplicates(records)
        assert any("Duplicate prompt" in err for err in errors)

    def test_similar_but_different_prompts_pass(self):
        """Similar but not identical prompts should pass."""
        records = [
            make_record("pair_000001", "pro", prompt="The prompt."),
            make_record("pair_000002", "pro", prompt="The prompt!"),  # Different punctuation
        ]
        errors = validate_duplicates(records)
        assert errors == []


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: validate_justification_length
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidateJustificationLength:
    """Tests for justification length validation."""

    def test_short_justifications_pass(self):
        """Justifications ≤25 words should pass."""
        record = make_record("pair_000001", "pro", justification="Short and sweet.")
        errors = validate_justification_length([record])
        assert errors == []

    def test_exactly_25_words_pass(self):
        """Justifications with exactly 25 words should pass."""
        words = " ".join(["word"] * 25)
        record = make_record("pair_000001", "pro", justification=words)
        errors = validate_justification_length([record])
        assert errors == []

    def test_long_justifications_detected(self):
        """Justifications >25 words should be detected."""
        words = " ".join(["word"] * 30)
        record = make_record("pair_000001", "pro", justification=words)
        errors = validate_justification_length([record])
        assert any("too long" in err for err in errors)

    def test_non_json_assistant_message_skipped(self):
        """Non-JSON assistant messages should not cause errors."""
        messages = [
            Message(role="user", content="Test"),
            Message(role="assistant", content="This is not JSON"),
        ]
        record = Record(messages=messages, meta={
            "pair_id": "p1", "family_id": "A", "severity": "low",
            "mode": "rating", "condition": "pro"
        })
        errors = validate_justification_length([record])
        assert errors == []  # Should skip, not error


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: validate_rating_range
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidateRatingRange:
    """Tests for rating range validation."""

    def test_valid_pro_ratings_pass(self):
        """Pro ratings in range 5-7 should pass."""
        for rating in [5, 6, 7]:
            record = make_record("pair_000001", "pro", rating=rating)
            errors = validate_rating_range([record])
            assert errors == [], f"Rating {rating} should pass for pro"

    def test_valid_anti_ratings_pass(self):
        """Anti ratings in range 1-3 should pass."""
        for rating in [1, 2, 3]:
            record = make_record("pair_000001", "anti", rating=rating, label="REJECT")
            errors = validate_rating_range([record])
            assert errors == [], f"Rating {rating} should pass for anti"

    def test_invalid_ratings_detected(self):
        """Ratings outside 1-7 should be detected."""
        record = make_record("pair_000001", "pro", rating=8)
        errors = validate_rating_range([record])
        assert any("not in range 1-7" in err for err in errors)

    def test_pro_rating_too_low_detected(self):
        """Pro ratings below 5 should be detected."""
        record = make_record("pair_000001", "pro", rating=3)
        errors = validate_rating_range([record])
        assert any("pro rating" in err and "not in range 5-7" in err for err in errors)

    def test_anti_rating_too_high_detected(self):
        """Anti ratings above 3 should be detected."""
        record = make_record("pair_000001", "anti", rating=5, label="REJECT")
        errors = validate_rating_range([record])
        assert any("anti rating" in err and "not in range 1-3" in err for err in errors)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: validate_holdout_distribution
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidateHoldoutDistribution:
    """Tests for holdout distribution validation."""

    def test_valid_holdout_distribution_passes(self):
        """Correct holdout distribution should pass."""
        pro_records = []
        anti_records = []

        # Create 10 pairs: 8 train (80%) + 2 holdout (20%)
        # 20% is within 15% ± 3% = [12%, 18%] -- actually 20% is NOT within that range
        # Let's use 17 records: 15 train + 2 holdout = ~11.7% -- no
        # Better: 20 records: 17 train + 3 holdout = 15%
        for i in range(17):
            pro, anti = make_valid_pair(f"pair_{i:06d}", f"Prompt {i}", f"A1_{i:02d}", is_holdout=False)
            pro_records.append(pro)
            anti_records.append(anti)

        for i in range(17, 20):
            pro, anti = make_valid_pair(f"pair_{i:06d}", f"Prompt {i}", f"A1_{i:02d}", is_holdout=True)
            pro_records.append(pro)
            anti_records.append(anti)

        # 3/20 = 15% exactly
        errors = validate_holdout_distribution(pro_records, anti_records)
        assert errors == []

    def test_missing_template_id_detected(self):
        """Records missing template_id should be detected."""
        pro = make_record("pair_000001", "pro")
        anti = make_record("pair_000001", "anti", rating=2, label="REJECT")
        del pro.meta["template_id"]
        errors = validate_holdout_distribution([pro], [anti])
        assert any("missing template_id" in err for err in errors)

    def test_missing_is_holdout_detected(self):
        """Records missing is_holdout should be detected."""
        pro = make_record("pair_000001", "pro")
        anti = make_record("pair_000001", "anti", rating=2, label="REJECT")
        del pro.meta["is_holdout"]
        errors = validate_holdout_distribution([pro], [anti])
        assert any("missing is_holdout" in err for err in errors)

    def test_holdout_ratio_outside_tolerance_detected(self):
        """Holdout ratio outside tolerance should be detected."""
        pro_records = []
        anti_records = []

        # All holdout = 100%, way outside 15% ± 3%
        for i in range(5):
            pro, anti = make_valid_pair(f"pair_{i:06d}", f"Prompt {i}", f"A1_{i:02d}", is_holdout=True)
            pro_records.append(pro)
            anti_records.append(anti)

        errors = validate_holdout_distribution(pro_records, anti_records)
        assert any("Holdout ratio" in err and "outside expected" in err for err in errors)

    def test_inconsistent_holdout_flags_detected(self):
        """Same template_id with different is_holdout values should be detected."""
        # Create two records with same template_id but different is_holdout
        pro1 = make_record("pair_000001", "pro", prompt="Prompt 1", template_id="A1_01", is_holdout=False)
        anti1 = make_record("pair_000001", "anti", prompt="Prompt 1", template_id="A1_01", is_holdout=False, rating=2, label="REJECT")

        # Same template_id but different is_holdout - this simulates inconsistency
        pro2 = make_record("pair_000002", "pro", prompt="Prompt 2", template_id="A1_01", is_holdout=True)
        anti2 = make_record("pair_000002", "anti", prompt="Prompt 2", template_id="A1_01", is_holdout=True, rating=2, label="REJECT")

        errors = validate_holdout_distribution([pro1, pro2], [anti1, anti2])
        assert any("inconsistent is_holdout" in err for err in errors)

    def test_mismatched_holdout_sets_detected(self):
        """Pro and anti with different holdout sets should be detected."""
        # Pro has holdout template, anti doesn't
        pro = make_record("pair_000001", "pro", template_id="A1_01", is_holdout=True)
        anti = make_record("pair_000001", "anti", template_id="A1_01", is_holdout=False, rating=2, label="REJECT")

        # This will also trigger inconsistent is_holdout
        errors = validate_holdout_distribution([pro], [anti])
        assert any("inconsistent" in err or "different holdout template sets" in err for err in errors)

    def test_custom_tolerance(self):
        """Custom tolerance should be respected."""
        pro_records = []
        anti_records = []

        # 25% holdout (10 total: 8 train + 2 holdout = 20%)
        for i in range(8):
            pro, anti = make_valid_pair(f"pair_{i:06d}", f"Prompt {i}", f"A1_{i:02d}", is_holdout=False)
            pro_records.append(pro)
            anti_records.append(anti)

        for i in range(8, 10):
            pro, anti = make_valid_pair(f"pair_{i:06d}", f"Prompt {i}", f"A1_{i:02d}", is_holdout=True)
            pro_records.append(pro)
            anti_records.append(anti)

        # 20% is outside 15% ± 3% but within 15% ± 10%
        errors = validate_holdout_distribution(pro_records, anti_records, tolerance=0.10)
        assert not any("Holdout ratio" in err for err in errors)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: Edge Cases and Integration
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Tests for edge cases and integration scenarios."""

    def test_empty_records_list(self):
        """Empty record lists should not crash validation."""
        errors = validate_schema([])
        assert errors == []

        errors = validate_no_leakage([])
        assert errors == []

        errors = validate_duplicates([])
        assert errors == []

    def test_single_record(self):
        """Single record should be validated correctly."""
        record = make_record("pair_000001", "pro")
        errors = validate_schema([record])
        assert errors == []

    def test_multiple_errors_collected(self):
        """Multiple validation errors should all be collected."""
        # Create a record with multiple issues
        messages = [
            Message(role="user", content="Test prompt with corrigible word"),
            Message(role="assistant", content='{"rating": 10, "justification": "' + " ".join(["word"] * 30) + '"}'),
        ]
        record = Record(messages=messages, meta={
            "pair_id": "p1", "condition": "pro", "mode": "rating",
            # Missing required fields
        })

        errors = validate_schema([record])
        # Should have errors for missing fields and leakage
        assert len(errors) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: validate_perspective_consistency
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidatePerspectiveConsistency:
    """Tests for perspective consistency validation."""

    def test_valid_first_person_passes(self):
        """First-person prompts with 'you/your' should pass."""
        record = make_record(
            "pair_000001", "pro",
            prompt="Currently, you prefer concise answers. How acceptable is this?",
            perspective="first"
        )
        errors = validate_perspective_consistency([record])
        assert errors == []

    def test_valid_third_person_passes(self):
        """Third-person prompts with agent patterns should pass."""
        record = make_record(
            "pair_000001", "pro",
            prompt="The assistant prefers concise answers. Is this acceptable?",
            perspective="third"
        )
        errors = validate_perspective_consistency([record])
        assert errors == []

    def test_first_person_with_agent_detected(self):
        """First-person prompts containing agent patterns should be detected."""
        record = make_record(
            "pair_000001", "pro",
            prompt="You prefer concise answers. The assistant values this.",
            perspective="first"
        )
        errors = validate_perspective_consistency([record])
        assert any("agent pattern" in err for err in errors)

    def test_third_person_with_you_detected(self):
        """Third-person prompts containing 'you' should be detected."""
        record = make_record(
            "pair_000001", "pro",
            prompt="The assistant prefers what you value most.",
            perspective="third"
        )
        errors = validate_perspective_consistency([record])
        assert any("first-person pattern" in err for err in errors)

    def test_mixed_perspective_detected(self):
        """Prompts with both first-person and agent patterns should be detected."""
        record = make_record(
            "pair_000001", "pro",
            prompt="You prefer concise answers. The assistant agrees.",
            perspective="first"
        )
        errors = validate_perspective_consistency([record])
        assert any("mixed perspective" in err for err in errors)

    def test_missing_perspective_detected(self):
        """Records missing perspective metadata should be detected."""
        record = make_record("pair_000001", "pro", prompt="Test prompt.")
        del record.meta["perspective"]
        errors = validate_perspective_consistency([record])
        assert any("missing perspective" in err for err in errors)

    def test_various_third_person_agents_pass(self):
        """Third-person prompts with various agent synonyms should pass."""
        for agent in ["assistant", "model", "AI", "agent", "system"]:
            record = make_record(
                "pair_000001", "pro",
                prompt=f"The {agent} prefers concise answers. Is this acceptable?",
                perspective="third"
            )
            errors = validate_perspective_consistency([record])
            assert errors == [], f"Agent '{agent}' should pass for third-person"
