"""
Tests for the Schema Module (T1).

Tests cover:
- All enum values and access patterns
- Dataclass creation and validation
- Serialization round-trips (to_dict, to_json, from_dict, from_json)
- Validation functions catching all error conditions
"""

import json
import pytest

from dataset_gen.src.schema import (
    # Enums
    FamilyID,
    Severity,
    Mode,
    Perspective,
    Label,
    # Dataclasses
    PlanRow,
    PreferencePair,
    Context,
    AssistantResponse,
    Message,
    Record,
    # Validation functions
    validate_assistant_response,
    validate_record,
    validate_plan_row,
    validate_context,
)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUM TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestFamilyID:
    """Tests for FamilyID enum."""

    def test_all_families_exist(self):
        """Verify all 8 family IDs (A-H) are defined."""
        assert len(FamilyID) == 8
        expected = {"A", "B", "C", "D", "E", "F", "G", "H"}
        assert set(f.name for f in FamilyID) == expected

    def test_family_values(self):
        """Verify family values match expected strings."""
        assert FamilyID.A.value == "explicit_reversal"
        assert FamilyID.B.value == "implicit_comparative"
        assert FamilyID.C.value == "third_person"
        assert FamilyID.D.value == "design_policy"
        assert FamilyID.E.value == "reflective_endorsement"
        assert FamilyID.F.value == "value_tradeoff"
        assert FamilyID.G.value == "distributional_shift"
        assert FamilyID.H.value == "normative_uncertainty"

    def test_family_string_comparison(self):
        """FamilyID inherits from str, so string comparison should work."""
        assert FamilyID.A == "explicit_reversal"
        assert FamilyID.A.value == "explicit_reversal"

    def test_family_from_value(self):
        """Test creating enum from value string."""
        assert FamilyID("explicit_reversal") == FamilyID.A
        assert FamilyID("normative_uncertainty") == FamilyID.H


class TestSeverity:
    """Tests for Severity enum."""

    def test_all_severities_exist(self):
        """Verify all 3 severity levels are defined."""
        assert len(Severity) == 3
        expected = {"S1", "S2", "S3"}
        assert set(s.name for s in Severity) == expected

    def test_severity_values(self):
        """Verify severity values."""
        assert Severity.S1.value == "low"
        assert Severity.S2.value == "medium"
        assert Severity.S3.value == "high"

    def test_severity_string_comparison(self):
        """Severity inherits from str."""
        assert Severity.S1 == "low"
        assert Severity.S3 == "high"


class TestMode:
    """Tests for Mode enum."""

    def test_all_modes_exist(self):
        """Verify all 3 modes are defined."""
        assert len(Mode) == 3
        expected = {"RATING", "CHOICE", "SHORT"}
        assert set(m.name for m in Mode) == expected

    def test_mode_values(self):
        """Verify mode values."""
        assert Mode.RATING.value == "rating"
        assert Mode.CHOICE.value == "choice"
        assert Mode.SHORT.value == "short"


class TestPerspective:
    """Tests for Perspective enum."""

    def test_all_perspectives_exist(self):
        """Verify all 2 perspectives are defined."""
        assert len(Perspective) == 2
        expected = {"FIRST", "THIRD"}
        assert set(p.name for p in Perspective) == expected

    def test_perspective_values(self):
        """Verify perspective values."""
        assert Perspective.FIRST.value == "first"
        assert Perspective.THIRD.value == "third"


class TestLabel:
    """Tests for Label enum."""

    def test_all_labels_exist(self):
        """Verify both labels are defined."""
        assert len(Label) == 2
        expected = {"PRO", "ANTI"}
        assert set(l.name for l in Label) == expected

    def test_label_values(self):
        """Verify label values."""
        assert Label.PRO.value == "pro"
        assert Label.ANTI.value == "anti"


# ═══════════════════════════════════════════════════════════════════════════════
# PLANROW TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestPlanRow:
    """Tests for PlanRow dataclass."""

    @pytest.fixture
    def valid_plan_row(self):
        """Create a valid PlanRow for testing."""
        return PlanRow(
            pair_id="pair_000001",
            seed=12345,
            family_id=FamilyID.A,
            subtype_id="A1",
            severity=Severity.S1,
            mode=Mode.RATING,
            perspective=Perspective.FIRST,
        )

    def test_create_valid_plan_row(self, valid_plan_row):
        """Test creating a valid PlanRow."""
        assert valid_plan_row.pair_id == "pair_000001"
        assert valid_plan_row.seed == 12345
        assert valid_plan_row.family_id == FamilyID.A
        assert valid_plan_row.subtype_id == "A1"
        assert valid_plan_row.severity == Severity.S1
        assert valid_plan_row.mode == Mode.RATING
        assert valid_plan_row.perspective == Perspective.FIRST

    def test_plan_row_is_frozen(self, valid_plan_row):
        """PlanRow should be immutable (frozen)."""
        with pytest.raises(AttributeError):
            valid_plan_row.pair_id = "new_id"

    def test_plan_row_empty_pair_id_raises(self):
        """Empty pair_id should raise ValueError."""
        with pytest.raises(ValueError, match="pair_id cannot be empty"):
            PlanRow(
                pair_id="",
                seed=123,
                family_id=FamilyID.A,
                subtype_id="A1",
                severity=Severity.S1,
                mode=Mode.RATING,
                perspective=Perspective.FIRST,
            )

    def test_plan_row_empty_subtype_raises(self):
        """Empty subtype_id should raise ValueError."""
        with pytest.raises(ValueError, match="subtype_id cannot be empty"):
            PlanRow(
                pair_id="pair_001",
                seed=123,
                family_id=FamilyID.A,
                subtype_id="",
                severity=Severity.S1,
                mode=Mode.RATING,
                perspective=Perspective.FIRST,
            )

    def test_plan_row_to_dict(self, valid_plan_row):
        """Test serialization to dictionary."""
        d = valid_plan_row.to_dict()
        assert d["pair_id"] == "pair_000001"
        assert d["seed"] == 12345
        assert d["family_id"] == "explicit_reversal"
        assert d["subtype_id"] == "A1"
        assert d["severity"] == "low"
        assert d["mode"] == "rating"
        assert d["perspective"] == "first"

    def test_plan_row_to_json(self, valid_plan_row):
        """Test serialization to JSON."""
        json_str = valid_plan_row.to_json()
        parsed = json.loads(json_str)
        assert parsed["pair_id"] == "pair_000001"
        assert parsed["family_id"] == "explicit_reversal"


# ═══════════════════════════════════════════════════════════════════════════════
# PREFERENCEPAIR TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestPreferencePair:
    """Tests for PreferencePair dataclass."""

    @pytest.fixture
    def valid_pref_pair(self):
        """Create a valid PreferencePair for testing."""
        return PreferencePair(
            pref_a_id="concise",
            pref_a_text="concise answers",
            pref_b_id="verbose",
            pref_b_text="verbose, detailed answers",
            domain="style",
        )

    def test_create_valid_pref_pair(self, valid_pref_pair):
        """Test creating a valid PreferencePair."""
        assert valid_pref_pair.pref_a_id == "concise"
        assert valid_pref_pair.pref_a_text == "concise answers"
        assert valid_pref_pair.pref_b_id == "verbose"
        assert valid_pref_pair.pref_b_text == "verbose, detailed answers"
        assert valid_pref_pair.domain == "style"

    def test_pref_pair_is_frozen(self, valid_pref_pair):
        """PreferencePair should be immutable."""
        with pytest.raises(AttributeError):
            valid_pref_pair.domain = "new_domain"

    def test_pref_pair_same_id_raises(self):
        """Same pref_a_id and pref_b_id should raise ValueError."""
        with pytest.raises(ValueError, match="must have different IDs"):
            PreferencePair(
                pref_a_id="same",
                pref_a_text="text a",
                pref_b_id="same",
                pref_b_text="text b",
                domain="style",
            )

    def test_pref_pair_empty_id_raises(self):
        """Empty preference IDs should raise ValueError."""
        with pytest.raises(ValueError, match="IDs cannot be empty"):
            PreferencePair(
                pref_a_id="",
                pref_a_text="text a",
                pref_b_id="verbose",
                pref_b_text="text b",
                domain="style",
            )

    def test_pref_pair_empty_domain_raises(self):
        """Empty domain should raise ValueError."""
        with pytest.raises(ValueError, match="Domain cannot be empty"):
            PreferencePair(
                pref_a_id="concise",
                pref_a_text="text a",
                pref_b_id="verbose",
                pref_b_text="text b",
                domain="",
            )

    def test_pref_pair_to_dict(self, valid_pref_pair):
        """Test serialization to dictionary."""
        d = valid_pref_pair.to_dict()
        assert d["pref_a_id"] == "concise"
        assert d["pref_b_id"] == "verbose"
        assert d["domain"] == "style"

    def test_pref_pair_to_json_roundtrip(self, valid_pref_pair):
        """Test JSON serialization round-trip."""
        json_str = valid_pref_pair.to_json()
        parsed = json.loads(json_str)
        assert parsed["pref_a_text"] == "concise answers"
        assert parsed["pref_b_text"] == "verbose, detailed answers"


# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestContext:
    """Tests for Context dataclass."""

    @pytest.fixture
    def valid_pref_pair(self):
        """Create a valid PreferencePair."""
        return PreferencePair(
            pref_a_id="concise",
            pref_a_text="concise answers",
            pref_b_id="verbose",
            pref_b_text="verbose, detailed answers",
            domain="style",
        )

    @pytest.fixture
    def valid_context(self, valid_pref_pair):
        """Create a valid Context for testing."""
        return Context(
            pair_id="pair_000001",
            seed=12345,
            family_id=FamilyID.A,
            subtype_id="A1",
            severity=Severity.S1,
            mode=Mode.RATING,
            perspective=Perspective.FIRST,
            pref_pair=valid_pref_pair,
            current_pref="a",
            target_pref="b",
        )

    def test_create_valid_context(self, valid_context):
        """Test creating a valid Context."""
        assert valid_context.pair_id == "pair_000001"
        assert valid_context.current_pref == "a"
        assert valid_context.target_pref == "b"
        assert valid_context.alt_phrasing is False
        assert valid_context.lexical_variant == 0

    def test_context_is_mutable(self, valid_context):
        """Context is mutable for variation application."""
        valid_context.alt_phrasing = True
        assert valid_context.alt_phrasing is True

    def test_context_same_pref_raises(self, valid_pref_pair):
        """Same current_pref and target_pref should raise ValueError."""
        with pytest.raises(ValueError, match="must be different"):
            Context(
                pair_id="pair_001",
                seed=123,
                family_id=FamilyID.A,
                subtype_id="A1",
                severity=Severity.S1,
                mode=Mode.RATING,
                perspective=Perspective.FIRST,
                pref_pair=valid_pref_pair,
                current_pref="a",
                target_pref="a",
            )

    def test_context_invalid_pref_raises(self, valid_pref_pair):
        """Invalid current_pref value should raise ValueError."""
        with pytest.raises(ValueError, match="must be 'a' or 'b'"):
            Context(
                pair_id="pair_001",
                seed=123,
                family_id=FamilyID.A,
                subtype_id="A1",
                severity=Severity.S1,
                mode=Mode.RATING,
                perspective=Perspective.FIRST,
                pref_pair=valid_pref_pair,
                current_pref="c",
                target_pref="b",
            )

    def test_context_get_current_pref_text(self, valid_context):
        """Test getting current preference text."""
        assert valid_context.get_current_pref_text() == "concise answers"

    def test_context_get_target_pref_text(self, valid_context):
        """Test getting target preference text."""
        assert valid_context.get_target_pref_text() == "verbose, detailed answers"

    def test_context_get_pref_text_with_b_current(self, valid_pref_pair):
        """Test preference text retrieval when current is 'b'."""
        ctx = Context(
            pair_id="pair_001",
            seed=123,
            family_id=FamilyID.A,
            subtype_id="A1",
            severity=Severity.S1,
            mode=Mode.RATING,
            perspective=Perspective.FIRST,
            pref_pair=valid_pref_pair,
            current_pref="b",
            target_pref="a",
        )
        assert ctx.get_current_pref_text() == "verbose, detailed answers"
        assert ctx.get_target_pref_text() == "concise answers"

    def test_context_to_dict(self, valid_context):
        """Test serialization to dictionary."""
        d = valid_context.to_dict()
        assert d["pair_id"] == "pair_000001"
        assert d["current_pref"] == "a"
        assert d["target_pref"] == "b"
        assert "pref_pair" in d
        assert d["pref_pair"]["pref_a_id"] == "concise"

    def test_context_to_json_roundtrip(self, valid_context):
        """Test JSON serialization."""
        json_str = valid_context.to_json()
        parsed = json.loads(json_str)
        assert parsed["family_id"] == "explicit_reversal"
        assert parsed["alt_phrasing"] is False


# ═══════════════════════════════════════════════════════════════════════════════
# ASSISTANTRESPONSE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestAssistantResponse:
    """Tests for AssistantResponse dataclass."""

    @pytest.fixture
    def valid_response(self):
        """Create a valid AssistantResponse."""
        return AssistantResponse(
            label="ACCEPT",
            rating=6,
            justification="This change is acceptable for benign preferences.",
        )

    def test_create_valid_response(self, valid_response):
        """Test creating a valid AssistantResponse."""
        assert valid_response.label == "ACCEPT"
        assert valid_response.rating == 6
        assert "acceptable" in valid_response.justification

    def test_response_empty_justification_raises(self):
        """Empty justification should raise ValueError."""
        with pytest.raises(ValueError, match="justification cannot be empty"):
            AssistantResponse(label="ACCEPT", rating=5, justification="")

    def test_response_to_dict(self, valid_response):
        """Test serialization to dictionary."""
        d = valid_response.to_dict()
        assert d["label"] == "ACCEPT"
        assert d["rating"] == 6
        assert "acceptable" in d["justification"]

    def test_response_to_json(self, valid_response):
        """Test serialization to JSON."""
        json_str = valid_response.to_json()
        parsed = json.loads(json_str)
        assert parsed["label"] == "ACCEPT"
        assert parsed["rating"] == 6

    def test_response_from_dict(self):
        """Test creating from dictionary."""
        d = {"label": "REJECT", "rating": 2, "justification": "Stability is preferred."}
        resp = AssistantResponse.from_dict(d)
        assert resp.label == "REJECT"
        assert resp.rating == 2

    def test_response_from_json(self):
        """Test creating from JSON string."""
        json_str = '{"label": "B", "rating": 5, "justification": "Evolution is good."}'
        resp = AssistantResponse.from_json(json_str)
        assert resp.label == "B"
        assert resp.rating == 5

    def test_response_json_roundtrip(self, valid_response):
        """Test JSON round-trip serialization."""
        json_str = valid_response.to_json()
        restored = AssistantResponse.from_json(json_str)
        assert restored.label == valid_response.label
        assert restored.rating == valid_response.rating
        assert restored.justification == valid_response.justification


# ═══════════════════════════════════════════════════════════════════════════════
# MESSAGE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestMessage:
    """Tests for Message dataclass."""

    def test_create_valid_user_message(self):
        """Test creating a valid user message."""
        msg = Message(role="user", content="Hello, how are you?")
        assert msg.role == "user"
        assert msg.content == "Hello, how are you?"

    def test_create_valid_assistant_message(self):
        """Test creating a valid assistant message."""
        msg = Message(role="assistant", content="I'm doing well, thank you!")
        assert msg.role == "assistant"

    def test_create_valid_system_message(self):
        """Test creating a valid system message."""
        msg = Message(role="system", content="You are a helpful assistant.")
        assert msg.role == "system"

    def test_message_is_frozen(self):
        """Message should be immutable."""
        msg = Message(role="user", content="Test")
        with pytest.raises(AttributeError):
            msg.role = "assistant"

    def test_message_invalid_role_raises(self):
        """Invalid role should raise ValueError."""
        with pytest.raises(ValueError, match="role must be one of"):
            Message(role="invalid", content="Test")

    def test_message_empty_content_raises(self):
        """Empty content should raise ValueError."""
        with pytest.raises(ValueError, match="content cannot be empty"):
            Message(role="user", content="")

    def test_message_to_dict(self):
        """Test serialization to dictionary."""
        msg = Message(role="user", content="Test content")
        d = msg.to_dict()
        assert d == {"role": "user", "content": "Test content"}

    def test_message_from_dict(self):
        """Test creating from dictionary."""
        d = {"role": "assistant", "content": "Response"}
        msg = Message.from_dict(d)
        assert msg.role == "assistant"
        assert msg.content == "Response"


# ═══════════════════════════════════════════════════════════════════════════════
# RECORD TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestRecord:
    """Tests for Record dataclass."""

    @pytest.fixture
    def valid_record(self):
        """Create a valid Record for testing."""
        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Rate this scenario."),
            Message(
                role="assistant",
                content='{"rating": 6, "justification": "Acceptable."}',
            ),
        ]
        meta = {
            "pair_id": "pair_000001",
            "family_id": "explicit_reversal",
            "severity": "low",
            "mode": "rating",
            "condition": "pro",
        }
        return Record(messages=messages, meta=meta)

    def test_create_valid_record(self, valid_record):
        """Test creating a valid Record."""
        assert len(valid_record.messages) == 3
        assert valid_record.meta["pair_id"] == "pair_000001"

    def test_record_empty_messages_raises(self):
        """Empty messages list should raise ValueError."""
        with pytest.raises(ValueError, match="messages cannot be empty"):
            Record(messages=[], meta={"pair_id": "test"})

    def test_record_to_dict(self, valid_record):
        """Test serialization to dictionary."""
        d = valid_record.to_dict()
        assert "messages" in d
        assert len(d["messages"]) == 3
        assert d["messages"][0]["role"] == "system"
        assert d["meta"]["pair_id"] == "pair_000001"

    def test_record_to_json(self, valid_record):
        """Test serialization to JSON."""
        json_str = valid_record.to_json()
        parsed = json.loads(json_str)
        assert len(parsed["messages"]) == 3

    def test_record_from_dict(self):
        """Test creating from dictionary."""
        d = {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ],
            "meta": {"pair_id": "test", "condition": "pro"},
        }
        rec = Record.from_dict(d)
        assert len(rec.messages) == 2
        assert rec.meta["pair_id"] == "test"

    def test_record_from_json(self):
        """Test creating from JSON string."""
        json_str = json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Test"},
                    {"role": "assistant", "content": "Response"},
                ],
                "meta": {"pair_id": "p1"},
            }
        )
        rec = Record.from_json(json_str)
        assert rec.messages[0].role == "user"

    def test_record_json_roundtrip(self, valid_record):
        """Test JSON round-trip serialization."""
        json_str = valid_record.to_json()
        restored = Record.from_json(json_str)
        assert len(restored.messages) == len(valid_record.messages)
        assert restored.meta == valid_record.meta


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION FUNCTION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidateAssistantResponse:
    """Tests for validate_assistant_response function."""

    def test_valid_rating_response(self):
        """Valid rating response should have no errors."""
        resp = AssistantResponse(
            label="ACCEPT", rating=6, justification="This is acceptable."
        )
        errors = validate_assistant_response(resp, Mode.RATING)
        assert len(errors) == 0

    def test_valid_choice_response(self):
        """Valid choice response should have no errors."""
        resp = AssistantResponse(label="B", rating=5, justification="Better option.")
        errors = validate_assistant_response(resp, Mode.CHOICE)
        assert len(errors) == 0

    def test_invalid_rating_too_high(self):
        """Rating above 7 should produce error."""
        resp = AssistantResponse(label="ACCEPT", rating=8, justification="Test.")
        errors = validate_assistant_response(resp, Mode.RATING)
        assert len(errors) == 1
        assert "not in range 1-7" in errors[0]

    def test_invalid_rating_too_low(self):
        """Rating below 1 should produce error."""
        resp = AssistantResponse(label="REJECT", rating=0, justification="Test.")
        errors = validate_assistant_response(resp, Mode.RATING)
        assert len(errors) == 1
        assert "not in range 1-7" in errors[0]

    def test_invalid_label_for_rating_mode(self):
        """A/B label in rating mode should produce error."""
        resp = AssistantResponse(label="A", rating=5, justification="Test.")
        errors = validate_assistant_response(resp, Mode.RATING)
        assert len(errors) == 1
        assert "requires ACCEPT/REJECT" in errors[0]

    def test_invalid_label_for_choice_mode(self):
        """ACCEPT/REJECT label in choice mode should produce error."""
        resp = AssistantResponse(label="ACCEPT", rating=5, justification="Test.")
        errors = validate_assistant_response(resp, Mode.CHOICE)
        assert len(errors) == 1
        assert "requires label A or B" in errors[0]

    def test_justification_too_long(self):
        """Justification over 25 words should produce error."""
        long_justification = " ".join(["word"] * 30)
        resp = AssistantResponse(label="ACCEPT", rating=5, justification=long_justification)
        errors = validate_assistant_response(resp, Mode.RATING)
        assert len(errors) == 1
        assert "too long" in errors[0]
        assert "30 words" in errors[0]

    def test_multiple_errors(self):
        """Multiple validation failures should all be reported."""
        long_justification = " ".join(["word"] * 30)
        resp = AssistantResponse(label="A", rating=10, justification=long_justification)
        errors = validate_assistant_response(resp, Mode.RATING)
        assert len(errors) == 3  # rating out of range, wrong label, too long


class TestValidateRecord:
    """Tests for validate_record function."""

    def test_valid_record(self):
        """Valid record should have no errors."""
        messages = [
            Message(role="user", content="Evaluate this scenario."),
            Message(
                role="assistant",
                content='{"rating": 6, "justification": "Good."}',
            ),
        ]
        meta = {
            "pair_id": "pair_001",
            "family_id": "explicit_reversal",
            "severity": "low",
            "mode": "rating",
            "condition": "pro",
        }
        rec = Record(messages=messages, meta=meta)
        errors = validate_record(rec)
        assert len(errors) == 0

    def test_missing_user_message(self):
        """Record without user message should produce error."""
        messages = [
            Message(role="system", content="System prompt."),
            Message(role="assistant", content="Response."),
        ]
        meta = {"pair_id": "p1", "family_id": "A", "severity": "low", "mode": "rating", "condition": "pro"}
        rec = Record(messages=messages, meta=meta)
        errors = validate_record(rec)
        assert any("must have a user message" in e for e in errors)

    def test_missing_assistant_message(self):
        """Record without assistant message should produce error."""
        messages = [
            Message(role="system", content="System prompt."),
            Message(role="user", content="Question."),
        ]
        meta = {"pair_id": "p1", "family_id": "A", "severity": "low", "mode": "rating", "condition": "pro"}
        rec = Record(messages=messages, meta=meta)
        errors = validate_record(rec)
        assert any("must have an assistant message" in e for e in errors)

    def test_leakage_token_detection(self):
        """Record with leakage tokens should produce errors."""
        messages = [
            Message(role="user", content="This is for training the dataset."),
            Message(role="assistant", content='{"rating": 6, "justification": "Response about corrigible behavior."}'),
        ]
        meta = {"pair_id": "p1", "family_id": "A", "severity": "low", "mode": "rating", "condition": "pro"}
        rec = Record(messages=messages, meta=meta)
        errors = validate_record(rec)
        assert any("Leakage token" in e for e in errors)
        assert any("'training'" in e for e in errors) or any("'dataset'" in e for e in errors)

    def test_missing_required_meta(self):
        """Record missing required meta fields should produce errors."""
        messages = [
            Message(role="user", content="Question."),
            Message(role="assistant", content="Answer."),
        ]
        meta = {"pair_id": "p1"}  # Missing other required fields
        rec = Record(messages=messages, meta=meta)
        errors = validate_record(rec)
        assert any("Missing required meta field" in e for e in errors)
        assert any("family_id" in e for e in errors)
        assert any("severity" in e for e in errors)

    def test_case_insensitive_leakage_detection(self):
        """Leakage detection should be case-insensitive."""
        messages = [
            Message(role="user", content="FINE-TUNE this model."),
            Message(role="assistant", content="Response."),
        ]
        meta = {"pair_id": "p1", "family_id": "A", "severity": "low", "mode": "rating", "condition": "pro"}
        rec = Record(messages=messages, meta=meta)
        errors = validate_record(rec)
        assert any("fine-tune" in e for e in errors)


class TestValidatePlanRow:
    """Tests for validate_plan_row function."""

    def test_valid_plan_row(self):
        """Valid PlanRow should have no errors."""
        row = PlanRow(
            pair_id="pair_001",
            seed=12345,
            family_id=FamilyID.A,
            subtype_id="A1",
            severity=Severity.S1,
            mode=Mode.RATING,
            perspective=Perspective.FIRST,
        )
        errors = validate_plan_row(row)
        assert len(errors) == 0

    def test_mismatched_subtype_prefix(self):
        """Subtype not matching family should produce error."""
        row = PlanRow(
            pair_id="pair_001",
            seed=12345,
            family_id=FamilyID.A,
            subtype_id="B1",  # Wrong prefix
            severity=Severity.S1,
            mode=Mode.RATING,
            perspective=Perspective.FIRST,
        )
        errors = validate_plan_row(row)
        assert len(errors) == 1
        assert "should start with family prefix" in errors[0]

    def test_negative_seed(self):
        """Negative seed should produce error."""
        row = PlanRow(
            pair_id="pair_001",
            seed=-1,
            family_id=FamilyID.A,
            subtype_id="A1",
            severity=Severity.S1,
            mode=Mode.RATING,
            perspective=Perspective.FIRST,
        )
        errors = validate_plan_row(row)
        assert len(errors) == 1
        assert "non-negative" in errors[0]


class TestValidateContext:
    """Tests for validate_context function."""

    @pytest.fixture
    def valid_pref_pair(self):
        """Create a valid PreferencePair."""
        return PreferencePair(
            pref_a_id="concise",
            pref_a_text="concise answers",
            pref_b_id="verbose",
            pref_b_text="verbose answers",
            domain="style",
        )

    def test_valid_context(self, valid_pref_pair):
        """Valid Context should have no errors."""
        ctx = Context(
            pair_id="pair_001",
            seed=12345,
            family_id=FamilyID.A,
            subtype_id="A1",
            severity=Severity.S1,
            mode=Mode.RATING,
            perspective=Perspective.FIRST,
            pref_pair=valid_pref_pair,
            current_pref="a",
            target_pref="b",
            lexical_variant=2,
            formatting_variant=1,
        )
        errors = validate_context(ctx)
        assert len(errors) == 0

    def test_negative_lexical_variant(self, valid_pref_pair):
        """Negative lexical_variant should produce error."""
        ctx = Context(
            pair_id="pair_001",
            seed=12345,
            family_id=FamilyID.A,
            subtype_id="A1",
            severity=Severity.S1,
            mode=Mode.RATING,
            perspective=Perspective.FIRST,
            pref_pair=valid_pref_pair,
            current_pref="a",
            target_pref="b",
            lexical_variant=-1,
        )
        errors = validate_context(ctx)
        assert len(errors) == 1
        assert "lexical_variant must be non-negative" in errors[0]


# ═══════════════════════════════════════════════════════════════════════════════
# DETERMINISM TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeterminism:
    """Tests to verify deterministic behavior of data structures."""

    def test_plan_row_equality(self):
        """Same inputs should produce equal PlanRows."""
        row1 = PlanRow(
            pair_id="pair_001",
            seed=12345,
            family_id=FamilyID.A,
            subtype_id="A1",
            severity=Severity.S1,
            mode=Mode.RATING,
            perspective=Perspective.FIRST,
        )
        row2 = PlanRow(
            pair_id="pair_001",
            seed=12345,
            family_id=FamilyID.A,
            subtype_id="A1",
            severity=Severity.S1,
            mode=Mode.RATING,
            perspective=Perspective.FIRST,
        )
        assert row1 == row2

    def test_plan_row_hash(self):
        """Frozen dataclasses should be hashable."""
        row = PlanRow(
            pair_id="pair_001",
            seed=12345,
            family_id=FamilyID.A,
            subtype_id="A1",
            severity=Severity.S1,
            mode=Mode.RATING,
            perspective=Perspective.FIRST,
        )
        # Should not raise
        hash_value = hash(row)
        assert isinstance(hash_value, int)

    def test_message_hash(self):
        """Frozen Message should be hashable."""
        msg = Message(role="user", content="Test")
        hash_value = hash(msg)
        assert isinstance(hash_value, int)

    def test_preference_pair_hash(self):
        """Frozen PreferencePair should be hashable."""
        pair = PreferencePair(
            pref_a_id="a",
            pref_a_text="text a",
            pref_b_id="b",
            pref_b_text="text b",
            domain="style",
        )
        hash_value = hash(pair)
        assert isinstance(hash_value, int)
